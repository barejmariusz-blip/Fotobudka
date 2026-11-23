"""Wyciąganie podglądu JPEG i metadanych (czas, orientacja) z plików RAW/JPG.

Obsługa:
  * CR3 (Canon R6 itd.) - parsowanie ISO-BMFF: trak #1 = pełnowymiarowy JPEG,
    metadane z boksów CMT1 (IFD0) i CMT2 (EXIF).
  * CR2 / NEF / ARW / DNG / PEF / TIF - parsowanie TIFF, największy osadzony JPEG.
  * JPG / JPEG - plik wprost.
  * Awaryjnie: skanowanie pliku w poszukiwaniu największego strumienia JPEG.
"""
from __future__ import annotations

import mmap
import os
import struct
from dataclasses import dataclass
from datetime import datetime

RAW_EXT = {".cr3", ".cr2", ".nef", ".arw", ".dng", ".pef", ".tif", ".tiff", ".raf", ".rw2", ".orf"}
JPG_EXT = {".jpg", ".jpeg"}
SUPPORTED_EXT = RAW_EXT | JPG_EXT


@dataclass
class PhotoMeta:
    capture_time: float | None = None   # sekundy (z ułamkiem) od epoki
    orientation: int = 1                 # EXIF 1..8
    preview_offset: int = 0              # offset JPEG w pliku
    preview_length: int = 0              # długość JPEG (0 = cały plik)


# --------------------------------------------------------------------------- TIFF
_TYPE_SIZE = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8, 13: 4}


def _read_ifd(buf, base: int, off: int, e: str) -> tuple[dict, int]:
    """Zwraca (tagi, offset następnego IFD). Offsety względem `base`."""
    tags: dict[int, object] = {}
    p = base + off
    if p + 2 > len(buf):
        return tags, 0
    (n,) = struct.unpack_from(e + "H", buf, p)
    if n > 1000:
        return tags, 0
    p += 2
    for i in range(n):
        q = p + i * 12
        if q + 12 > len(buf):
            break
        tag, typ, cnt = struct.unpack_from(e + "HHI", buf, q)
        size = _TYPE_SIZE.get(typ, 1) * cnt
        vp = q + 8 if size <= 4 else base + struct.unpack_from(e + "I", buf, q + 8)[0]
        if vp + size > len(buf) or cnt > 100000:
            continue
        if typ == 2:
            tags[tag] = bytes(buf[vp:vp + cnt]).split(b"\0", 1)[0].decode("latin-1", "ignore")
        elif typ == 3:
            tags[tag] = list(struct.unpack_from(e + "%dH" % cnt, buf, vp))
        elif typ in (4, 13):
            tags[tag] = list(struct.unpack_from(e + "%dI" % cnt, buf, vp))
        elif typ == 7:
            tags[tag] = bytes(buf[vp:vp + cnt])
        else:
            tags[tag] = None
    nxt_p = p + n * 12
    nxt = struct.unpack_from(e + "I", buf, nxt_p)[0] if nxt_p + 4 <= len(buf) else 0
    return tags, nxt


def _tiff_endian(buf, base: int) -> str | None:
    hdr = bytes(buf[base:base + 4])
    if hdr[:2] == b"II":
        return "<"
    if hdr[:2] == b"MM":
        return ">"
    return None


def _parse_dt(s, subsec=None) -> float | None:
    if not isinstance(s, str) or len(s) < 19:
        return None
    try:
        t = datetime.strptime(s[:19], "%Y:%m:%d %H:%M:%S").timestamp()
    except ValueError:
        return None
    if isinstance(subsec, str) and subsec.strip().isdigit():
        ss = subsec.strip()
        t += int(ss) / (10 ** len(ss))
    return t


def _first(v, default=None):
    if isinstance(v, list) and v:
        return v[0]
    return default


def _tiff_block_meta(buf, base: int) -> dict:
    """Czyta IFD0 (+ EXIF IFD) z bloku TIFF zaczynającego się w `base`."""
    e = _tiff_endian(buf, base)
    if not e:
        return {}
    off = struct.unpack_from(e + "I", buf, base + 4)[0]
    tags, _ = _read_ifd(buf, base, off, e)
    out = dict(tags)
    exif_off = _first(tags.get(0x8769))
    if exif_off:
        exif, _ = _read_ifd(buf, base, exif_off, e)
        out.update(exif)
    return out


def _scan_tiff_previews(buf) -> tuple[PhotoMeta, list[tuple[int, int]]]:
    meta = PhotoMeta()
    e = _tiff_endian(buf, 0)
    if not e:
        return meta, []
    cands: list[tuple[int, int]] = []
    seen = set()
    stack = [struct.unpack_from(e + "I", buf, 4)[0]]
    first = True
    while stack:
        off = stack.pop()
        if not off or off in seen or off >= len(buf) or len(seen) > 50:
            continue
        seen.add(off)
        tags, nxt = _read_ifd(buf, 0, off, e)
        if first:
            first = False
            meta.orientation = _first(tags.get(0x0112), 1) or 1
            exif_off = _first(tags.get(0x8769))
            if exif_off:
                ex, _ = _read_ifd(buf, 0, exif_off, e)
                meta.capture_time = _parse_dt(ex.get(0x9003), ex.get(0x9291))
            if meta.capture_time is None:
                meta.capture_time = _parse_dt(tags.get(0x0132))
        jo, jl = _first(tags.get(0x201)), _first(tags.get(0x202))
        if jo and jl:
            cands.append((jo, jl))
        comp = _first(tags.get(0x103))
        so, sl = tags.get(0x111), tags.get(0x117)
        if comp in (6, 7) and so and sl and len(so) == 1:
            cands.append((so[0], sl[0]))
        for sub in tags.get(0x14A) or []:
            stack.append(sub)
        if nxt:
            stack.append(nxt)
    valid = [(o, l) for o, l in cands if o + l <= len(buf) and bytes(buf[o:o + 2]) == b"\xff\xd8"]
    return meta, valid


# --------------------------------------------------------------------------- CR3
def _boxes(buf, start: int, end: int):
    p = start
    while p + 8 <= end:
        size, typ = struct.unpack_from(">I4s", buf, p)
        hdr = 8
        if size == 1:
            size = struct.unpack_from(">Q", buf, p + 8)[0]
            hdr = 16
        elif size == 0:
            size = end - p
        if size < hdr or p + size > end:
            return
        yield typ, p + hdr, p + size
        p += size


def _cr3(buf) -> tuple[PhotoMeta, list[tuple[int, int]]]:
    meta = PhotoMeta()
    cands = []
    moov = next(((s, e) for t, s, e in _boxes(buf, 0, len(buf)) if t == b"moov"), None)
    if not moov:
        return meta, cands
    ms, me = moov
    region = bytes(buf[ms:me])

    def cmt(name: bytes):
        i = region.find(name)
        if i < 4:
            return None
        size = struct.unpack_from(">I", region, i - 4)[0]
        return region[i + 4:i - 4 + size]

    c1, c2 = cmt(b"CMT1"), cmt(b"CMT2")
    if c1:
        t1 = _tiff_block_meta(c1, 0)
        meta.orientation = _first(t1.get(0x0112), 1) or 1
        meta.capture_time = _parse_dt(t1.get(0x0132))
    if c2:
        e = _tiff_endian(c2, 0)
        if e:
            off = struct.unpack_from(e + "I", c2, 4)[0]
            t2, _ = _read_ifd(c2, 0, off, e)
            ct = _parse_dt(t2.get(0x9003), t2.get(0x9291))
            if ct:
                meta.capture_time = ct

    # trak #1 -> mdia/minf/stbl -> stsz + co64/stco
    for t, s, e in _boxes(buf, ms, me):
        if t != b"trak":
            continue
        stbl = _find_path(buf, s, e, [b"mdia", b"minf", b"stbl"])
        if stbl:
            size = offset = None
            for t2, s2, e2 in _boxes(buf, *stbl):
                if t2 == b"stsz":
                    ss, cnt = struct.unpack_from(">II", buf, s2 + 4)
                    size = ss if ss else (struct.unpack_from(">I", buf, s2 + 12)[0] if cnt else None)
                elif t2 == b"co64":
                    offset = struct.unpack_from(">Q", buf, s2 + 8)[0]
                elif t2 == b"stco":
                    offset = struct.unpack_from(">I", buf, s2 + 8)[0]
            if size and offset and offset + size <= len(buf) and bytes(buf[offset:offset + 2]) == b"\xff\xd8":
                cands.append((offset, size))
        break  # tylko pierwsza ścieżka = pełny JPEG
    # PRVW (1620x1080) jako zapas
    for t, s, e in _boxes(buf, 0, len(buf)):
        if t == b"uuid":
            i = bytes(buf[s:min(e, s + 64)]).find(b"PRVW")
            if i >= 0:
                j = s + i
                k = bytes(buf[j:j + 64]).find(b"\xff\xd8")
                if k >= 0:
                    psize = struct.unpack_from(">I", buf, j - 4)[0]
                    jo = j + k
                    cands.append((jo, max(0, j - 4 + psize - jo)))
    return meta, cands


def _find_path(buf, s, e, path):
    for name in path:
        nxt = next(((s2, e2) for t, s2, e2 in _boxes(buf, s, e) if t == name), None)
        if not nxt:
            return None
        s, e = nxt
    return s, e


# --------------------------------------------------------------------------- awaryjnie
def _jpeg_len(buf, start: int) -> int:
    """Długość strumienia JPEG od SOI (parsowanie markerów)."""
    p = start + 2
    n = len(buf)
    while p + 4 <= n:
        if buf[p] != 0xFF:
            return 0
        m = buf[p + 1]
        if m == 0xFF:
            p += 1
            continue
        if m in (0xD8, 0x01) or 0xD0 <= m <= 0xD7:
            p += 2
            continue
        seg = struct.unpack_from(">H", buf, p + 2)[0]
        if m == 0xDA:  # SOS -> szukaj EOI
            q = buf.find(b"\xff\xd9", p + 2 + seg)
            return (q + 2 - start) if q >= 0 else 0
        p += 2 + seg
    return 0


def _scan_jpegs(buf, limit=40) -> list[tuple[int, int]]:
    out = []
    p = 0
    while len(out) < limit:
        p = buf.find(b"\xff\xd8\xff", p)
        if p < 0:
            break
        ln = _jpeg_len(buf, p)
        if ln > 1024:
            out.append((p, ln))
            p += ln
        else:
            p += 3
    return out


# --------------------------------------------------------------------------- API
def read_meta(path: str) -> PhotoMeta:
    ext = os.path.splitext(path)[1].lower()
    if ext in JPG_EXT:
        meta = PhotoMeta()
        try:
            from PIL import Image
            with Image.open(path) as im:
                ex = im.getexif()
                meta.orientation = ex.get(0x0112, 1) or 1
                sub = ex.get_ifd(0x8769)
                meta.capture_time = _parse_dt(sub.get(0x9003), sub.get(0x9291)) or _parse_dt(ex.get(0x0132))
        except Exception:
            pass
        if meta.capture_time is None:
            meta.capture_time = os.path.getmtime(path)
        return meta

    with open(path, "rb") as f:
        buf = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            if ext == ".cr3" or bytes(buf[4:12]) == b"ftypcrx ":
                meta, cands = _cr3(buf)
            else:
                meta, cands = _scan_tiff_previews(buf)
            if not cands:
                cands = _scan_jpegs(buf)
        finally:
            buf.close()
    if cands:
        o, l = max(cands, key=lambda c: c[1])
        meta.preview_offset, meta.preview_length = o, l
    if meta.capture_time is None:
        meta.capture_time = os.path.getmtime(path)
    if meta.orientation not in range(1, 9):
        meta.orientation = 1
    return meta


def read_preview_bytes(path: str, meta: PhotoMeta) -> bytes:
    with open(path, "rb") as f:
        if meta.preview_length <= 0:
            return f.read()
        f.seek(meta.preview_offset)
        return f.read(meta.preview_length)
