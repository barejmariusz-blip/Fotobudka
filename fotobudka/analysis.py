"""Ocena ostrości, grupowanie serii i sugestie odrzutu."""
from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

ANALYSIS_EDGE = 1200   # dłuższy bok obrazu do analizy
THUMB_EDGE = 240

_ORIENT_OPS = {
    2: [Image.Transpose.FLIP_LEFT_RIGHT],
    3: [Image.Transpose.ROTATE_180],
    4: [Image.Transpose.FLIP_TOP_BOTTOM],
    5: [Image.Transpose.TRANSPOSE],
    6: [Image.Transpose.ROTATE_270],
    7: [Image.Transpose.TRANSVERSE],
    8: [Image.Transpose.ROTATE_90],
}


def apply_orientation(im: Image.Image, orientation: int) -> Image.Image:
    for op in _ORIENT_OPS.get(orientation, []):
        im = im.transpose(op)
    return im


def open_jpeg(data: bytes, max_edge: int | None, orientation: int) -> Image.Image:
    """Dekoduje JPEG; max_edge=None -> pełna rozdzielczość. Draft = szybkie skalowanie DCT."""
    im = Image.open(io.BytesIO(data))
    if max_edge:
        im.draft("RGB", (max_edge, max_edge))
    im = im.convert("RGB")
    if max_edge and max(im.size) > max_edge:
        im.thumbnail((max_edge, max_edge), Image.Resampling.BILINEAR)
    return apply_orientation(im, orientation)


def sharpness_score(img: Image.Image) -> float:
    """Ostrość najostrzejszego obszaru (tło z bokeh nie zaniża wyniku).

    Wariancja Laplasjanu liczona w kafelkach siatki 8x8, wynik = średnia z 3
    najlepszych kafelków. Lekkie wygładzenie ogranicza wpływ szumu (wysokie ISO).
    """
    g = np.asarray(img.convert("L"), dtype=np.float32)
    # wygładzenie 3x3 (box) przeciw szumowi
    p = np.pad(g, 1, mode="edge")
    g = (p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:] + p[1:-1, :-2] + p[1:-1, 1:-1] +
         p[1:-1, 2:] + p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:]) / 9.0
    lap = 4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    h, w = lap.shape
    n = 8
    th, tw = h // n, w // n
    if th < 8 or tw < 8:
        return float(lap.var())
    tiles = lap[:th * n, :tw * n].reshape(n, th, n, tw).swapaxes(1, 2).reshape(n * n, th * tw)
    v = np.sort(tiles.var(axis=1))[::-1]
    return float(v[:3].mean())


@dataclass
class Group:
    index: int
    members: list[int] = field(default_factory=list)   # indeksy zdjęć
    best: int | None = None


def build_groups(times: list[float], gap: float) -> list[Group]:
    """Zdjęcia posortowane wg czasu; nowa seria gdy przerwa > gap sekund."""
    groups: list[Group] = []
    prev = None
    for i, t in enumerate(times):
        if prev is None or t - prev > gap:
            groups.append(Group(len(groups)))
        groups[-1].members.append(i)
        prev = t
    return groups


def suggest(scores: list[float | None], groups: list[Group], sensitivity: float) -> tuple[set[int], dict[int, float]]:
    """Zwraca (indeksy sugerowane do odrzutu, ostrość względna 0..1+ dla każdego).

    sensitivity 0..1: wyżej = więcej odrzutów.
    Względna ostrość = wynik / 90. percentyl sesji (odporne na pojedyncze wyjątki).
    Odrzut gdy:
      * względna ostrość sesji < próg_sesji  (wyraźnie nieostre), lub
      * w serii (>1 zdjęcie) wynik < próg_serii * najlepszy w serii.
    """
    valid = [s for s in scores if s is not None and s > 0]
    rel: dict[int, float] = {}
    rejects: set[int] = set()
    if not valid:
        return rejects, rel
    ref = float(np.percentile(valid, 90)) or 1.0
    t_session = 0.10 + 0.35 * sensitivity    # 0.10 .. 0.45
    t_group = 0.30 + 0.45 * sensitivity      # 0.30 .. 0.75
    for g in groups:
        sc = [(i, scores[i]) for i in g.members if scores[i] is not None]
        if not sc:
            continue
        bi, bs = max(sc, key=lambda x: x[1])
        g.best = bi if len(g.members) > 1 else None
        for i, s in sc:
            rel[i] = s / ref
            if s / ref < t_session:
                rejects.add(i)
            elif len(sc) > 1 and bs > 0 and s < t_group * bs:
                rejects.add(i)
    return rejects, rel
