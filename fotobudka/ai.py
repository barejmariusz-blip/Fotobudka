"""Rozpoznawanie zwierząt (AI) - wszystko liczone lokalnie, modele pobierane jeden raz.

Modele (ONNX, razem ok. 200 MB, w %LOCALAPPDATA%/PanRodla/Fotobudka/models):
  * YOLOv8n  - wykrywa zwierzę (pies, kot, koń, ptak...) i osobę w kadrze,
  * DINOv2-small - „odcisk wyglądu” konkretnego zwierzęcia / osoby (umaszczenie, łatki, sylwetka, włosy),
  * CLIP ViT-B/32 - nazwa rasy / gatunku (podpowiedź do nazwy grupy).

Grupowanie:
  1. zdjęcia łączone wg wyglądu ZWIERZĘCIA (DINOv2 na wycinku zwierzęcia), łączenie średnie (average linkage),
  2. zdjęcia w tej samej serii zdjęć dostają to samo zwierzę (głosowanie) - poprawia pojedyncze pomyłki,
  3. grupy zwierząt z tą samą OSOBĄ (DINOv2 na wycinku osoby) łączone w „właściciela”.
"""
from __future__ import annotations

import base64
import hashlib
import os
import sys
import threading
import urllib.request

import numpy as np
from PIL import Image

from . import analysis

try:
    from ._model import MODELS
except ImportError:  # wersja deweloperska
    MODELS = {}

AI_VERSION = 3
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
IMNET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMNET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# klasy COCO
PERSON = 0
ANIMALS = {14: "ptak", 15: "kot", 16: "pies", 17: "koń", 18: "owca", 19: "krowa", 21: "niedźwiedź"}

_lock = threading.Lock()
_sessions: dict = {}
_labels = None


# --------------------------------------------------------------------------- pliki modeli
def data_dir() -> str:
    base = os.environ.get("LOCALAPPDATA") or os.path.join(os.path.expanduser("~"), ".local", "share")
    d = os.path.join(base, "PanRodla", "Fotobudka", "models")
    os.makedirs(d, exist_ok=True)
    return d


def model_path(key: str) -> str:
    return os.path.join(data_dir(), MODELS[key]["file"] if key in MODELS else f"{key}.onnx")


def available() -> bool:
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    return bool(MODELS)


def missing(group: str = "core") -> list[str]:
    out = []
    for k, m in MODELS.items():
        if m.get("group", "core") != group:
            continue
        p = model_path(k)
        if not (os.path.exists(p) and os.path.getsize(p) == m["size"]):
            out.append(k)
    return out


def model_ready(group: str = "core") -> bool:
    return available() and any(m.get("group", "core") == group for m in MODELS.values()) and not missing(group)


def download_size(group: str = "core") -> int:
    return sum(MODELS[k]["size"] for k in missing(group))


def download_models(progress=None, cancelled=lambda: False, group: str = "core") -> None:
    keys = missing(group)
    total = sum(MODELS[k]["size"] for k in keys)
    done = 0
    for k in keys:
        m = MODELS[k]
        dst = model_path(k)
        tmp = dst + ".part"
        h = hashlib.sha256()
        req = urllib.request.Request(m["url"], headers={"User-Agent": "Fotobudka (PanRodla)"})
        with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
            while True:
                if cancelled():
                    raise RuntimeError("Przerwano")
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                h.update(chunk)
                done += len(chunk)
                if progress:
                    progress(done, total)
        if h.hexdigest() != m["sha256"]:
            os.remove(tmp)
            raise RuntimeError(f"Pobrany model {k} jest uszkodzony (zła suma kontrolna) - spróbuj ponownie.")
        os.replace(tmp, dst)


def _resource(rel: str) -> str:
    base = getattr(sys, "_MEIPASS", None)
    if base:
        p = os.path.join(base, "fotobudka", rel)
        if os.path.exists(p):
            return p
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), rel)


def session(key: str):
    with _lock:
        if key not in _sessions:
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.intra_op_num_threads = 2
            so.inter_op_num_threads = 1
            _sessions[key] = ort.InferenceSession(model_path(key), so, providers=["CPUExecutionProvider"])
        return _sessions[key]


def labels():
    """(nazwy_pl, embeddingi tekstu CLIP [N,512]) albo (None, None)."""
    global _labels
    if _labels is None:
        p = _resource(os.path.join("data", "labels.npz"))
        if os.path.exists(p):
            z = np.load(p, allow_pickle=False)
            _labels = (list(z["names"]), z["emb"].astype(np.float32))
        else:
            _labels = (None, None)
    return _labels


# --------------------------------------------------------------------------- detekcja (YOLOv8)
def detect(img: Image.Image, conf: float = 0.30) -> list[tuple[int, float, tuple[float, float, float, float]]]:
    """Zwraca [(klasa, pewność, (x0, y0, x1, y1))] we współrzędnych obrazu."""
    sess = session("yolo")
    size = 640
    w, h = img.size
    s = size / max(w, h)
    nw, nh = int(round(w * s)), int(round(h * s))
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    px, py = (size - nw) // 2, (size - nh) // 2
    canvas.paste(img.convert("RGB").resize((nw, nh), Image.Resampling.BILINEAR), (px, py))
    x = (np.asarray(canvas, dtype=np.float32) / 255.0).transpose(2, 0, 1)[None]
    out = sess.run(None, {sess.get_inputs()[0].name: x})[0]
    out = out.reshape(out.shape[-2], out.shape[-1])                  # [84, 8400]
    if out.shape[0] > out.shape[1]:
        out = out.T
    boxes, scores = out[:4].T, out[4:].T
    cls = scores.argmax(1)
    sc = scores[np.arange(len(cls)), cls]
    keep = sc >= conf
    boxes, cls, sc = boxes[keep], cls[keep], sc[keep]
    res = []
    for c in np.unique(cls):
        if c != PERSON and c not in ANIMALS:
            continue
        m = cls == c
        b, s_ = boxes[m], sc[m]
        xyxy = np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2,
                         b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1)
        for i in _nms(xyxy, s_, 0.5):
            x0, y0, x1, y1 = xyxy[i]
            res.append((int(c), float(s_[i]),
                        ((x0 - px) / s, (y0 - py) / s, (x1 - px) / s, (y1 - py) / s)))
    return res


def _nms(b, s, thr):
    order = np.argsort(-s)
    keep = []
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    while len(order):
        i = order[0]
        keep.append(i)
        rest = order[1:]
        if not len(rest):
            break
        xx0 = np.maximum(b[i, 0], b[rest, 0])
        yy0 = np.maximum(b[i, 1], b[rest, 1])
        xx1 = np.minimum(b[i, 2], b[rest, 2])
        yy1 = np.minimum(b[i, 3], b[rest, 3])
        inter = np.clip(xx1 - xx0, 0, None) * np.clip(yy1 - yy0, 0, None)
        iou = inter / (area[i] + area[rest] - inter + 1e-9)
        order = rest[iou < thr]
    return keep


def square_crop(img: Image.Image, box, pad: float = 0.10) -> Image.Image:
    w, h = img.size
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = max(x1 - x0, y1 - y0) * (1 + 2 * pad)
    side = max(16.0, min(side, max(w, h)))
    left, top = cx - side / 2, cy - side / 2
    canvas = Image.new("RGB", (int(side), int(side)), (0, 0, 0))
    src = img.crop((int(max(0, left)), int(max(0, top)), int(min(w, left + side)), int(min(h, top + side))))
    canvas.paste(src, (int(max(0, -left)), int(max(0, -top))))
    return canvas


# --------------------------------------------------------------------------- embeddingi
def _prep(img: Image.Image, mean, std) -> np.ndarray:
    im = img.convert("RGB").resize((224, 224), Image.Resampling.BICUBIC)
    a = (np.asarray(im, dtype=np.float32) / 255.0 - mean) / std
    return a.transpose(2, 0, 1)[None]


def _norm(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    return v / max(float(np.linalg.norm(v)), 1e-9)


def dino(img: Image.Image) -> np.ndarray:
    sess = session("dino")
    outs = [o.name for o in sess.get_outputs()]
    r = sess.run(None, {sess.get_inputs()[0].name: _prep(img, IMNET_MEAN, IMNET_STD)})
    d = dict(zip(outs, r))
    if "last_hidden_state" in d:
        hs = d["last_hidden_state"][0]
        v = np.concatenate([_norm(hs[0]), _norm(hs[1:].mean(0))])   # CLS + średnia z łatek
    else:
        v = r[0]
    return _norm(v)


def clip(img: Image.Image) -> np.ndarray:
    sess = session("clip")
    outs = [o.name for o in sess.get_outputs()]
    name = "image_embeds" if "image_embeds" in outs else outs[0]
    return _norm(sess.run([name], {sess.get_inputs()[0].name: _prep(img, CLIP_MEAN, CLIP_STD)})[0])


def analyze_image(im: Image.Image) -> dict:
    """{"a": emb zwierzęcia | None, "p": emb osoby | None, "c": emb CLIP zwierzęcia | None, "k": gatunek}"""
    dets = detect(im)
    W, H = im.size
    animals = [d for d in dets if d[0] in ANIMALS]
    persons = [d for d in dets if d[0] == PERSON]

    def score(d):
        x0, y0, x1, y1 = d[2]
        return d[1] * ((x1 - x0) * (y1 - y0)) / (W * H)

    out = {"a": None, "p": None, "c": None, "k": None, "f": clip(_center_square(im)), "q": None}
    if animals:
        a = max(animals, key=score)
        crop = square_crop(im, a[2], 0.08)
        out["a"] = dino(crop)
        out["c"] = clip(crop)
        out["k"] = ANIMALS[a[0]]
    if persons:
        p = max(persons, key=score)
        x0, y0, x1, y1 = p[2]
        top = (x0, y0, x1, y0 + (y1 - y0) * 0.6)    # głowa, włosy i tułów
        pc = square_crop(im, top, 0.05)
        out["p"] = dino(pc)
        out["q"] = clip(pc)
    return out


def _center_square(im: Image.Image) -> Image.Image:
    w, h = im.size
    side = min(w, h)
    return im.crop(((w - side) // 2, (h - side) // 2, (w + side) // 2, (h + side) // 2))


EMB_KEYS = ("a", "p", "c", "f", "q")


def analyze_photo(data: bytes, orientation: int) -> dict:
    return analyze_image(analysis.open_jpeg(data, 1280, orientation))


def to_b64(v: np.ndarray) -> str:
    return base64.b64encode(np.asarray(v, dtype=np.float16).tobytes()).decode("ascii")


def from_b64(s: str) -> np.ndarray:
    return _norm(np.frombuffer(base64.b64decode(s), dtype=np.float16).astype(np.float32))


# --------------------------------------------------------------------------- grupowanie
def agglomerate(X: np.ndarray, threshold: float, k: int = 0) -> np.ndarray:
    """Średnie łączenie (average linkage) dla podobieństwa kosinusowego.

    Średnie podobieństwo dwóch grup = (suma_A · suma_B) / (|A|·|B|) - liczone z sum wektorów
    (dla wektorów jednostkowych to dokładnie średnia podobieństw wszystkich par).
    Najpierw szybkie mikro-grupy (prawie identyczne kadry), potem pełne łączenie.
    k > 0: łączenie aż zostanie k grup; inaczej do progu.
    """
    n = len(X)
    if n == 0:
        return np.zeros(0, dtype=int)
    micro = np.full(n, -1)
    reps: list[int] = []
    tight = min(0.97, max(threshold + 0.15, 0.90))
    for i in range(n):
        if reps:
            sims = X[reps] @ X[i]
            j = int(np.argmax(sims))
            if sims[j] >= tight:
                micro[i] = micro[reps[j]]
                continue
        micro[i] = len(reps)
        reps.append(i)
    m = len(reps)
    S = np.zeros((m, X.shape[1]), dtype=np.float64)
    cnt = np.zeros(m)
    for i in range(n):
        S[micro[i]] += X[i]
        cnt[micro[i]] += 1
    alive = list(range(m))
    owner = np.arange(m)
    target = k if k > 0 else 1
    while len(alive) > target:
        A = S[alive]
        D = (A @ A.T) / np.outer(cnt[alive], cnt[alive])
        np.fill_diagonal(D, -np.inf)
        a, b = np.unravel_index(np.argmax(D), D.shape)
        if k <= 0 and D[a, b] < threshold:
            break
        ia, ib = alive[a], alive[b]
        S[ia] += S[ib]
        cnt[ia] += cnt[ib]
        owner[owner == ib] = ia
        alive.pop(int(b))
    pos = {c: j for j, c in enumerate(alive)}
    return np.array([pos[owner[micro[i]]] for i in range(n)])


def cluster(photos: list[dict | None], series: list[int], threshold: float, k: int = 0,
            person_threshold: float = 0.75) -> tuple[list[int], list[int]]:
    """photos[i] = wynik analyze_photo (embeddingi numpy) albo None.

    Zwraca (zwierzę dla każdego zdjęcia [-1 = nie rozpoznano], właściciel dla każdej grupy zwierząt).
    """
    n = len(photos)
    idx = [i for i in range(n) if photos[i] and photos[i].get("a") is not None]
    lab = [-1] * n
    if idx:
        X = np.stack([photos[i]["a"] for i in idx])
        g = agglomerate(X, threshold, k)
        for j, i in enumerate(idx):
            lab[i] = int(g[j])
    # głosowanie w seriach: zdjęcia z jednej serii to zwykle to samo zwierzę
    by_series: dict[int, list[int]] = {}
    for i, s in enumerate(series):
        by_series.setdefault(s, []).append(i)
    for members in by_series.values():
        votes = [lab[i] for i in members if lab[i] >= 0]
        if not votes:
            continue
        best = max(set(votes), key=votes.count)
        if votes.count(best) >= 0.6 * len(votes):
            for i in members:
                lab[i] = best
    used = sorted({x for x in lab if x >= 0}, key=lambda c: -lab.count(c))
    remap = {c: j for j, c in enumerate(used)}
    lab = [remap.get(x, -1) for x in lab]
    na = len(used)
    # właściciel: grupy zwierząt z tą samą osobą na zdjęciach
    pc = []
    for c in range(na):
        ps = [photos[i]["p"] for i in range(n) if lab[i] == c and photos[i] and photos[i].get("p") is not None]
        pc.append(_norm(np.mean(ps, 0)) if len(ps) >= 3 else None)
    owners = list(range(na))
    for a in range(na):
        for b in range(a + 1, na):
            if pc[a] is not None and pc[b] is not None and float(pc[a] @ pc[b]) >= person_threshold:
                oa, ob = owners[a], owners[b]
                owners = [oa if o == ob else o for o in owners]
    remap_o = {o: j for j, o in enumerate(dict.fromkeys(owners))}
    return lab, [remap_o[o] for o in owners]


def centroid(embs: list[np.ndarray]) -> np.ndarray:
    return _norm(np.mean(np.stack(embs), axis=0))


def name_for(clip_embs: list[np.ndarray]) -> tuple[str | None, float]:
    """Rasa / gatunek dla grupy - średnia z prawdopodobieństw CLIP na zdjęciach grupy."""
    names, emb = labels()
    if names is None or not clip_embs:
        return None, 0.0
    E = np.stack(clip_embs)
    logits = 100.0 * (E @ emb.T)
    p = np.exp(logits - logits.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    mean = p.mean(0)
    i = int(np.argmax(mean))
    return names[i], float(mean[i])


# --------------------------------------------------------------------------- wyszukiwanie opisem (po polsku)
_tok = None


def text_embed(query: str) -> np.ndarray:
    """Opis w dowolnym języku -> wektor w przestrzeni obrazów CLIP (wielojęzyczny CLIP)."""
    global _tok
    with _lock:
        if _tok is None:
            from tokenizers import Tokenizer
            _tok = Tokenizer.from_file(model_path("mtok"))
            _tok.no_padding()
            _tok.enable_truncation(128)
    enc = _tok.encode(query)
    ids = np.array([enc.ids], dtype=np.int64)
    mask = np.ones_like(ids)
    sess = session("mtext")
    feeds = {}
    for i in sess.get_inputs():
        feeds[i.name] = mask if "mask" in i.name else ids
    return _norm(sess.run(None, feeds)[0])


def search_scores(t: np.ndarray, photos: list[dict | None]) -> np.ndarray:
    """Dopasowanie opisu do zdjęć: najlepsze z (zwierzę, osoba, cały kadr). NaN = brak danych."""
    out = np.full(len(photos), np.nan, dtype=np.float32)
    for i, r in enumerate(photos):
        if not r:
            continue
        vals = [float(t @ r[k]) for k in ("c", "q", "f") if r.get(k) is not None]
        if vals:
            out[i] = max(vals)
    return out


def search_select(scores: np.ndarray, strictness: float = 1.0) -> np.ndarray:
    """Które zdjęcia pasują: wynik wyraźnie powyżej typowego dla tej sesji (z-score)."""
    v = scores[~np.isnan(scores)]
    if len(v) < 3:
        return ~np.isnan(scores)
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med))) * 1.4826 + 1e-6
    z = (scores - med) / mad
    return np.nan_to_num(z, nan=-99) >= strictness
