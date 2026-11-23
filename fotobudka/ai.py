"""Rozpoznawanie zwierząt (AI): model CLIP ViT-B/32 (ONNX, ok. 90 MB, pobierany raz).

* Embedding liczony z wycinka kadru wokół obiektu w ostrości (zwierzę + osoba obok),
  więc tło studia / pleneru ma mniejszy wpływ.
* Grupowanie: podobieństwo kosinusowe embeddingów (to samo zwierzę / ta sama osoba).
* Nazwa grupy: rasa / gatunek rozpoznany „zero-shot” (porównanie z opisami tekstowymi
  przeliczonymi przy budowaniu programu - data/labels.npz).
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
    from ._model import MODEL_URL, MODEL_SHA256, MODEL_SIZE
except ImportError:  # wersja deweloperska
    MODEL_URL = ("https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/"
                 "onnx/vision_model_quantized.onnx")
    MODEL_SHA256 = None
    MODEL_SIZE = 0

MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

_lock = threading.Lock()
_session = None
_labels = None


def data_dir() -> str:
    base = os.environ.get("LOCALAPPDATA") or os.path.join(os.path.expanduser("~"), ".local", "share")
    d = os.path.join(base, "PanRodla", "Fotobudka", "models")
    os.makedirs(d, exist_ok=True)
    return d


def model_path() -> str:
    return os.path.join(data_dir(), "clip-vit-b32-vision-q.onnx")


def available() -> bool:
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    return True


def model_ready() -> bool:
    p = model_path()
    return os.path.exists(p) and (not MODEL_SIZE or os.path.getsize(p) == MODEL_SIZE)


def download_model(progress=None, cancelled=lambda: False) -> None:
    """Pobiera model do %LOCALAPPDATA%; sprawdza rozmiar i SHA-256."""
    dst = model_path()
    tmp = dst + ".part"
    req = urllib.request.Request(MODEL_URL, headers={"User-Agent": "Fotobudka (PanRodla)"})
    h = hashlib.sha256()
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or MODEL_SIZE or 0)
        done = 0
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
    if MODEL_SHA256 and h.hexdigest() != MODEL_SHA256:
        os.remove(tmp)
        raise RuntimeError("Pobrany model jest uszkodzony (zła suma kontrolna) - spróbuj ponownie.")
    os.replace(tmp, dst)


def _resource(rel: str) -> str:
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    p = os.path.join(base, "fotobudka", rel)
    return p if os.path.exists(p) else os.path.join(os.path.dirname(os.path.abspath(__file__)), rel)


def session():
    global _session
    with _lock:
        if _session is None:
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.intra_op_num_threads = 2
            so.inter_op_num_threads = 1
            _session = ort.InferenceSession(model_path(), so, providers=["CPUExecutionProvider"])
        return _session


def labels():
    """(nazwy_pl, macierz embeddingów tekstu [N,512]) albo (None, None)."""
    global _labels
    if _labels is None:
        p = _resource(os.path.join("data", "labels.npz"))
        if os.path.exists(p):
            z = np.load(p, allow_pickle=False)
            _labels = (list(z["names"]), z["emb"].astype(np.float32))
        else:
            _labels = (None, None)
    return _labels


# --------------------------------------------------------------------------- obraz -> embedding
def subject_crop(img: Image.Image, tiles: np.ndarray | None, expand: float = 0.35) -> Image.Image:
    """Kwadratowy wycinek wokół najostrzejszych kafelków (obiekt w ostrości)."""
    w, h = img.size
    if tiles is None:
        side = min(w, h)
        return img.crop(((w - side) // 2, (h - side) // 2, (w + side) // 2, (h + side) // 2))
    g = tiles.shape[0]
    sel = tiles >= np.quantile(tiles, 0.88)
    ys, xs = np.nonzero(sel)
    x0, x1 = xs.min() / g * w, (xs.max() + 1) / g * w
    y0, y1 = ys.min() / g * h, (ys.max() + 1) / g * h
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = max(x1 - x0, y1 - y0) * (1 + expand)
    side = min(max(side, min(w, h) * 0.45), min(w, h))
    left = min(max(0, cx - side / 2), w - side)
    top = min(max(0, cy - side / 2), h - side)
    return img.crop((int(left), int(top), int(left + side), int(top + side)))


def preprocess(img: Image.Image) -> np.ndarray:
    im = img.convert("RGB").resize((224, 224), Image.Resampling.BICUBIC)
    a = (np.asarray(im, dtype=np.float32) / 255.0 - MEAN) / STD
    return a.transpose(2, 0, 1)[None]


def embed_image(img: Image.Image) -> np.ndarray:
    sess = session()
    inp = sess.get_inputs()[0].name
    outs = [o.name for o in sess.get_outputs()]
    name = "image_embeds" if "image_embeds" in outs else outs[0]
    v = sess.run([name], {inp: preprocess(img)})[0].reshape(-1).astype(np.float32)
    return v / max(float(np.linalg.norm(v)), 1e-9)


def embed_photo(data: bytes, orientation: int, mode: str = "mix") -> np.ndarray:
    """mode: crop (obiekt w ostrości), full (cały kadr wyśrodkowany), mix (średnia obu)."""
    im = analysis.open_jpeg(data, 1024, orientation)
    if mode == "full":
        return embed_image(subject_crop(im, None))
    _, tiles = analysis.sharpness_score(im, return_tiles=True)
    c = embed_image(subject_crop(im, tiles))
    if mode == "crop":
        return c
    v = c + embed_image(subject_crop(im, None))
    return v / max(float(np.linalg.norm(v)), 1e-9)


def to_b64(v: np.ndarray) -> str:
    return base64.b64encode(v.astype(np.float16).tobytes()).decode("ascii")


def from_b64(s: str) -> np.ndarray:
    v = np.frombuffer(base64.b64decode(s), dtype=np.float16).astype(np.float32)
    return v / max(float(np.linalg.norm(v)), 1e-9)


# --------------------------------------------------------------------------- grupowanie
def cluster(embs: list[np.ndarray | None], threshold: float = 0.86, k: int = 0) -> list[int]:
    """Grupy zdjęć tego samego zwierzęcia/osoby. Zwraca etykiety 0.. (największa = 0), -1 brak.

    k = 0: automatycznie wg progu podobieństwa (lider + łączenie podobnych grup + doprecyzowanie).
    k > 0: dokładnie k grup (k-means na sferze).
    """
    idx = [i for i, e in enumerate(embs) if e is not None]
    out = [-1] * len(embs)
    if not idx:
        return out
    X = np.stack([embs[i] for i in idx])
    if k > 0:
        lab = _spherical_kmeans(X, min(k, len(idx)))
    else:
        C: list[np.ndarray] = []
        S: list[np.ndarray] = []
        cnt: list[int] = []
        lab = np.zeros(len(X), dtype=int)
        for j, x in enumerate(X):
            if C:
                sims = np.stack(C) @ x
                b = int(np.argmax(sims))
                if sims[b] >= threshold:
                    lab[j] = b
                    S[b] = S[b] + x
                    cnt[b] += 1
                    C[b] = S[b] / np.linalg.norm(S[b])
                    continue
            lab[j] = len(C)
            C.append(x.copy())
            S.append(x.copy())
            cnt.append(1)
        # łączenie grup o podobnych środkach
        merged = True
        while merged and len(C) > 1:
            merged = False
            M = np.stack(C) @ np.stack(C).T
            np.fill_diagonal(M, -1)
            a, b = np.unravel_index(np.argmax(M), M.shape)
            if M[a, b] >= threshold:
                a, b = min(a, b), max(a, b)
                lab[lab == b] = a
                lab[lab > b] -= 1
                S[a] = S[a] + S[b]
                C[a] = S[a] / np.linalg.norm(S[a])
                cnt[a] += cnt[b]
                del C[b], S[b], cnt[b]
                merged = True
        # doprecyzowanie: każde zdjęcie do najbliższego środka
        Cm = np.stack(C)
        for _ in range(3):
            lab = np.argmax(X @ Cm.T, axis=1)
            Cm = np.stack([X[lab == c].mean(0) if (lab == c).any() else Cm[c] for c in range(len(Cm))])
            Cm /= np.linalg.norm(Cm, axis=1, keepdims=True)
        # małe grupy (1-2 zdjęcia) dołącz do najbliższej, jeśli w miarę podobne
        sizes = np.bincount(lab, minlength=len(Cm))
        for c in np.where(sizes < 3)[0]:
            if len(Cm) < 2:
                break
            sims = Cm @ Cm[c]
            sims[c] = -1
            sims[sizes < 3] = -1
            b = int(np.argmax(sims))
            if sims[b] >= threshold - 0.06:
                lab[lab == c] = b
    order = [c for c, _ in sorted(((c, int((lab == c).sum())) for c in set(lab.tolist())), key=lambda t: -t[1])]
    remap = {c: i for i, c in enumerate(order)}
    for j, i in enumerate(idx):
        out[i] = remap[int(lab[j])]
    return out


def _spherical_kmeans(X: np.ndarray, k: int, iters: int = 30) -> np.ndarray:
    rng = np.random.default_rng(0)
    C = [X[rng.integers(len(X))]]
    for _ in range(1, k):
        d = 1 - np.max(X @ np.stack(C).T, axis=1)
        d = np.clip(d, 0, None) ** 2
        C.append(X[rng.choice(len(X), p=d / d.sum())] if d.sum() > 0 else X[rng.integers(len(X))])
    C = np.stack(C)
    lab = np.zeros(len(X), dtype=int)
    for _ in range(iters):
        new = np.argmax(X @ C.T, axis=1)
        if _ and (new == lab).all():
            break
        lab = new
        for c in range(k):
            if (lab == c).any():
                m = X[lab == c].mean(0)
                C[c] = m / np.linalg.norm(m)
    return lab


def centroid(embs: list[np.ndarray]) -> np.ndarray:
    m = np.mean(np.stack(embs), axis=0)
    return m / max(float(np.linalg.norm(m)), 1e-9)


def name_for(c: np.ndarray) -> tuple[str | None, float]:
    """Rasa / gatunek dla środka grupy (zero-shot CLIP)."""
    names, emb = labels()
    if names is None:
        return None, 0.0
    logits = 100.0 * (emb @ c)
    p = np.exp(logits - logits.max())
    p /= p.sum()
    i = int(np.argmax(p))
    return names[i], float(p[i])
