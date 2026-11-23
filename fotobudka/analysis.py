"""Ocena ostrości, grupowanie serii i sugestie odrzutu."""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from PIL import Image

ANALYSIS_EDGE = 2000   # dłuższy bok obrazu do analizy
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


METRIC_VERSION = 4


def _box1d(a: np.ndarray, k: int, axis: int) -> np.ndarray:
    pad = [(0, 0), (0, 0)]
    pad[axis] = (k // 2, k // 2)
    c = np.cumsum(np.pad(a, pad, mode="edge"), axis=axis, dtype=np.float64)
    zero = np.zeros_like(np.take(c, [0], axis=axis))
    c = np.concatenate([zero, c], axis=axis)
    n = a.shape[axis]
    return ((np.take(c, range(k, k + n), axis=axis) - np.take(c, range(0, n), axis=axis)) / k).astype(np.float32)


def sharpness_score(img: Image.Image, grid: int = 12, k: int = 9, return_tiles: bool = False):
    """Ostrość 0..1 najostrzejszego miejsca kadru (np. oczu), niezależna od kontrastu i tła.

    Metoda „ponownego rozmycia” (Crete i in.): obraz rozmywa się filtrem 9 px i mierzy,
    ile szczegółu ubyło. Ostre krawędzie tracą dużo, nieostre prawie nic. Wynik jest
    stosunkiem, więc nie zależy od tego, czy zwierzę ma kontrastową sierść, czy jasną.
    Liczone w kafelkach siatki 12x12, tylko w 15% kafelków z największą ilością detalu
    (płaskie tło i bokeh są pomijane), osobno w poziomie i pionie (wykrywa poruszenie);
    wynik = gorsza z osi, średnia z 2 najlepszych kafelków.
    """
    g = np.asarray(img.convert("L"), dtype=np.float32)
    h, w = g.shape[0] - 1, g.shape[1] - 1
    th, tw = h // grid, w // grid
    if th < 8 or tw < 8:
        return (0.0, None) if return_tiles else 0.0

    def tiles(a):
        return a[:th * grid, :tw * grid].reshape(grid, th, grid, tw).sum(axis=(1, 3))

    per_axis = []
    for ax in (0, 1):
        dF = np.abs(np.diff(g, axis=ax))[:h, :w]
        dB = np.abs(np.diff(_box1d(g, k, ax), axis=ax))[:h, :w]
        per_axis.append((tiles(dF), tiles(np.maximum(0, dF - dB))))
    activity = (per_axis[0][0] + per_axis[1][0]).ravel()
    top = np.argsort(activity)[::-1][:max(3, int(activity.size * 0.15))]
    res = []
    for F, V in per_axis:
        s = V.ravel()[top] / np.maximum(F.ravel()[top], 1e-6)
        res.append(float(np.sort(s)[::-1][:2].mean()))
    if not return_tiles:
        return min(res)
    F = per_axis[0][0] + per_axis[1][0]
    V = per_axis[0][1] + per_axis[1][1]
    sharp = V / np.maximum(F, 1e-6)
    act = F / max(float(F.max()), 1e-6)
    return min(res), (sharp ** 2) * act        # waga kafelka: ostry i z detalem = obiekt


# ------------------------------------------------------------------ zwierzę / kolorystyka
def subject_signature(img: Image.Image, tiles: np.ndarray | None) -> tuple[list[float], list[float]]:
    """Histogram kolorów obiektu w ostrości (najczęściej zwierzę) + jego średni kolor RGB.

    Obiekt = 12% kafelków z największą wagą (ostre i z detalem); tło w bokeh jest pomijane.
    Biny: 12 odcieni x 2 nasycenia + 3 neutralne (ciemny / szary / jasny) = 27.
    """
    small = img.convert("RGB").resize((192, 128) if img.width >= img.height else (128, 192),
                                      Image.Resampling.BILINEAR)
    rgb = np.asarray(small, dtype=np.float32)
    hsv = np.asarray(small.convert("HSV"), dtype=np.float32)
    h, w = rgb.shape[:2]
    mask = np.ones((h, w), dtype=bool)
    if tiles is not None:
        g = tiles.shape[0]
        thr = np.quantile(tiles, 0.88)
        sel = tiles >= thr
        mask = np.kron(sel, np.ones((int(np.ceil(h / g)), int(np.ceil(w / g))), dtype=bool))[:h, :w]
        if mask.sum() < 50:
            mask = np.ones((h, w), dtype=bool)
    H, S, V = hsv[..., 0][mask], hsv[..., 1][mask], hsv[..., 2][mask]
    neutral = (S < 45) | (V < 45)
    hist = np.zeros(27, dtype=np.float64)
    hb = np.minimum((H[~neutral] / 256 * 12).astype(int), 11)
    sb = (S[~neutral] >= 130).astype(int)
    np.add.at(hist, hb * 2 + sb, 1)
    vb = np.where(V[neutral] < 70, 0, np.where(V[neutral] < 170, 1, 2))
    np.add.at(hist, 24 + vb, 1)
    hist /= max(hist.sum(), 1)
    mean = rgb[mask].mean(axis=0)
    return [round(float(x), 4) for x in hist], [round(float(x), 1) for x in mean]


def _kmeans(X: np.ndarray, k: int, seed: int = 0, iters: int = 40) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(X)
    C = [X[rng.integers(n)]]
    for _ in range(1, k):
        d = np.min(((X[:, None, :] - np.array(C)[None]) ** 2).sum(-1), axis=1)
        C.append(X[rng.choice(n, p=d / d.sum())] if d.sum() > 0 else X[rng.integers(n)])
    C = np.array(C)
    lab = np.zeros(n, dtype=int)
    for _ in range(iters):
        d = ((X[:, None, :] - C[None]) ** 2).sum(-1)
        new = d.argmin(1)
        if (new == lab).all() and _ > 0:
            break
        lab = new
        for j in range(k):
            if (lab == j).any():
                C[j] = X[lab == j].mean(0)
    return lab


def _silhouette(X: np.ndarray, lab: np.ndarray) -> float:
    if len(set(lab.tolist())) < 2:
        return -1.0
    if len(X) > 800:   # próbka dla szybkości
        idx = np.random.default_rng(1).choice(len(X), 800, replace=False)
        X, lab = X[idx], lab[idx]
    D = np.sqrt(((X[:, None, :] - X[None]) ** 2).sum(-1))
    s = []
    for i in range(len(X)):
        same = lab == lab[i]
        if same.sum() <= 1:
            s.append(0.0)
            continue
        a = D[i, same].sum() / (same.sum() - 1)
        b = min(D[i, lab == j].mean() for j in set(lab.tolist()) if j != lab[i])
        s.append((b - a) / max(a, b, 1e-9))
    return float(np.mean(s))


def animal_clusters(sigs: list[list[float] | None], k: int = 0) -> list[int]:
    """Przypisanie zdjęć do grup kolorystycznych (0..k-1); -1 = brak danych.

    k = 0 -> automatycznie 1..6 (najlepszy wskaźnik silhouette; poniżej 0.25 = jedna grupa).
    Odległość Hellingera (pierwiastek z histogramu + euklides).
    """
    idx = [i for i, s in enumerate(sigs) if s]
    out = [-1] * len(sigs)
    if not idx:
        return out
    X = np.sqrt(np.array([sigs[i] for i in idx], dtype=np.float64))
    if k == 1 or len(idx) < 4:
        lab = np.zeros(len(idx), dtype=int)
    elif k > 1:
        lab = _kmeans(X, min(k, len(idx)))
    else:
        best, lab = 0.25, np.zeros(len(idx), dtype=int)
        for kk in range(2, min(6, len(idx) - 1) + 1):
            l2 = _kmeans(X, kk)
            sc = _silhouette(X, l2)
            if sc > best + 0.02:
                best, lab = sc, l2
    # numeracja wg liczności (największa grupa = 1)
    order = [c for c, _ in sorted(((c, (lab == c).sum()) for c in set(lab.tolist())), key=lambda t: -t[1])]
    remap = {c: i for i, c in enumerate(order)}
    for j, i in enumerate(idx):
        out[i] = remap[int(lab[j])]
    return out


def color_name(rgb) -> str:
    import colorsys
    r, g, b = (c / 255 for c in rgb)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    if v < 0.25:
        return "ciemne"
    if s < 0.18:
        return "jasne" if v > 0.7 else "szare"
    hd = h * 360
    if hd < 15 or hd >= 330:
        return "czerwonawe"
    if hd < 45:
        return "rude / brązowe" if v < 0.75 else "rude / kremowe"
    if hd < 70:
        return "złote / żółte"
    if hd < 170:
        return "zielonkawe"
    if hd < 260:
        return "niebieskawe"
    return "fioletowe"


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


def suggest(scores: list[float | None], groups: list[Group], sensitivity: float) -> tuple[set[int], dict[int, float], dict[int, str]]:
    """Zwraca (sugerowane do odrzutu, ostrość 0..1, powód: 'blur' | 'series').

    Pracujemy na „rozmyciu” b = 1 - ostrość (ostre zdjęcie ~0.1-0.2, nieostre > 0.4).
    sensitivity 0..1: wyżej = więcej odrzutów.
      * nieostre: b > próg_bezwzględny (0.50 .. 0.25, domyślnie ~0.37),
      * słabsze w serii: b > F * b_najlepszego i różnica > 0.05 (F 2.6 .. 1.3).
    Brak porównywania z resztą sesji - zdjęcie z małą głębią ostrości i ostrymi oczami
    nie przegrywa ze zdjęciami ostrymi w całym kadrze.
    """
    rel: dict[int, float] = {}
    rejects: set[int] = set()
    reasons: dict[int, str] = {}
    t_abs = 0.50 - 0.25 * sensitivity
    f_series = 2.6 - 1.3 * sensitivity
    for g in groups:
        sc = [(i, scores[i]) for i in g.members if scores[i] is not None]
        if not sc:
            continue
        bi, bs = max(sc, key=lambda x: x[1])
        g.best = bi if len(g.members) > 1 else None
        b_best = 1 - bs
        for i, s in sc:
            rel[i] = s
            b = 1 - s
            if b > t_abs:
                rejects.add(i)
                reasons[i] = "blur"
            elif len(sc) > 1 and b > f_series * b_best and b - b_best > 0.05:
                rejects.add(i)
                reasons[i] = "series"
    return rejects, rel, reasons


# ------------------------------------------------------------------ ekspozycja
OVER_CLIP = 0.10    # >10% kadru z wypalonym kanałem = prześwietlone (białe zwierzę ~3% to norma)
UNDER_TOP = 170     # najjaśniejsze 0,5% kadru ciemniejsze niż 170/255 = niedoświetlone


def exposure_stats(img: Image.Image) -> tuple[float, float, float]:
    """(udział wypalonych pikseli, jasność najjaśniejszych 0,5%, średnia jasność 0..255).

    Wypalony = którykolwiek kanał >= 253 (np. rudy kot potrafi wypalić tylko czerwony).
    """
    a = np.asarray(img.convert("RGB"))
    mx = a.max(axis=2)
    lum = a.astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    return float((mx >= 253).mean()), float(np.percentile(mx, 99.5)), float(lum.mean())


def exposure_flag(ev) -> str | None:
    if not ev:
        return None
    hi, top, _mean = ev
    if hi > OVER_CLIP:
        return "over"
    if top < UNDER_TOP:
        return "under"
    return None


# ------------------------------------------------------------------ podobne kadry
def dhash(img: Image.Image, size: int = 16) -> str:
    """Hash różnicowy 16x16 = 256 bitów (hex). Odporny na rozmycie i drobne zmiany."""
    g = np.asarray(img.convert("L").resize((size + 1, size), Image.Resampling.BILINEAR), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).ravel()
    return np.packbits(bits).tobytes().hex()


@lru_cache(maxsize=20000)
def _bits(h: str) -> np.ndarray:
    return np.unpackbits(np.frombuffer(bytes.fromhex(h), dtype=np.uint8))


def similarity_groups(hashes: list[str | None], threshold: float) -> list[list[int]]:
    """Grupuje zdjęcia (w kolejności czasu) o podobieństwie >= threshold (0..1).

    Zachłannie: zdjęcie trafia do grupy, której pierwsze zdjęcie jest najbardziej podobne
    (bez „łańcuchowania”, które skleiłoby całą sesję na tym samym tle).
    """
    groups: list[list[int]] = []
    R = np.zeros((len(hashes), 256), dtype=np.uint8)   # wzorce grup (pierwsze zdjęcie)
    owner: list[int] = []
    for i, h in enumerate(hashes):
        if not h:
            groups.append([i])
            continue
        b = _bits(h)
        n = len(owner)
        if n:
            sims = 1.0 - np.count_nonzero(R[:n] != b, axis=1) / b.size
            k = int(np.argmax(sims))
            if sims[k] >= threshold:
                groups[owner[k]].append(i)
                continue
        groups.append([i])
        R[n] = b
        owner.append(len(groups) - 1)
    return groups
