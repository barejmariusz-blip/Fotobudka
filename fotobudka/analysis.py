"""Ocena ostrości, grupowanie serii i sugestie odrzutu."""
from __future__ import annotations

import io
from dataclasses import dataclass, field

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


METRIC_VERSION = 2


def _box1d(a: np.ndarray, k: int, axis: int) -> np.ndarray:
    pad = [(0, 0), (0, 0)]
    pad[axis] = (k // 2, k // 2)
    c = np.cumsum(np.pad(a, pad, mode="edge"), axis=axis, dtype=np.float64)
    zero = np.zeros_like(np.take(c, [0], axis=axis))
    c = np.concatenate([zero, c], axis=axis)
    n = a.shape[axis]
    return ((np.take(c, range(k, k + n), axis=axis) - np.take(c, range(0, n), axis=axis)) / k).astype(np.float32)


def sharpness_score(img: Image.Image, grid: int = 12, k: int = 9) -> float:
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
        return 0.0

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
    return min(res)


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
