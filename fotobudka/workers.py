"""Zadania w tle (QThreadPool): metadane, analiza + miniatury, ładowanie podglądu, kopiowanie."""
from __future__ import annotations

import os
import shutil
import traceback

from PySide6.QtCore import QObject, QRunnable, Signal
from PySide6.QtGui import QImage

from . import analysis, raw


def pil_to_qimage(im) -> QImage:
    im = im.convert("RGB")
    data = im.tobytes("raw", "RGB")
    q = QImage(data, im.width, im.height, im.width * 3, QImage.Format.Format_RGB888)
    return q.copy()


class Signals(QObject):
    meta_done = Signal(int, int, object)                 # gen, idx, PhotoMeta | None
    analyzed = Signal(int, int, object, object, object)  # gen, idx, score | None, QImage | None, extras | None
    view_loaded = Signal(int, int, bool, object)         # gen, idx, full, QImage | None
    progress = Signal(int, int, str)                     # done, total, current name
    finished = Signal(int, int, list)                    # ok, failed, errors


class MetaTask(QRunnable):
    def __init__(self, sig: Signals, gen: int, idx: int, path: str):
        super().__init__()
        self.sig, self.gen, self.idx, self.path = sig, gen, idx, path

    def run(self):
        try:
            meta = raw.read_meta(self.path)
        except Exception:
            meta = None
        self.sig.meta_done.emit(self.gen, self.idx, meta)


class AnalyzeTask(QRunnable):
    def __init__(self, sig: Signals, gen: int, idx: int, path: str, meta, need_score: bool):
        super().__init__()
        self.sig, self.gen, self.idx, self.path = sig, gen, idx, path
        self.meta, self.need_score = meta, need_score

    def run(self):
        score = None
        thumb = None
        extras = None
        try:
            data = raw.read_preview_bytes(self.path, self.meta)
            if self.need_score:
                im = analysis.open_jpeg(data, analysis.ANALYSIS_EDGE, self.meta.orientation)
                score = analysis.sharpness_score(im)
                extras = {"ev": analysis.exposure_stats(im), "h": analysis.dhash(im)}
            else:
                im = analysis.open_jpeg(data, analysis.THUMB_EDGE * 2, self.meta.orientation)
            im.thumbnail((analysis.THUMB_EDGE, analysis.THUMB_EDGE))
            thumb = pil_to_qimage(im)
        except Exception:
            pass
        self.sig.analyzed.emit(self.gen, self.idx, score, thumb, extras)


class ViewTask(QRunnable):
    def __init__(self, sig: Signals, gen: int, idx: int, path: str, meta, max_edge: int | None):
        super().__init__()
        self.sig, self.gen, self.idx, self.path = sig, gen, idx, path
        self.meta, self.max_edge = meta, max_edge

    def run(self):
        img = None
        try:
            data = raw.read_preview_bytes(self.path, self.meta)
            img = pil_to_qimage(analysis.open_jpeg(data, self.max_edge, self.meta.orientation))
        except Exception:
            pass
        self.sig.view_loaded.emit(self.gen, self.idx, self.max_edge is None, img)


class FileOpTask(QRunnable):
    """Kopiowanie lub przenoszenie plików (+ XMP) do folderu docelowego."""

    def __init__(self, sig: Signals, paths: list[str], dest: str, move: bool):
        super().__init__()
        self.sig, self.paths, self.dest, self.move = sig, paths, dest, move
        self.cancelled = False

    def run(self):
        ok = 0
        errors: list[str] = []
        os.makedirs(self.dest, exist_ok=True)
        total = len(self.paths)
        for n, p in enumerate(self.paths):
            if self.cancelled:
                break
            self.sig.progress.emit(n, total, os.path.basename(p))
            try:
                for src in (p, os.path.splitext(p)[0] + ".xmp"):
                    if not os.path.exists(src):
                        continue
                    dst = os.path.join(self.dest, os.path.basename(src))
                    if self.move:
                        shutil.move(src, dst)
                    else:
                        shutil.copy2(src, dst)
                ok += 1
            except Exception as e:
                errors.append(f"{os.path.basename(p)}: {e}")
                traceback.print_exc()
        self.sig.progress.emit(total, total, "")
        self.sig.finished.emit(ok, len(errors), errors)
