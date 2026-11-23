"""Fotobudka - szybka selekcja zdjęć po sesji. made by PanRodla."""
from __future__ import annotations

import os
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass

from PySide6.QtCore import (QAbstractListModel, QModelIndex, QPointF, QRect, QRectF, QSettings,
                            QSize, Qt, QThreadPool, QTimer)
from PySide6.QtGui import (QAction, QColor, QFont, QGuiApplication, QImage, QKeySequence, QPainter,
                           QPalette, QPen, QPixmap)
from PySide6.QtWidgets import QPushButton
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                               QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout, QLabel,
                               QListView, QMainWindow, QMessageBox, QProgressBar, QProgressDialog,
                               QSlider, QSpinBox, QSplitter, QStyle, QStyledItemDelegate, QToolBar,
                               QVBoxLayout, QWidget)

from . import analysis, raw, storage
from .workers import AnalyzeTask, FileOpTask, MetaTask, Signals, ViewTask

APP_NAME = "Fotobudka"
VERSION = "1.0"
LABEL_COLORS = {"Red": "#d9534f", "Yellow": "#e8c547", "Green": "#5cb85c", "Blue": "#4a90d9"}
REJ_PL = {"blur": "automat – nieostre", "series": "automat – słabsze w serii", "manual": "ręcznie"}
LABEL_PL = {"Red": "czerwona", "Yellow": "żółta", "Green": "zielona", "Blue": "niebieska"}

FILTERS = [
    ("Wszystkie", lambda p: True),
    ("Nieocenione", lambda p: p.rating == 0 and not p.rejected),
    ("Ocenione ≥ 1★", lambda p: p.rating >= 1 and not p.rejected),
    ("Ocenione ≥ 2★", lambda p: p.rating >= 2 and not p.rejected),
    ("Ocenione ≥ 3★", lambda p: p.rating >= 3 and not p.rejected),
    ("Ocenione ≥ 4★", lambda p: p.rating >= 4 and not p.rejected),
    ("Bez odrzuconych", lambda p: not p.rejected),
    ("Odrzucone", lambda p: p.rejected),
    ("Sugerowane do odrzutu", lambda p: p.suggested and not p.rejected),
    ("Najlepsze z serii", lambda p: p.best),
    ("Z etykietą", lambda p: p.label is not None),
]


@dataclass
class Photo:
    path: str
    name: str
    size: int
    mtime: float
    meta: raw.PhotoMeta | None = None
    score: float | None = None
    thumb: QImage | None = None
    rating: int = 0
    rejected: bool = False
    rej_by: str | None = None          # blur / series / manual
    label: str | None = None
    group: int = -1
    group_pos: int = 0
    group_len: int = 1
    suggested: bool = False
    reason: str | None = None
    rel: float | None = None
    best: bool = False
    failed: bool = False


# =========================================================================== miniatury
class ThumbModel(QAbstractListModel):
    def __init__(self, win: "MainWindow"):
        super().__init__()
        self.win = win
        self.rows: list[int] = []

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.rows)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        if role == Qt.ItemDataRole.UserRole:
            return self.rows[index.row()]
        if role == Qt.ItemDataRole.ToolTipRole:
            return self.win.photos[self.rows[index.row()]].name
        return None

    def set_rows(self, rows: list[int]):
        self.beginResetModel()
        self.rows = rows
        self.endResetModel()

    def refresh_photo(self, idx: int):
        try:
            r = self.rows.index(idx)
        except ValueError:
            return
        mi = self.index(r)
        self.dataChanged.emit(mi, mi)


class ThumbDelegate(QStyledItemDelegate):
    W, H = 150, 128

    def __init__(self, win: "MainWindow"):
        super().__init__()
        self.win = win

    def sizeHint(self, option, index):
        return QSize(self.W, self.H)

    def paint(self, painter: QPainter, option, index):
        p: Photo = self.win.photos[index.data(Qt.ItemDataRole.UserRole)]
        r = option.rect.adjusted(3, 3, -3, -3)
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        painter.fillRect(r, QColor("#3a3a3a" if selected else "#262626"))
        if selected:
            painter.setPen(QPen(QColor("#e0e0e0"), 2))
            painter.drawRect(r.adjusted(1, 1, -1, -1))

        img_area = QRect(r.left() + 5, r.top() + 5, r.width() - 10, r.height() - 32)
        if p.thumb is not None and not p.thumb.isNull():
            ts = p.thumb.size().scaled(img_area.size(), Qt.AspectRatioMode.KeepAspectRatio)
            tr = QRect(0, 0, ts.width(), ts.height())
            tr.moveCenter(img_area.center())
            painter.drawImage(tr, p.thumb)
            if p.rejected:
                if self.win.reviewing_rejected():
                    painter.setPen(QPen(QColor("#e04646"), 3))
                    painter.drawRect(tr.adjusted(1, 1, -1, -1))
                else:
                    painter.fillRect(tr, QColor(0, 0, 0, 150))
                    painter.setPen(QPen(QColor("#e04646"), 3))
                    painter.drawLine(tr.topLeft(), tr.bottomRight())
                    painter.drawLine(tr.topRight(), tr.bottomLeft())
        else:
            painter.setPen(QColor("#777"))
            painter.drawText(img_area, Qt.AlignmentFlag.AlignCenter, "błąd" if p.failed else "…")

        # pasek serii
        if p.group_len > 1:
            col = QColor("#2f8f8f") if p.group % 2 == 0 else QColor("#7a5aa8")
            painter.fillRect(QRect(r.left(), r.bottom() - 3, r.width(), 4), col)

        # etykieta kolorowa
        if p.label:
            painter.fillRect(QRect(r.left(), r.top(), 6, r.height() - 4), QColor(LABEL_COLORS[p.label]))

        # podpis: gwiazdki / nazwa
        bottom = QRect(r.left() + 6, r.bottom() - 26, r.width() - 12, 20)
        f = painter.font()
        f.setPointSizeF(8.5)
        painter.setFont(f)
        if p.rating:
            painter.setPen(QColor("#f0c419"))
            painter.drawText(bottom, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, "★" * p.rating)
        else:
            painter.setPen(QColor("#9a9a9a"))
            painter.drawText(bottom, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                             p.name if len(p.name) < 18 else p.name[:16] + "…")

        # odznaki po prawej
        badges = []
        if p.best:
            badges.append(("TOP", "#c99a10"))
        if p.suggested and not p.rejected:
            badges.append(("?", "#d9822b"))
        x = bottom.right()
        for text, color in badges:
            w = 30 if len(text) > 1 else 18
            br = QRect(x - w, bottom.top() + 2, w, 16)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(color))
            painter.drawRoundedRect(br, 4, 4)
            painter.setPen(QColor("#111"))
            painter.drawText(br, Qt.AlignmentFlag.AlignCenter, text)
            x -= w + 4
        painter.restore()


# =========================================================================== podgląd
class Viewer(QWidget):
    def __init__(self, win: "MainWindow"):
        super().__init__()
        self.win = win
        self.pix: QPixmap | None = None
        self.full_pix: QPixmap | None = None
        self.zoom = False
        self.center = QPointF(0.5, 0.5)   # środek widoku w znormalizowanych współrzędnych
        self._drag = None
        self._scaled_cache: tuple | None = None
        self.message = ""
        self.setMinimumSize(300, 200)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

    def set_pixmaps(self, pix, full_pix):
        self.pix, self.full_pix = pix, full_pix
        self._scaled_cache = None
        self.update()

    def set_zoom(self, z: bool, at: QPointF | None = None):
        self.zoom = z
        if z and at is not None:
            rect = self._fit_rect()
            if rect and rect.width() > 0:
                self.center = QPointF(min(1, max(0, (at.x() - rect.x()) / rect.width())),
                                      min(1, max(0, (at.y() - rect.y()) / rect.height())))
        elif z:
            self.center = QPointF(0.5, 0.5)
        if z:
            self.win.request_full()
        self.update()

    def _src(self):
        return self.pix or self.full_pix

    def _fit_rect(self) -> QRectF | None:
        src = self._src()
        if not src:
            return None
        s = QSize(src.width(), src.height()).scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
        return QRectF((self.width() - s.width()) / 2, (self.height() - s.height()) / 2, s.width(), s.height())

    def paintEvent(self, e):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#181818"))
        src = self._src()
        if src is None:
            painter.setPen(QColor("#888"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             self.message or "Otwórz folder z sesją (Ctrl+O)")
            return
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        if self.zoom:
            dpr = self.devicePixelRatioF()
            img = self.full_pix or src
            scale = 1.0 if self.full_pix else (
                (self.win.current_full_width() or img.width()) / img.width())
            w, h = img.width() * scale / dpr, img.height() * scale / dpr
            x = self.width() / 2 - self.center.x() * w
            y = self.height() / 2 - self.center.y() * h
            if w < self.width():
                x = (self.width() - w) / 2
            if h < self.height():
                y = (self.height() - h) / 2
            painter.drawPixmap(QRectF(x, y, w, h), img, QRectF(img.rect()))
        else:
            rect = self._fit_rect()
            key = (rect.width(), rect.height(), id(src))
            if not self._scaled_cache or self._scaled_cache[0] != key:
                dpr = self.devicePixelRatioF()
                sp = src.scaled(int(rect.width() * dpr), int(rect.height() * dpr),
                                Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                sp.setDevicePixelRatio(dpr)
                self._scaled_cache = (key, sp)
            painter.drawPixmap(rect.topLeft(), self._scaled_cache[1])
        self._overlay(painter)

    def _overlay(self, painter: QPainter):
        p = self.win.current_photo()
        if not p:
            return
        f = QFont(painter.font())
        f.setPointSize(15)
        f.setBold(True)
        painter.setFont(f)
        box = QRect(14, 12, 420, 34)
        if p.rejected:
            txt = "✕ ODRZUCONE" + (f"  ({REJ_PL[p.rej_by]})" if p.rej_by in REJ_PL else "")
            w = painter.fontMetrics().horizontalAdvance(txt) + 20
            painter.fillRect(QRect(12, 10, w, 36), QColor(200, 40, 40, 210))
            painter.setPen(QColor("white"))
            painter.drawText(QRect(14, 12, w, 34), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, txt)
            if self.win.reviewing_rejected():
                f2 = QFont(painter.font()); f2.setPointSize(10); f2.setBold(False); painter.setFont(f2)
                painter.setPen(QColor("#eee"))
                painter.drawText(QRect(14, 50, 600, 22), Qt.AlignmentFlag.AlignLeft,
                                 "X – przywróć  •  B – pokaż najlepsze z tej serii  •  Z – zoom 100%")
        elif p.rating:
            painter.fillRect(QRect(12, 10, 22 * p.rating + 16, 36), QColor(0, 0, 0, 150))
            painter.setPen(QColor("#f0c419"))
            painter.drawText(box, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, "★" * p.rating)
        if self.zoom:
            f.setPointSize(10)
            painter.setFont(f)
            painter.setPen(QColor("#ddd"))
            painter.drawText(self.rect().adjusted(0, 0, -14, -10),
                             Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom,
                             "100%" if self.full_pix else "100% (wczytywanie…)")

    # --- mysz
    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._drag = (e.position(), QPointF(self.center), False)

    def mouseMoveEvent(self, e):
        if self._drag and self.zoom:
            start, c0, _ = self._drag
            src = self.full_pix or self._src()
            if not src:
                return
            dpr = self.devicePixelRatioF()
            fw = self.win.current_full_width() or src.width()
            fh = fw * src.height() / max(1, src.width())
            d = e.position() - start
            if abs(d.x()) + abs(d.y()) > 3:
                self._drag = (start, c0, True)
            self.center = QPointF(min(1, max(0, c0.x() - d.x() * dpr / fw)),
                                  min(1, max(0, c0.y() - d.y() * dpr / fh)))
            self.update()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton and self._drag:
            moved = self._drag[2]
            self._drag = None
            if not moved:
                self.set_zoom(not self.zoom, e.position())

    def wheelEvent(self, e):
        self.win.step(-1 if e.angleDelta().y() > 0 else 1)


# =========================================================================== dialogi
class ExportDialog(QDialog):
    def __init__(self, parent, title: str):
        super().__init__(parent)
        self.setWindowTitle(title)
        lay = QFormLayout(self)
        self.min_rating = QSpinBox()
        self.min_rating.setRange(0, 5)
        self.min_rating.setValue(1)
        self.min_rating.setSuffix(" ★")
        self.label = QComboBox()
        self.label.addItem("dowolna", None)
        for k, v in LABEL_PL.items():
            self.label.addItem(v, k)
        self.best = QCheckBox("dołącz „najlepsze z serii” (nawet bez oceny)")
        lay.addRow("Minimalna ocena:", self.min_rating)
        lay.addRow("Etykieta:", self.label)
        lay.addRow("", self.best)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addRow(bb)

    def predicate(self):
        mr, lab, best = self.min_rating.value(), self.label.currentData(), self.best.isChecked()

        def ok(p: Photo):
            if p.rejected or (lab and p.label != lab):
                return False
            return p.rating >= mr or (best and p.best)
        return ok


HELP = """<b>Skróty</b><br>
← → / kółko – poprzednie / następne<br>
1–5 – ocena, 0 – usuń ocenę<br>
X – odrzuć / przywróć<br>
O – przegląd odrzuconych<br>
6 7 8 9 – etykieta czerw./żółta/ziel./nieb.<br>
Z / Spacja / klik – zoom 100%<br>
G / Shift+G – następna / poprzednia seria<br>
B – najlepsze w serii<br>
Ctrl+Z – cofnij<br>
Ctrl+S – zapisz XMP<br>
F11 – pełny ekran"""


# =========================================================================== okno
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} – selekcja zdjęć")
        self.settings = QSettings("PanRodla", APP_NAME)
        self.folder: str | None = None
        self.photos: list[Photo] = []
        self.groups: list[analysis.Group] = []
        self.state: dict = {"photos": {}}
        self.gen = 0
        self.cur: int | None = None          # indeks w self.photos
        self.dirty_xmp: set[int] = set()
        self.undo: list[list[tuple[int, int, bool, str | None]]] = []
        self.view_cache: OrderedDict[int, QPixmap] = OrderedDict()
        self.full_cache: tuple[int, QPixmap] | None = None
        self.pending_view: set[tuple[int, bool]] = set()
        self.meta_left = 0
        self.analyze_left = 0
        self.analyze_total = 0
        self.analysis_running = False
        self.analysis_t0 = 0.0
        self.analysis_cached = False

        self.pool = QThreadPool()
        self.pool.setMaxThreadCount(max(2, (os.cpu_count() or 4) - 2))
        self.view_pool = QThreadPool()
        self.view_pool.setMaxThreadCount(2)
        self.sig = Signals()
        self.sig.meta_done.connect(self._on_meta)
        self.sig.analyzed.connect(self._on_analyzed)
        self.sig.view_loaded.connect(self._on_view)

        self.save_timer = QTimer(self, singleShot=True, interval=1200, timeout=self._save_now)
        self.suggest_timer = QTimer(self, singleShot=True, interval=600, timeout=self._recompute)

        self._build_ui()
        self._load_settings()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        tb = QToolBar("Główny")
        tb.setMovable(False)
        tb.setIconSize(QSize(18, 18))
        self.addToolBar(tb)

        def act(text, slot, shortcut=None, tip=None):
            a = QAction(text, self)
            a.triggered.connect(slot)
            if shortcut:
                a.setShortcut(QKeySequence(shortcut))
            if tip:
                a.setToolTip(tip)
            tb.addAction(a)
            return a

        act("📂 Otwórz folder", self.open_folder_dialog, "Ctrl+O")
        act("💾 Zapisz XMP", self.save_all_xmp, "Ctrl+S",
            "Zapisz oceny do plików .xmp (Lightroom / Capture One / Bridge)")
        act("📤 Kopiuj wybrane…", lambda: self.export(False), None, "Skopiuj wybrane zdjęcia do innego folderu")
        act("🗑 Przenieś odrzucone…", self.move_rejected, None, "Przenieś odrzucone do podfolderu _ODRZUCONE")
        act("📋 Lista nazw", self.copy_names, None, "Skopiuj do schowka nazwy plików wybranych zdjęć")
        self.addToolBarBreak()
        tb = QToolBar("Selekcja")
        tb.setMovable(False)
        self.addToolBar(tb)

        tb.addWidget(QLabel(" Filtr: "))
        self.filter_box = QComboBox()
        for name, _ in FILTERS:
            self.filter_box.addItem(name)
        self.filter_box.currentIndexChanged.connect(lambda _: self.apply_filter())
        self.filter_box.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        tb.addWidget(self.filter_box)
        self.rej_btn = QPushButton("🗑 Odrzucone (0)")
        self.rej_btn.setCheckable(True)
        self.rej_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.rej_btn.setToolTip("Przejrzyj odrzucone zdjęcia – sprawdź, czy automat się nie pomylił (klawisz O)")
        self.rej_btn.setStyleSheet("QPushButton{padding:3px 10px;} QPushButton:checked{background:#8a2a2a; color:white;}")
        self.rej_btn.clicked.connect(lambda _: self.toggle_rejected_view())
        tb.addWidget(self.rej_btn)
        tb.addSeparator()

        tb.addWidget(QLabel(" Czułość: "))
        self.sens = QSlider(Qt.Orientation.Horizontal)
        self.sens.setRange(0, 100)
        self.sens.setValue(50)
        self.sens.setFixedWidth(110)
        self.sens.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.sens.setToolTip("Im wyżej, tym więcej zdjęć trafia do sugerowanych odrzutów")
        self.sens.valueChanged.connect(lambda _: self._settings_changed())
        tb.addWidget(self.sens)
        self.sens_lbl = QLabel(" 50%")
        self.sens_lbl.setMinimumWidth(38)
        tb.addWidget(self.sens_lbl)

        tb.addWidget(QLabel("  Seria co: "))
        self.gap = QDoubleSpinBox()
        self.gap.setRange(0.2, 30.0)
        self.gap.setSingleStep(0.5)
        self.gap.setValue(1.5)
        self.gap.setSuffix(" s")
        self.gap.setToolTip("Zdjęcia zrobione w odstępie mniejszym niż ten czas = jedna seria")
        self.gap.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.gap.valueChanged.connect(lambda _: self._settings_changed())
        tb.addWidget(self.gap)
        self.recalc_btn = QPushButton("🔄 Przelicz")
        self.recalc_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.recalc_btn.setEnabled(False)
        self.recalc_btn.setToolTip("Przelicz sugestie dla nowej czułości / długości serii")
        self.recalc_btn.setStyleSheet("QPushButton{padding:3px 10px;} "
                                      "QPushButton:enabled{background:#b5651d; color:white; font-weight:bold;}")
        self.recalc_btn.clicked.connect(self.recalc)
        tb.addWidget(self.recalc_btn)
        self.preview_lbl = QLabel("")
        self.preview_lbl.setStyleSheet("color:#e0a060; padding-left:6px;")
        tb.addWidget(self.preview_lbl)
        tb.addSeparator()
        self.apply_act = act("✔ Zastosuj sugestie", self.apply_suggestions, None,
                             "Oznacz sugerowane (nieostre / słabsze w serii) jako odrzucone")
        self.apply_act.setEnabled(False)
        self.auto_adv = QCheckBox("Auto-dalej")
        self.auto_adv.setToolTip("Po ocenie / odrzuceniu przejdź automatycznie do następnego")
        self.auto_adv.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        tb.addWidget(self.auto_adv)
        self.auto_xmp = QCheckBox("Auto-XMP")
        self.auto_xmp.setToolTip("Zapisuj pliki .xmp na bieżąco")
        self.auto_xmp.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        tb.addWidget(self.auto_xmp)

        # centrum
        self.viewer = Viewer(self)
        self.info = QLabel()
        self.info.setTextFormat(Qt.TextFormat.RichText)
        self.info.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.info.setWordWrap(True)
        self.info.setMinimumWidth(240)
        self.info.setMaximumWidth(300)
        self.info.setStyleSheet("padding:10px; background:#202020;")
        help_lbl = QLabel(HELP)
        help_lbl.setStyleSheet("padding:10px; color:#8a8a8a; background:#202020; font-size:11px;")
        help_lbl.setWordWrap(True)
        side = QWidget()
        sl = QVBoxLayout(side)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(1)
        sl.addWidget(self.info, 1)
        sl.addWidget(help_lbl)
        side.setMaximumWidth(300)

        top = QSplitter(Qt.Orientation.Horizontal)
        top.addWidget(self.viewer)
        top.addWidget(side)
        top.setStretchFactor(0, 1)
        top.setStretchFactor(1, 0)

        self.model = ThumbModel(self)
        self.strip = QListView()
        self.strip.setModel(self.model)
        self.strip.setItemDelegate(ThumbDelegate(self))
        self.strip.setViewMode(QListView.ViewMode.IconMode)
        self.strip.setFlow(QListView.Flow.LeftToRight)
        self.strip.setWrapping(False)
        self.strip.setUniformItemSizes(True)
        self.strip.setResizeMode(QListView.ResizeMode.Adjust)
        self.strip.setMovement(QListView.Movement.Static)
        self.strip.setHorizontalScrollMode(QListView.ScrollMode.ScrollPerPixel)
        self.strip.setFixedHeight(ThumbDelegate.H + 22)
        self.strip.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.strip.clicked.connect(lambda mi: self.show_photo(mi.data(Qt.ItemDataRole.UserRole)))
        self.strip.setStyleSheet("QListView{background:#1c1c1c; border:none;}")

        self.banner = QWidget()
        bl = QHBoxLayout(self.banner)
        bl.setContentsMargins(14, 8, 8, 8)
        self.banner_lbl = QLabel()
        self.banner_lbl.setTextFormat(Qt.TextFormat.RichText)
        self.banner_lbl.setWordWrap(True)
        self.banner_apply = QPushButton("Zastosuj sugestie")
        self.banner_apply.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.banner_apply.clicked.connect(self.apply_suggestions)
        close_btn = QPushButton("✕")
        close_btn.setFixedWidth(30)
        close_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        close_btn.clicked.connect(lambda: self.banner.setVisible(False))
        bl.addWidget(self.banner_lbl, 1)
        bl.addWidget(self.banner_apply)
        bl.addWidget(close_btn)
        self.banner.setVisible(False)

        central = QWidget()
        cl = QVBoxLayout(central)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(0)
        cl.addWidget(self.banner)
        cl.addWidget(top, 1)
        cl.addWidget(self.strip)
        self.setCentralWidget(central)

        # status
        self.status_lbl = QLabel()
        self.progress = QProgressBar()
        self.progress.setMinimumWidth(380)
        self.progress.setVisible(False)
        self.progress.setTextVisible(True)
        sign = QLabel("made by PanRodla  ")
        sign.setStyleSheet("color:#666;")
        self.statusBar().addWidget(self.status_lbl, 1)
        self.statusBar().addPermanentWidget(self.progress)
        self.statusBar().addPermanentWidget(sign)

        # skróty bez przycisków
        def sc(key, fn):
            a = QAction(self)
            a.setShortcut(QKeySequence(key))
            a.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
            a.triggered.connect(fn)
            self.addAction(a)

        sc("Right", lambda: self.step(1))
        sc("Left", lambda: self.step(-1))
        sc("Home", lambda: self.goto_row(0))
        sc("End", lambda: self.goto_row(len(self.model.rows) - 1))
        for n in range(6):
            sc(str(n), lambda n=n: self.set_rating(n))
        for k, lab in zip("6789", ["Red", "Yellow", "Green", "Blue"]):
            sc(k, lambda lab=lab: self.toggle_label(lab))
        sc("X", self.toggle_reject)
        sc("O", self.toggle_rejected_view)
        sc("Z", lambda: self.viewer.set_zoom(not self.viewer.zoom))
        sc("Space", lambda: self.viewer.set_zoom(not self.viewer.zoom))
        sc("G", lambda: self.jump_group(1))
        sc("Shift+G", lambda: self.jump_group(-1))
        sc("B", self.jump_best)
        sc("Ctrl+Z", self.do_undo)
        sc("F11", self.toggle_fullscreen)
        sc("Escape", self._escape)

    def _load_settings(self):
        s = self.settings
        g = s.value("geometry")
        if g:
            self.restoreGeometry(g)
        else:
            self.resize(1500, 950)
        self.sens.blockSignals(True)
        self.sens.setValue(int(s.value("sens", 50)))
        self.sens.blockSignals(False)
        self.gap.blockSignals(True)
        self.gap.setValue(float(s.value("gap", 1.5)))
        self.gap.blockSignals(False)
        self.applied_sens, self.applied_gap = self.sens.value(), self.gap.value()
        self.sens_lbl.setText(f" {self.applied_sens}%")
        self.auto_adv.setChecked(s.value("auto_adv", "true") in (True, "true"))
        self.auto_xmp.setChecked(s.value("auto_xmp", "true") in (True, "true"))

    def closeEvent(self, e):
        self._save_now()
        if self.auto_xmp.isChecked():
            self._flush_xmp()
        s = self.settings
        s.setValue("geometry", self.saveGeometry())
        s.setValue("sens", self.sens.value())
        s.setValue("gap", self.gap.value())
        s.setValue("auto_adv", self.auto_adv.isChecked())
        s.setValue("auto_xmp", self.auto_xmp.isChecked())
        self.pool.clear()
        self.view_pool.clear()
        super().closeEvent(e)

    # ------------------------------------------------------------------ folder
    def open_folder_dialog(self):
        start = self.folder or self.settings.value("last_folder", "")
        d = QFileDialog.getExistingDirectory(self, "Wybierz folder z sesją", start)
        if d:
            self.open_folder(d)

    def open_folder(self, folder: str):
        if self.folder:
            self._save_now()
            if self.auto_xmp.isChecked():
                self._flush_xmp()
        self.gen += 1
        self.pool.clear()
        self.view_pool.clear()
        self.folder = folder
        self.settings.setValue("last_folder", folder)
        self.setWindowTitle(f"{APP_NAME} – {folder}")
        self.state = storage.load_state(folder)
        self.photos = []
        self.groups = []
        self.cur = None
        self.undo.clear()
        self.dirty_xmp.clear()
        self.view_cache.clear()
        self.full_cache = None
        self.pending_view.clear()
        self.viewer.set_pixmaps(None, None)
        self.model.set_rows([])
        self.banner.setVisible(False)
        self.analysis_running = False
        self._update_apply()

        names = sorted(n for n in os.listdir(folder)
                       if os.path.splitext(n)[1].lower() in raw.SUPPORTED_EXT
                       and os.path.isfile(os.path.join(folder, n)))
        # gdy jest RAW+JPG o tej samej nazwie - bierz RAW
        stems = {}
        for n in names:
            stem, ext = os.path.splitext(n)
            if stem in stems and ext.lower() in raw.JPG_EXT:
                continue
            stems[stem] = n
        names = sorted(stems.values())
        if not names:
            self.viewer.message = "W tym folderze nie ma obsługiwanych zdjęć (CR3, CR2, NEF, ARW, DNG, JPG…)"
            self.viewer.update()
            self._update_status()
            return
        self.viewer.message = "Wczytywanie…"

        saved = self.state["photos"]
        need_meta = []
        for n in names:
            path = os.path.join(folder, n)
            st = os.stat(path)
            p = Photo(path, n, st.st_size, st.st_mtime)
            s = saved.get(n, {})
            p.rating = int(s.get("r", 0))
            p.rejected = bool(s.get("x", False))
            p.rej_by = s.get("xb") if p.rejected else None
            p.label = s.get("l")
            if s.get("sz") == st.st_size and abs(s.get("mt", 0) - st.st_mtime) < 2 and "t" in s:
                p.meta = raw.PhotoMeta(s["t"], s.get("o", 1), s.get("po", 0), s.get("pl", 0))
                p.score = s.get("s") if s.get("sv") == analysis.METRIC_VERSION else None
            self.photos.append(p)
            if p.meta is None:
                need_meta.append(len(self.photos) - 1)

        self.meta_left = len(need_meta)
        self.analysis_t0 = time.time()
        self.analysis_cached = not need_meta and all(p.score is not None for p in self.photos)
        if need_meta:
            self.analysis_running = True
            self._update_apply()
            self._progress(0, len(need_meta), "Etap 1/2 – odczyt plików")
            for i in need_meta:
                self.pool.start(MetaTask(self.sig, self.gen, i, self.photos[i].path))
        else:
            self._after_meta()

    def _on_meta(self, gen, idx, meta):
        if gen != self.gen:
            return
        p = self.photos[idx]
        if meta is None:
            p.failed = True
            meta = raw.PhotoMeta(p.mtime)
        p.meta = meta
        p.score = None
        self.meta_left -= 1
        total = self.progress.maximum()
        self._progress(total - self.meta_left, total, "Etap 1/2 – odczyt plików")
        if self.meta_left == 0:
            self._after_meta()

    def _after_meta(self):
        # sortowanie wg czasu wykonania
        order = sorted(range(len(self.photos)), key=lambda i: (self.photos[i].meta.capture_time or 0, self.photos[i].name))
        self.photos = [self.photos[i] for i in order]
        self._regroup(recompute=False)
        self._recompute()
        self.apply_filter()
        if self.model.rows:
            self.show_photo(self.model.rows[0])
        self.viewer.message = ""
        self._save_state_cache()
        # analiza: miniatury + ostrość
        self.analyze_total = self.analyze_left = len(self.photos)
        self.analysis_running = True
        self.analysis_t1 = time.time()
        self._update_apply()
        self._progress(0, self.analyze_total, self._stage2_name())
        for i in range(len(self.photos)):
            p = self.photos[i]
            if p.failed:
                self.analyze_left -= 1
                continue
            self.pool.start(AnalyzeTask(self.sig, self.gen, i, p.path, p.meta, p.score is None))
        if self.analyze_left == 0:
            self._analysis_finished()

    def _stage2_name(self):
        return "Wczytywanie miniatur" if self.analysis_cached else "Etap 2/2 – analiza ostrości"

    def _on_analyzed(self, gen, idx, score, thumb):
        if gen != self.gen:
            return
        p = self.photos[idx]
        if thumb is None:
            p.failed = True
        else:
            p.thumb = thumb
        if score is not None:
            p.score = score
        self.model.refresh_photo(idx)
        self.analyze_left -= 1
        done = self.analyze_total - self.analyze_left
        self._progress(done, self.analyze_total, self._stage2_name(), eta_from=self.analysis_t1)
        if score is not None and done % 40 == 0:
            self.suggest_timer.start()
        if self.analyze_left == 0:
            self._analysis_finished()

    def _analysis_finished(self):
        self.analysis_running = False
        self.progress.setVisible(False)
        self._recompute()
        self._save_state_cache()
        n = len(self.photos)
        series = sum(1 for g in self.groups if len(g.members) > 1)
        blur = sum(1 for p in self.photos if p.suggested and p.reason == "blur" and not p.rejected)
        weak = sum(1 for p in self.photos if p.suggested and p.reason == "series" and not p.rejected)
        failed = sum(1 for p in self.photos if p.failed)
        secs = int(time.time() - self.analysis_t0)
        head = ("✔ <b>Analiza wczytana z poprzedniego razu</b>" if self.analysis_cached
                else f"✔ <b>Analiza zakończona</b> w {secs // 60}:{secs % 60:02d}")
        txt = (f"{head} — {n} zdjęć, {series} serii, {self.best_count()} najlepszych klatek (TOP).<br>"
               f"Do odrzutu: <b>{blur + weak}</b> ({blur} nieostrych, {weak} wyraźnie słabszych w serii).")
        if failed:
            txt += f"<br><span style='color:#e07070'>Nie udało się odczytać {failed} plików.</span>"
        if blur + weak == 0:
            txt += "<br>Nic nie wymaga odrzucenia – możesz od razu oceniać."
        self.banner_lbl.setText(txt)
        self.banner_apply.setVisible(blur + weak > 0)
        self.banner.setStyleSheet("background:#1f3d2a; color:#e6e6e6;")
        self.banner.setVisible(True)
        self.statusBar().showMessage("Analiza zakończona", 8000)
        QApplication.alert(self, 0)

    def best_count(self):
        return sum(1 for p in self.photos if p.best)

    # ------------------------------------------------------------------ serie / sugestie
    def _count_pending(self, sens: int, gap: float) -> int:
        """Ile zdjęć byłoby do odrzutu przy podanych ustawieniach (bez zmiany stanu)."""
        if not self.photos or any(p.meta is None for p in self.photos):
            return 0
        groups = analysis.build_groups([p.meta.capture_time or 0 for p in self.photos], gap)
        rej, _, _ = analysis.suggest([p.score for p in self.photos], groups, sens / 100)
        return sum(1 for i in rej if not self.photos[i].rejected and self.photos[i].rating == 0)

    def _settings_changed(self):
        self.sens_lbl.setText(f" {self.sens.value()}%")
        pending = (self.sens.value() != self.applied_sens or abs(self.gap.value() - self.applied_gap) > 1e-9)
        ready = bool(self.photos) and not self.analysis_running
        self.recalc_btn.setEnabled(pending and ready)
        if pending and ready:
            now = self._count_pending(self.applied_sens, self.applied_gap)
            new = self._count_pending(self.sens.value(), self.gap.value())
            diff = new - now
            self.preview_lbl.setText(f"po przeliczeniu: {new} do odrzutu ({'+' if diff >= 0 else ''}{diff})")
        elif pending and self.analysis_running:
            self.preview_lbl.setText("zostanie użyte po zakończeniu analizy")
        else:
            self.preview_lbl.setText("")

    def recalc(self):
        old_sens, old_gap = self.applied_sens, self.applied_gap
        before = sum(1 for p in self.photos if p.suggested and not p.rejected and p.rating == 0)
        self._regroup(recompute=False)
        self._recompute()
        self._settings_changed()
        after = sum(1 for p in self.photos if p.suggested and not p.rejected and p.rating == 0)
        series = sum(1 for g in self.groups if len(g.members) > 1)
        changes = []
        if self.applied_sens != old_sens:
            changes.append(f"czułość {old_sens}% → {self.applied_sens}%")
        if abs(self.applied_gap - old_gap) > 1e-9:
            changes.append(f"seria co {old_gap:g} s → {self.applied_gap:g} s ({series} serii)")
        txt = (f"🔄 <b>Przeliczono</b> ({', '.join(changes)}). "
               f"Sugerowane do odrzutu: <b>{after}</b> (było {before}).")
        if any(p.rejected and p.rej_by in ("blur", "series") for p in self.photos):
            txt += ("<br>Zdjęcia odrzucone wcześniej zostają odrzucone – przy niższej czułości "
                    "sprawdź je przyciskiem 🗑 Odrzucone (X przywraca).")
        self.banner_lbl.setText(txt)
        self.banner_apply.setVisible(after > 0)
        self.banner.setStyleSheet("background:#3d2f1f; color:#e6e6e6;")
        self.banner.setVisible(True)

    def _regroup(self, recompute=True):
        if not self.photos or any(p.meta is None for p in self.photos):
            return
        times = [p.meta.capture_time or 0 for p in self.photos]
        self.applied_gap = self.gap.value()
        self.groups = analysis.build_groups(times, self.applied_gap)
        for g in self.groups:
            for pos, i in enumerate(g.members):
                p = self.photos[i]
                p.group, p.group_pos, p.group_len = g.index, pos, len(g.members)
        if recompute:
            self._recompute()

    def _recompute(self):
        if not self.groups:
            return
        scores = [p.score for p in self.photos]
        self.applied_sens = self.sens.value()
        rejects, rel, reasons = analysis.suggest(scores, self.groups, self.applied_sens / 100)
        best = {g.best for g in self.groups if g.best is not None}
        for i, p in enumerate(self.photos):
            p.suggested = i in rejects
            p.reason = reasons.get(i)
            p.rel = rel.get(i)
            p.best = i in best
        if self.filter_box.currentIndex() in (8, 9):
            self.apply_filter()
        else:
            self.strip.viewport().update()
        self._update_info()
        self._update_status()
        self._update_apply()
        self._settings_changed()

    def _update_apply(self):
        if self.analysis_running:
            self.apply_act.setEnabled(False)
            self.apply_act.setText("⏳ Analiza trwa…")
            self.apply_act.setToolTip("Poczekaj na koniec analizy – wtedy sugestie będą kompletne")
            return
        n = sum(1 for p in self.photos if p.suggested and not p.rejected and p.rating == 0)
        self.apply_act.setEnabled(n > 0)
        self.apply_act.setText(f"✔ Zastosuj sugestie ({n})" if n else "✔ Brak sugestii")
        self.apply_act.setToolTip(f"Oznacz {n} sugerowanych zdjęć (nieostre / słabsze w serii) jako odrzucone")

    # ------------------------------------------------------------------ filtr / nawigacja
    REJ_FILTER = 7

    def reviewing_rejected(self) -> bool:
        return self.filter_box.currentIndex() == self.REJ_FILTER

    def toggle_rejected_view(self):
        if self.reviewing_rejected():
            self.filter_box.setCurrentIndex(getattr(self, "_prev_filter", 0))
        else:
            if not any(p.rejected for p in self.photos):
                self.rej_btn.setChecked(False)
                self.statusBar().showMessage("Brak odrzuconych zdjęć", 4000)
                return
            self._prev_filter = self.filter_box.currentIndex()
            self.cur = None
            self.filter_box.setCurrentIndex(self.REJ_FILTER)
            if self.model.rows:
                self.show_photo(self.model.rows[0])

    def apply_filter(self):
        pred = FILTERS[self.filter_box.currentIndex()][1]
        rows = [i for i, p in enumerate(self.photos) if pred(p) or i == self.cur]
        self.model.set_rows(rows)
        self._select_current()
        self.viewer.update()
        self._update_status()

    def _select_current(self):
        if self.cur is None:
            return
        try:
            r = self.model.rows.index(self.cur)
        except ValueError:
            return
        mi = self.model.index(r)
        self.strip.setCurrentIndex(mi)
        self.strip.scrollTo(mi, QListView.ScrollHint.PositionAtCenter)

    def current_photo(self) -> Photo | None:
        return self.photos[self.cur] if self.cur is not None and self.cur < len(self.photos) else None

    def current_row(self) -> int:
        try:
            return self.model.rows.index(self.cur)
        except ValueError:
            return -1

    def step(self, d: int):
        if not self.model.rows:
            return
        r = self.current_row()
        self.goto_row(max(0, min(len(self.model.rows) - 1, (r if r >= 0 else 0) + d)))

    def goto_row(self, r: int):
        if 0 <= r < len(self.model.rows):
            self.show_photo(self.model.rows[r])

    def jump_group(self, d: int):
        p = self.current_photo()
        if not p or not self.groups:
            return
        rows = self.model.rows
        r = self.current_row()
        rng = range(r + 1, len(rows)) if d > 0 else range(r - 1, -1, -1)
        target_group = None
        for k in rng:
            g = self.photos[rows[k]].group
            if g != p.group and target_group is None:
                target_group = g
                if d > 0:
                    self.goto_row(k)
                    return
            if d < 0 and target_group is not None and self.photos[rows[k]].group != target_group:
                self.goto_row(k + 1)
                return
        if d < 0 and target_group is not None:
            self.goto_row(0)

    def jump_best(self):
        p = self.current_photo()
        if p and 0 <= p.group < len(self.groups):
            b = self.groups[p.group].best
            if b is not None:
                if b not in self.model.rows:
                    self.filter_box.setCurrentIndex(0)
                self.show_photo(b)

    def show_photo(self, idx: int):
        if idx is None or not (0 <= idx < len(self.photos)):
            return
        # zoom i jego pozycja zostają przy zmianie zdjęcia - wygodne porównanie ostrości w serii
        self.cur = idx
        pix = self.view_cache.get(idx)
        full = self.full_cache[1] if self.full_cache and self.full_cache[0] == idx else None
        if pix is not None:
            self.view_cache.move_to_end(idx)
        thumb = self.photos[idx].thumb
        self.viewer.set_pixmaps(pix or (QPixmap.fromImage(thumb) if thumb is not None else None), full)
        if pix is None:
            self._request_view(idx, False)
        if self.viewer.zoom and full is None:
            self._request_view(idx, True)
        # prefetch
        r = self.current_row()
        for d in (1, 2, -1, 3):
            k = r + d
            if 0 <= k < len(self.model.rows):
                j = self.model.rows[k]
                if j not in self.view_cache:
                    self._request_view(j, False)
        self._select_current()
        self._update_info()

    def _view_edge(self) -> int:
        dpr = self.devicePixelRatioF()
        return int(min(3200, max(self.viewer.width(), self.viewer.height()) * dpr))

    def _request_view(self, idx: int, full: bool):
        key = (idx, full)
        p = self.photos[idx]
        if key in self.pending_view or p.failed or p.meta is None:
            return
        self.pending_view.add(key)
        self.view_pool.start(ViewTask(self.sig, self.gen, idx, p.path, p.meta, None if full else self._view_edge()),
                             2 if idx == self.cur else 0)

    def request_full(self):
        if self.cur is not None and not (self.full_cache and self.full_cache[0] == self.cur):
            self._request_view(self.cur, True)

    def current_full_width(self) -> int | None:
        if self.full_cache and self.full_cache[0] == self.cur:
            return self.full_cache[1].width()
        return None

    def _on_view(self, gen, idx, full, img):
        if gen != self.gen:
            return
        self.pending_view.discard((idx, full))
        if img is None:
            return
        pix = QPixmap.fromImage(img)
        if full:
            self.full_cache = (idx, pix)
        else:
            self.view_cache[idx] = pix
            self.view_cache.move_to_end(idx)
            while len(self.view_cache) > 10:
                self.view_cache.popitem(last=False)
        if idx == self.cur:
            self.viewer.set_pixmaps(self.view_cache.get(idx) or self.viewer.pix,
                                    self.full_cache[1] if self.full_cache and self.full_cache[0] == idx else None)

    # ------------------------------------------------------------------ oceny
    def _snapshot(self, idxs):
        return [(i, self.photos[i].rating, self.photos[i].rejected, self.photos[i].label, self.photos[i].rej_by)
                for i in idxs]

    def _changed(self, idxs, advance=False):
        for i in idxs:
            self.dirty_xmp.add(i)
            self.model.refresh_photo(i)
        self.save_timer.start()
        self.viewer.update()
        self._update_info()
        if advance and self.auto_adv.isChecked():
            self.step(1)
        if self.filter_box.currentIndex() != 0:
            self.apply_filter()
        else:
            self._update_status()
        self._update_apply()

    def set_rating(self, n: int):
        p = self.current_photo()
        if not p:
            return
        self.undo.append(self._snapshot([self.cur]))
        p.rating = n
        if n > 0:
            p.rejected = False
        self._changed([self.cur], advance=True)

    def toggle_reject(self):
        p = self.current_photo()
        if not p:
            return
        self.undo.append(self._snapshot([self.cur]))
        p.rejected = not p.rejected
        p.rej_by = "manual" if p.rejected else None
        if p.rejected:
            p.rating = 0
        self._changed([self.cur], advance=p.rejected or self.reviewing_rejected())

    def toggle_label(self, lab: str):
        p = self.current_photo()
        if not p:
            return
        self.undo.append(self._snapshot([self.cur]))
        p.label = None if p.label == lab else lab
        self._changed([self.cur])

    def do_undo(self):
        if not self.undo:
            return
        snap = self.undo.pop()
        for i, r, x, l, rb in snap:
            if i < len(self.photos):
                p = self.photos[i]
                p.rating, p.rejected, p.label, p.rej_by = r, x, l, rb
        idxs = [s[0] for s in snap if s[0] < len(self.photos)]
        if len(idxs) == 1:
            self.show_photo(idxs[0])
        self._changed(idxs)

    def apply_suggestions(self):
        idxs = [i for i, p in enumerate(self.photos) if p.suggested and not p.rejected and p.rating == 0]
        if self.analysis_running:
            return
        if not idxs:
            QMessageBox.information(self, APP_NAME, "Analiza zakończona – brak zdjęć do odrzucenia.")
            return
        blur = sum(1 for i in idxs if self.photos[i].reason == "blur")
        if QMessageBox.question(self, APP_NAME,
                                f"Oznaczyć {len(idxs)} zdjęć jako odrzucone?\n"
                                f"• {blur} nieostrych\n• {len(idxs) - blur} wyraźnie słabszych od najlepszego w serii\n\n"
                                f"Zdjęcia z oceną nie są ruszane. Można cofnąć Ctrl+Z.") \
                != QMessageBox.StandardButton.Yes:
            return
        self.undo.append(self._snapshot(idxs))
        for i in idxs:
            self.photos[i].rejected = True
            self.photos[i].rej_by = self.photos[i].reason or "blur"
        self._changed(idxs)
        self.banner_lbl.setText(f"✔ <b>Odrzucono {len(idxs)} zdjęć.</b> Zostało {sum(1 for p in self.photos if not p.rejected)} "
                                f"do przejrzenia. Cofnięcie: Ctrl+Z. Przycisk „Przenieś odrzucone” schowa je do _ODRZUCONE.")
        self.banner_apply.setVisible(False)
        self.banner.setStyleSheet("background:#1f3d2a; color:#e6e6e6;")
        self.banner.setVisible(True)

    # ------------------------------------------------------------------ zapis
    def _save_state_cache(self):
        self._save_now()

    def _save_now(self):
        if not self.folder:
            return
        out = {}
        for p in self.photos:
            d = {"sz": p.size, "mt": p.mtime}
            if p.meta and not p.failed:
                d.update(t=p.meta.capture_time, o=p.meta.orientation, po=p.meta.preview_offset, pl=p.meta.preview_length)
            if p.score is not None:
                d["s"] = round(p.score, 4)
                d["sv"] = analysis.METRIC_VERSION
            if p.rating:
                d["r"] = p.rating
            if p.rejected:
                d["x"] = True
                if p.rej_by:
                    d["xb"] = p.rej_by
            if p.label:
                d["l"] = p.label
            out[p.name] = d
        self.state = {"app": f"{APP_NAME} {VERSION}", "photos": out}
        try:
            storage.save_state(self.folder, self.state)
        except OSError as e:
            self.statusBar().showMessage(f"Nie udało się zapisać stanu: {e}", 6000)
        if self.auto_xmp.isChecked():
            self._flush_xmp()

    def _flush_xmp(self):
        errs = 0
        for i in list(self.dirty_xmp):
            if i >= len(self.photos):
                continue
            p = self.photos[i]
            if os.path.splitext(p.name)[1].lower() in raw.JPG_EXT:
                continue  # Lightroom nie czyta .xmp dla JPG
            try:
                storage.write_xmp(p.path, p.rating, p.rejected, p.label)
            except OSError:
                errs += 1
        self.dirty_xmp.clear()
        if errs:
            self.statusBar().showMessage(f"Błąd zapisu {errs} plików XMP", 6000)

    def save_all_xmp(self):
        if not self.photos:
            return
        self.dirty_xmp.update(i for i, p in enumerate(self.photos) if p.rating or p.rejected or p.label
                              or os.path.exists(storage.xmp_path(p.path)))
        n = len(self.dirty_xmp)
        self._save_now()
        self._flush_xmp()
        self.statusBar().showMessage(f"Zapisano XMP dla {n} zdjęć", 5000)

    # ------------------------------------------------------------------ eksport
    def _selected_by_dialog(self, title):
        dlg = ExportDialog(self, title)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return None
        pred = dlg.predicate()
        return [p for p in self.photos if pred(p)]

    def export(self, _move: bool):
        if not self.photos:
            return
        sel = self._selected_by_dialog("Kopiuj wybrane")
        if sel is None:
            return
        if not sel:
            QMessageBox.information(self, APP_NAME, "Żadne zdjęcie nie spełnia warunków.")
            return
        dest = QFileDialog.getExistingDirectory(self, f"Folder docelowy ({len(sel)} zdjęć)", self.folder)
        if not dest:
            return
        if os.path.normcase(os.path.abspath(dest)) == os.path.normcase(os.path.abspath(self.folder)):
            QMessageBox.warning(self, APP_NAME, "Folder docelowy musi być inny niż folder sesji.")
            return
        self.save_all_xmp()
        self._run_fileop([p.path for p in sel], dest, move=False, title="Kopiowanie")

    def move_rejected(self):
        sel = [p for p in self.photos if p.rejected]
        if not sel:
            QMessageBox.information(self, APP_NAME, "Brak odrzuconych zdjęć.")
            return
        dest = os.path.join(self.folder, "_ODRZUCONE")
        if QMessageBox.question(self, APP_NAME,
                                f"Przenieść {len(sel)} odrzuconych zdjęć do podfolderu „_ODRZUCONE”?\n"
                                "Nic nie jest kasowane – możesz je przejrzeć i usunąć sam.") \
                != QMessageBox.StandardButton.Yes:
            return
        self.save_all_xmp()
        self._run_fileop([p.path for p in sel], dest, move=True, title="Przenoszenie")

    def _run_fileop(self, paths, dest, move, title):
        sig = Signals()
        task = FileOpTask(sig, paths, dest, move)
        dlg = QProgressDialog(f"{title}…", "Przerwij", 0, len(paths), self)
        dlg.setWindowTitle(APP_NAME)
        dlg.setWindowModality(Qt.WindowModality.WindowModal)
        dlg.setMinimumDuration(0)
        dlg.canceled.connect(lambda: setattr(task, "cancelled", True))
        sig.progress.connect(lambda d, t, n: (dlg.setValue(d), dlg.setLabelText(f"{title}: {n}")))

        def done(ok, failed, errors):
            dlg.close()
            msg = f"{title} zakończone: {ok} zdjęć."
            if failed:
                msg += f"\nBłędy ({failed}):\n" + "\n".join(errors[:10])
            QMessageBox.information(self, APP_NAME, msg)
            if move:
                self.open_folder(self.folder)
        sig.finished.connect(done)
        self._fileop_sig = sig  # utrzymanie referencji
        self.pool.start(task, 10)

    def copy_names(self):
        sel = self._selected_by_dialog("Lista nazw plików")
        if not sel:
            return
        text = "\n".join(os.path.splitext(p.name)[0] for p in sel)
        QGuiApplication.clipboard().setText(text)
        self.statusBar().showMessage(f"Skopiowano {len(sel)} nazw do schowka", 5000)

    # ------------------------------------------------------------------ info
    def _progress(self, done, total, what, eta_from=None):
        self.progress.setVisible(done < total)
        self.progress.setMaximum(max(1, total))
        self.progress.setValue(done)
        eta = ""
        if eta_from and done >= 5 and done < total:
            left = (time.time() - eta_from) / done * (total - done)
            eta = f" – zostało ok. {int(left // 60)}:{int(left % 60):02d}" if left >= 60 else f" – zostało ok. {int(left) + 1} s"
        self.progress.setFormat(f"{what}: {done}/{total}{eta}")

    def _update_status(self):
        n = len(self.photos)
        rated = sum(1 for p in self.photos if p.rating and not p.rejected)
        rej = sum(1 for p in self.photos if p.rejected)
        sug = sum(1 for p in self.photos if p.suggested and not p.rejected)
        series = sum(1 for g in self.groups if len(g.members) > 1)
        shown = len(self.model.rows)
        self.rej_btn.setText(f"🗑 Odrzucone ({rej})")
        self.rej_btn.setChecked(self.reviewing_rejected())
        self.status_lbl.setText(
            f"  Zdjęć: {n}   •   widoczne: {shown}   •   serii: {series}   •   ocenione: {rated}   •   "
            f"odrzucone: {rej}   •   sugerowane do odrzutu: {sug}"
            + ("   (analiza w toku – liczby jeszcze się zmienią)" if self.analysis_running else ""))

    def _update_info(self):
        p = self.current_photo()
        if not p:
            self.info.setText("")
            return
        from datetime import datetime
        t = datetime.fromtimestamp(p.meta.capture_time).strftime("%d.%m.%Y  %H:%M:%S.%f")[:-4] \
            if p.meta and p.meta.capture_time else "—"
        rows = [f"<div style='font-size:15px'><b>{p.name}</b></div>",
                f"<div style='color:#999'>{t}</div><br>"]
        if p.group_len > 1:
            rows.append(f"Seria #{p.group + 1}: zdjęcie <b>{p.group_pos + 1}</b> z {p.group_len}<br>")
        else:
            rows.append("Pojedyncze zdjęcie<br>")
        if p.rel is not None:
            pct = int(round(p.rel * 100))
            t_abs = 0.50 - 0.25 * self.applied_sens / 100
            col = "#5cb85c" if 1 - p.rel <= t_abs * 0.75 else ("#e8c547" if 1 - p.rel <= t_abs else "#d9534f")
            bar = min(100, pct)
            rows.append(f"<br>Ostrość: <b style='color:{col}'>{pct}%</b>"
                        f"<table width='100%' cellspacing=0 cellpadding=0><tr>"
                        f"<td width='{bar}%' style='background:{col}' height=6></td>"
                        f"<td style='background:#333'></td></tr></table>"
                        f"<span style='color:#777;font-size:11px'>najostrzejsze miejsce kadru (np. oczy) – tło i bokeh nie mają wpływu</span><br>")
        elif p.failed:
            rows.append("<br><span style='color:#d9534f'>Nie udało się odczytać podglądu</span><br>")
        else:
            rows.append("<br><span style='color:#777'>Ostrość: analiza…</span><br>")
        if p.best:
            rows.append("<br><b style='color:#e0b020'>★ Najlepsze w serii</b><br>")
        if p.suggested and not p.rejected:
            if p.reason == "series":
                rows.append("<br><b style='color:#d9822b'>⚠ Wyraźnie słabsze od najlepszego w serii</b><br>")
            else:
                rows.append("<br><b style='color:#d9822b'>⚠ Prawdopodobnie nieostre</b><br>")
        rows.append("<br>")
        if p.rejected:
            rows.append("<b style='color:#e04646'>✕ Odrzucone</b>"
                        + (f"<br><span style='color:#bbb'>przez: {REJ_PL[p.rej_by]}</span>" if p.rej_by in REJ_PL else "")
                        + "<br><span style='color:#888'>X – przywróć</span>")
        elif p.rating:
            rows.append(f"<span style='color:#f0c419;font-size:18px'>{'★' * p.rating}</span>"
                        f"<span style='color:#444;font-size:18px'>{'★' * (5 - p.rating)}</span>")
        else:
            rows.append("<span style='color:#777'>Bez oceny</span>")
        if p.label:
            rows.append(f"<br><span style='color:{LABEL_COLORS[p.label]}'>■</span> etykieta {LABEL_PL[p.label]}")
        self.info.setText("".join(rows))

    def toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def _escape(self):
        if self.viewer.zoom:
            self.viewer.set_zoom(False)
        elif self.isFullScreen():
            self.showNormal()


def _dark_palette(app: QApplication):
    app.setStyle("Fusion")
    pal = QPalette()
    c = {
        QPalette.ColorRole.Window: "#232323", QPalette.ColorRole.WindowText: "#e6e6e6",
        QPalette.ColorRole.Base: "#1b1b1b", QPalette.ColorRole.AlternateBase: "#262626",
        QPalette.ColorRole.Text: "#e6e6e6", QPalette.ColorRole.Button: "#2d2d2d",
        QPalette.ColorRole.ButtonText: "#e6e6e6", QPalette.ColorRole.Highlight: "#4a7fb5",
        QPalette.ColorRole.HighlightedText: "#ffffff", QPalette.ColorRole.ToolTipBase: "#2d2d2d",
        QPalette.ColorRole.ToolTipText: "#e6e6e6",
    }
    for role, col in c.items():
        pal.setColor(role, QColor(col))
    app.setPalette(pal)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName("PanRodla")
    _dark_palette(app)
    w = MainWindow()
    w.show()
    if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]):
        w.open_folder(sys.argv[1])
    sys.exit(app.exec())
