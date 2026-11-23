"""Okna galerii dla klientów: ustawienia, tworzenie, lista galerii, wczytywanie wyboru."""
from __future__ import annotations

import os
import time
import webbrowser

from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QHBoxLayout, QLabel,
                               QLineEdit, QMessageBox, QPlainTextEdit, QProgressDialog, QPushButton, QSpinBox,
                               QTableWidget, QTableWidgetItem, QVBoxLayout, QAbstractItemView, QHeaderView)

from . import gallery

APP = "Fotobudka"


class _Sig(QObject):
    progress = Signal(int, int, str)
    done = Signal(object, str)          # wynik, błąd


class _Job(QRunnable):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.sig = _Sig()
        self.cancelled = False

    def run(self):
        try:
            r = self.fn(lambda d, t, s: self.sig.progress.emit(d, t, s), lambda: self.cancelled)
            self.sig.done.emit(r, "")
        except Exception as e:  # noqa: BLE001
            self.sig.done.emit(None, str(e) or e.__class__.__name__)


def run_job(parent, title: str, fn, on_done):
    job = _Job(fn)
    dlg = QProgressDialog(title, "Przerwij", 0, 0, parent)
    dlg.setWindowTitle(APP)
    dlg.setWindowModality(Qt.WindowModality.WindowModal)
    dlg.setMinimumDuration(0)
    dlg.canceled.connect(lambda: setattr(job, "cancelled", True))
    job.sig.progress.connect(lambda d, t, s: (dlg.setMaximum(t), dlg.setValue(d), dlg.setLabelText(f"{s}: {d} / {t}")))

    def fin(r, err):
        dlg.close()
        on_done(r, err)
    job.sig.done.connect(fin)
    parent._gallery_job = job
    QThreadPool.globalInstance().start(job)


# --------------------------------------------------------------------------- ustawienia
class SettingsDialog(QDialog):
    def __init__(self, parent, reg: dict):
        super().__init__(parent)
        self.setWindowTitle("Ustawienia galerii")
        self.reg = reg
        s = reg["settings"]
        lay = QFormLayout(self)
        lay.addRow(QLabel("<b>GitHub (gdzie leżą galerie)</b>"))
        self.owner = QLineEdit(s.get("owner", "barejmariusz-blip"))
        self.repo = QLineEdit(s.get("repo", "galerie"))
        self.token = QLineEdit(s.get("token", ""))
        self.token.setEchoMode(QLineEdit.EchoMode.Password)
        self.token.setPlaceholderText("github_pat_…")
        lay.addRow("Użytkownik:", self.owner)
        self.repo.setPlaceholderText("nazwa repo albo cały link z GitHuba")
        lay.addRow("Repozytorium:", self.repo)
        lay.addRow("Token:", self.token)
        lay.addRow(QLabel("<span style='color:#888'>Token zostaje tylko na tym komputerze.</span>"))
        lay.addRow(QLabel("<b>Dla klienta</b>"))
        self.studio = QLineEdit(s.get("studio", ""))
        self.studio.setPlaceholderText("np. Liza Billewicz Studio")
        self.wm = QLineEdit(s.get("watermark", s.get("studio", "")))
        self.wm.setPlaceholderText("tekst znaku wodnego (puste = bez znaku)")
        self.email = QLineEdit(s.get("email", ""))
        self.phone = QLineEdit(s.get("phone", ""))
        self.phone.setPlaceholderText("+48…  (do SMS / WhatsApp)")
        lay.addRow("Nazwa studia:", self.studio)
        lay.addRow("Znak wodny:", self.wm)
        lay.addRow("E-mail do wyboru:", self.email)
        lay.addRow("Telefon do wyboru:", self.phone)
        test = QPushButton("Sprawdź połączenie")
        test.clicked.connect(self.test)
        lay.addRow(test)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.save)
        bb.rejected.connect(self.reject)
        lay.addRow(bb)

    def values(self):
        owner, repo = gallery.parse_repo(self.owner.text(), self.repo.text())
        return {"owner": owner, "repo": repo, "token": self.token.text().strip(),
                "studio": self.studio.text().strip(), "watermark": self.wm.text().strip(),
                "email": self.email.text().strip(), "phone": self.phone.text().strip()}

    def test(self):
        v = self.values()
        try:
            gh = gallery.GitHub(v["owner"], v["repo"], v["token"])
            branch = gh.check()
            if gh.repo != v["repo"] or gh.owner != v["owner"]:
                self.owner.setText(gh.owner)
                self.repo.setText(gh.repo)
                v = self.values()
            gh.ensure_initialized(branch)
            gh.ensure_pages(branch)
            QMessageBox.information(self, APP, f"Działa ✔\nGalerie będą pod adresem:\n{gallery.site_url(v)}")
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, APP, f"Nie działa:\n{e}")

    def save(self):
        self.reg["settings"].update(self.values())
        gallery.save_registry(self.reg)
        self.accept()


def ensure_settings(parent, reg) -> bool:
    s = reg["settings"]
    if s.get("owner") and s.get("repo") and s.get("token"):
        return True
    QMessageBox.information(parent, APP, "Najpierw ustaw, gdzie mają leżeć galerie (GitHub) – zrobisz to raz.")
    return SettingsDialog(parent, reg).exec() == QDialog.DialogCode.Accepted


# --------------------------------------------------------------------------- nowa galeria
class CreateDialog(QDialog):
    def __init__(self, parent, n_visible: int, n_rated: dict, default_title: str):
        super().__init__(parent)
        self.setWindowTitle("Nowa galeria dla klienta")
        lay = QFormLayout(self)
        self.title = QLineEdit(default_title)
        self.src = QComboBox()
        self.src.addItem(f"zdjęcia widoczne teraz na pasku ({n_visible})", "visible")
        for r in (1, 2, 3, 4, 5):
            if n_rated.get(r):
                self.src.addItem(f"ocenione ≥ {r}★ ({n_rated[r]})", f"r{r}")
        self.src.addItem(f"bez odrzuconych ({n_rated.get(0, 0)})", "r0")
        self.max = QSpinBox()
        self.max.setRange(0, 999)
        self.max.setSpecialValueText("bez limitu")
        self.max.setSuffix(" zdjęć")
        self.note = QPlainTextEdit()
        self.note.setPlaceholderText("Wiadomość dla klienta, np. „Wybierz 15 zdjęć do obróbki. Galeria będzie aktywna 14 dni.”")
        self.note.setFixedHeight(70)
        lay.addRow("Tytuł (widzi klient):", self.title)
        lay.addRow("Zdjęcia:", self.src)
        lay.addRow("Pakiet:", self.max)
        lay.addRow("Wiadomość:", self.note)
        bb = QDialogButtonBox()
        bb.addButton("Opublikuj", QDialogButtonBox.ButtonRole.AcceptRole)
        bb.addButton("Anuluj", QDialogButtonBox.ButtonRole.RejectRole)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addRow(bb)


class LinkDialog(QDialog):
    def __init__(self, parent, link: str, title: str, fresh: bool):
        super().__init__(parent)
        self.setWindowTitle(APP)
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel(f"<b>Galeria „{title}”</b>" + (" jest gotowa ✔" if fresh else "")))
        if fresh:
            lay.addWidget(QLabel("<span style='color:#888'>Przy pierwszej galerii GitHub potrzebuje 1–2 minut, "
                                 "zanim strona zacznie działać.</span>"))
        ed = QLineEdit(link)
        ed.setReadOnly(True)
        ed.setMinimumWidth(560)
        lay.addWidget(ed)
        h = QHBoxLayout()
        cp = QPushButton("📋 Kopiuj link")
        cp.clicked.connect(lambda: (QGuiApplication.clipboard().setText(link), cp.setText("Skopiowano ✔")))
        op = QPushButton("🌐 Otwórz")
        op.clicked.connect(lambda: webbrowser.open(link))
        ok = QPushButton("Zamknij")
        ok.clicked.connect(self.accept)
        h.addWidget(cp)
        h.addWidget(op)
        h.addStretch(1)
        h.addWidget(ok)
        lay.addLayout(h)
        lay.addWidget(QLabel("<span style='color:#888'>Wyślij ten link klientowi (SMS, Messenger, e-mail). "
                             "Bez całego linku galerii nie da się otworzyć.</span>"))


# --------------------------------------------------------------------------- lista galerii
class ManagerDialog(QDialog):
    def __init__(self, win):
        super().__init__(win)
        self.win = win
        self.setWindowTitle("Galerie dla klientów")
        self.resize(860, 440)
        lay = QVBoxLayout(self)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Tytuł", "Data", "Zdjęć", "Pakiet", "Folder sesji"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self.table.doubleClicked.connect(lambda _: self.link())
        lay.addWidget(self.table)
        h = QHBoxLayout()
        for text, fn in (("📋 Link", self.link), ("🌐 Otwórz", self.open), ("📥 Wczytaj wybór klienta", self.import_sel),
                         ("🗑 Usuń galerię", self.delete)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            h.addWidget(b)
        h.addStretch(1)
        st = QPushButton("⚙ Ustawienia")
        st.clicked.connect(lambda: SettingsDialog(self, self.win.gallery_reg).exec())
        h.addWidget(st)
        cl = QPushButton("🧹 Wyczyść historię")
        cl.setToolTip("Usunięte galerie zostają w historii repozytorium (zaszyfrowane). To zwalnia miejsce.")
        cl.clicked.connect(self.cleanup)
        h.addWidget(cl)
        lay.addLayout(h)
        self.fill()

    def fill(self):
        gs = self.win.gallery_reg["galleries"]
        self.table.setRowCount(len(gs))
        for r, g in enumerate(reversed(gs)):
            vals = [g["title"], time.strftime("%d.%m.%Y %H:%M", time.localtime(g["created"])), str(g["count"]),
                    str(g.get("max") or "–"), g.get("folder", "")]
            for c, v in enumerate(vals):
                it = QTableWidgetItem(v)
                it.setData(Qt.ItemDataRole.UserRole, g["id"])
                self.table.setItem(r, c, it)
        if gs:
            self.table.selectRow(0)

    def current(self):
        r = self.table.currentRow()
        if r < 0:
            return None
        gid = self.table.item(r, 0).data(Qt.ItemDataRole.UserRole)
        return next((g for g in self.win.gallery_reg["galleries"] if g["id"] == gid), None)

    def link(self):
        g = self.current()
        if g:
            LinkDialog(self, gallery.gallery_link(self.win.gallery_reg["settings"], g), g["title"], False).exec()

    def open(self):
        g = self.current()
        if g:
            webbrowser.open(gallery.gallery_link(self.win.gallery_reg["settings"], g))

    def import_sel(self):
        g = self.current()
        if g:
            self.win.import_selection(g)

    def delete(self):
        g = self.current()
        if not g:
            return
        if QMessageBox.question(self, APP, f"Usunąć galerię „{g['title']}”?\nLink przestanie działać.") \
                != QMessageBox.StandardButton.Yes:
            return
        reg = self.win.gallery_reg

        def done(_, err):
            if err:
                QMessageBox.warning(self, APP, f"Nie udało się usunąć:\n{err}")
                return
            reg["galleries"] = [x for x in reg["galleries"] if x["id"] != g["id"]]
            gallery.save_registry(reg)
            self.fill()
        run_job(self, "Usuwanie galerii…", lambda p, c: gallery.delete(reg["settings"], g["id"]), done)

    def cleanup(self):
        if QMessageBox.question(self, APP, "Usunąć historię repozytorium (zostają tylko aktualne galerie)?") \
                != QMessageBox.StandardButton.Yes:
            return
        run_job(self, "Porządki…", lambda p, c: gallery.cleanup_history(self.win.gallery_reg["settings"]),
                lambda _, err: QMessageBox.information(self, APP, err or "Gotowe ✔"))


class SelectionDialog(QDialog):
    def __init__(self, parent, title: str):
        super().__init__(parent)
        self.setWindowTitle(f"Wybór klienta – {title}")
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Wklej wiadomość od klienta (SMS / e-mail / WhatsApp):"))
        self.text = QPlainTextEdit()
        self.text.setMinimumSize(520, 200)
        lay.addWidget(self.text)
        f = QFormLayout()
        self.mark = QComboBox()
        self.mark.addItem("zielona etykieta", "Green")
        self.mark.addItem("ocena 5★", "r5")
        self.mark.addItem("zielona etykieta + 5★", "both")
        f.addRow("Oznacz wybrane:", self.mark)
        self.show_only = QCheckBox("pokaż tylko wybrane")
        self.show_only.setChecked(True)
        f.addRow("", self.show_only)
        lay.addLayout(f)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
