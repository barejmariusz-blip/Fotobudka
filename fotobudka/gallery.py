"""Galerie dla klientów: zaszyfrowane podglądy na GitHub Pages, link z kluczem w części po „#”.

Struktura strony (repo np. `galerie`, gałąź main, GitHub Pages z katalogu głównego):
    index.html                   - przeglądarka galerii (wspólna dla wszystkich)
    robots.txt                   - zakaz indeksowania
    g/<id>/m.bin                 - opis galerii (tytuł, lista zdjęć...) - zaszyfrowany
    g/<id>/t/<n>.bin, p/<n>.bin  - miniatura / podgląd - zaszyfrowane (AES-256-GCM)

Link dla klienta: https://<user>.github.io/<repo>/?g=<id>#<klucz>
Klucz nigdy nie trafia na serwer (część po „#” nie jest wysyłana przez przeglądarkę),
więc nawet ktoś, kto przegląda repozytorium, widzi tylko zaszyfrowane pliki.
"""
from __future__ import annotations

import base64
import io
import json
import os
import secrets
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from PIL import Image, ImageDraw, ImageFont

from . import analysis, raw

PREVIEW_EDGE = 1600
THUMB_EDGE = 420
API = "https://api.github.com"


# --------------------------------------------------------------------------- lokalny rejestr
def registry_path() -> str:
    base = os.environ.get("APPDATA") or os.path.join(os.path.expanduser("~"), ".config")
    d = os.path.join(base, "PanRodla", "Fotobudka")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "galerie.json")


def load_registry() -> dict:
    try:
        with open(registry_path(), "r", encoding="utf-8") as f:
            d = json.load(f)
            d.setdefault("galleries", [])
            d.setdefault("settings", {})
            return d
    except (OSError, ValueError):
        return {"galleries": [], "settings": {}}


def save_registry(reg: dict) -> None:
    p = registry_path()
    with open(p + ".tmp", "w", encoding="utf-8") as f:
        json.dump(reg, f, ensure_ascii=False, indent=1)
    os.replace(p + ".tmp", p)


def parse_repo(owner: str, repo: str) -> tuple[str, str]:
    """Akceptuje nazwę, nazwę ze spacjami albo wklejony link https://github.com/user/repo(.git)."""
    import re
    owner, repo = owner.strip(), repo.strip()
    m = re.search(r"github\.com[/:]([^/\s]+)/([^/\s#?]+)", repo)
    if m:
        owner, repo = m.group(1), m.group(2)
    if repo.endswith(".git"):
        repo = repo[:-4]
    repo = re.sub(r"[^A-Za-z0-9._-]+", "-", repo).strip("-")   # GitHub zamienia spacje na „-”
    owner = re.sub(r"[^A-Za-z0-9-]+", "", owner)
    return owner, repo


def b64u(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode().rstrip("=")


def site_url(s: dict) -> str:
    owner, repo = s["owner"], s["repo"]
    if repo.lower() == f"{owner.lower()}.github.io":
        return f"https://{owner.lower()}.github.io/"
    return f"https://{owner.lower()}.github.io/{repo}/"


def gallery_link(s: dict, g: dict) -> str:
    return f"{site_url(s)}?g={g['id']}#{g['key']}"


# --------------------------------------------------------------------------- obrazy
def _font(size: int):
    for name in ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size)
    except TypeError:
        return ImageFont.load_default()


def watermark(im: Image.Image, text: str) -> Image.Image:
    if not text:
        return im
    im = im.convert("RGBA")
    layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    size = max(14, int(min(im.size) * 0.035))
    f = _font(size)
    box = d.textbbox((0, 0), text, font=f)
    tw, th = box[2] - box[0], box[3] - box[1]
    m = int(size * 0.8)
    x, y = im.width - tw - m, im.height - th - m - box[1]
    d.text((x + 1, y + 1), text, font=f, fill=(0, 0, 0, 90))
    d.text((x, y), text, font=f, fill=(255, 255, 255, 150))
    # dyskretny znak na środku utrudnia wycięcie
    big = _font(int(min(im.size) * 0.09))
    b = d.textbbox((0, 0), text, font=big)
    d.text(((im.width - (b[2] - b[0])) / 2, (im.height - (b[3] - b[1])) / 2 - b[1]), text, font=big,
           fill=(255, 255, 255, 28))
    return Image.alpha_composite(im, layer).convert("RGB")


def _jpeg(im: Image.Image, q: int) -> bytes:
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=q, optimize=True, progressive=True)
    return buf.getvalue()


def make_images(path: str, meta, wm: str) -> tuple[bytes, bytes, tuple[int, int]]:
    data = raw.read_preview_bytes(path, meta)
    im = analysis.open_jpeg(data, PREVIEW_EDGE, meta.orientation)
    if max(im.size) > PREVIEW_EDGE:
        im.thumbnail((PREVIEW_EDGE, PREVIEW_EDGE), Image.Resampling.LANCZOS)
    prev = watermark(im, wm)
    th = im.copy()
    th.thumbnail((THUMB_EDGE, THUMB_EDGE), Image.Resampling.LANCZOS)
    return _jpeg(prev, 82), _jpeg(th, 78), prev.size


# --------------------------------------------------------------------------- szyfrowanie
def encrypt(key: bytes, data: bytes) -> bytes:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    iv = secrets.token_bytes(12)
    return iv + AESGCM(key).encrypt(iv, data, None)


def decrypt(key: bytes, blob: bytes) -> bytes:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    return AESGCM(key).decrypt(blob[:12], blob[12:], None)


def build_files(items: list[tuple[str, object, str]], info: dict, wm: str, progress=None,
                cancelled=lambda: False) -> tuple[str, str, dict[str, bytes]]:
    """items: [(ścieżka, meta, nazwa wyświetlana)] -> (id, klucz b64u, {ścieżka_w_repo: bajty})."""
    gid = b64u(secrets.token_bytes(9))
    key = secrets.token_bytes(32)
    files: dict[str, bytes] = {}
    photos = []

    def one(j_item):
        j, (path, meta, name) = j_item
        p, t, size = make_images(path, meta, wm)
        return j, name, p, t, size

    done = 0
    with ThreadPoolExecutor(max_workers=max(2, (os.cpu_count() or 4) - 1)) as ex:
        for j, name, p, t, size in ex.map(one, enumerate(items)):
            if cancelled():
                raise RuntimeError("Przerwano")
            files[f"g/{gid}/p/{j}.bin"] = encrypt(key, p)
            files[f"g/{gid}/t/{j}.bin"] = encrypt(key, t)
            photos.append({"n": name, "w": size[0], "h": size[1]})
            done += 1
            if progress:
                progress(done, len(items), "Przygotowanie zdjęć")
    manifest = dict(info)
    manifest["photos"] = photos
    manifest["created"] = int(time.time())
    files[f"g/{gid}/m.bin"] = encrypt(key, json.dumps(manifest, ensure_ascii=False).encode("utf-8"))
    return gid, b64u(key), files


def viewer_html() -> bytes:
    base = getattr(sys, "_MEIPASS", None)
    cands = [os.path.join(base, "fotobudka", "data", "gallery_index.html")] if base else []
    cands.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "gallery_index.html"))
    for p in cands:
        if os.path.exists(p):
            with open(p, "rb") as f:
                return f.read()
    raise FileNotFoundError("gallery_index.html")


# --------------------------------------------------------------------------- GitHub
class GitHubError(RuntimeError):
    pass


class GitHub:
    def __init__(self, owner: str, repo: str, token: str):
        self.owner, self.repo, self.token = owner, repo, token

    def req(self, method: str, path: str, body=None, ok=(200, 201, 204)):
        url = path if path.startswith("http") else f"{API}/repos/{self.owner}/{self.repo}{path}"
        data = json.dumps(body).encode() if body is not None else None
        r = urllib.request.Request(url, data=data, method=method, headers={
            "Authorization": f"Bearer {self.token}", "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "Fotobudka (PanRodla)",
            **({"Content-Type": "application/json"} if data else {})})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(r, timeout=120) as resp:
                    txt = resp.read()
                    return json.loads(txt) if txt else {}
            except urllib.error.HTTPError as e:
                msg = e.read().decode("utf-8", "ignore")
                if e.code in ok:
                    return json.loads(msg) if msg else {}
                if e.code in (502, 503, 504) or (e.code == 403 and "secondary rate" in msg.lower()):
                    time.sleep(3 * (attempt + 1))
                    continue
                raise GitHubError(f"{method} {path}: HTTP {e.code} {_short(msg)}")
            except urllib.error.URLError as e:
                if attempt == 3:
                    raise GitHubError(f"Brak połączenia z GitHubem: {e.reason}")
                time.sleep(2)

    def check(self) -> str:
        """Sprawdza dostęp; zwraca domyślną gałąź."""
        try:
            r = self.req("GET", "")
        except GitHubError as e:
            if "404" in str(e):
                raise GitHubError(f"Nie widzę repozytorium {self.owner}/{self.repo}. Sprawdź nazwę i czy token ma do niego dostęp.")
            if "401" in str(e):
                raise GitHubError("Token jest nieprawidłowy albo wygasł.")
            raise
        full = r.get("full_name", "")
        if "/" in full:
            self.owner, self.repo = full.split("/", 1)       # prawdziwa nazwa (wielkość liter)
        if r.get("private"):
            raise GitHubError("Repozytorium jest prywatne – GitHub Pages za darmo działa tylko dla publicznych.")
        return r.get("default_branch") or "main"

    def head(self, branch: str):
        try:
            ref = self.req("GET", f"/git/ref/heads/{branch}")
        except GitHubError as e:
            if "409" in str(e) or "404" in str(e):
                return None
            raise
        sha = ref["object"]["sha"]
        tree = self.req("GET", f"/git/commits/{sha}")["tree"]["sha"]
        return sha, tree

    def ensure_initialized(self, branch: str):
        if self.head(branch) is None:
            self.req("PUT", "/contents/robots.txt", {
                "message": "Galerie: start", "branch": branch,
                "content": base64.b64encode(b"User-agent: *\nDisallow: /\n").decode()})

    def blob(self, data: bytes) -> str:
        return self.req("POST", "/git/blobs", {"content": base64.b64encode(data).decode(), "encoding": "base64"})["sha"]

    def commit_files(self, branch: str, add: dict[str, bytes], remove_prefixes: list[str], message: str,
                     progress=None, cancelled=lambda: False):
        self.ensure_initialized(branch)
        parent, base_tree = self.head(branch)
        entries = []
        if remove_prefixes:
            # pełne drzewo bez usuwanych katalogów
            full = self.req("GET", f"/git/trees/{base_tree}?recursive=1")
            if full.get("truncated"):
                raise GitHubError("Repozytorium jest za duże do edycji – użyj „Wyczyść historię”.")
            for e in full["tree"]:
                if e["type"] != "blob" or any(e["path"].startswith(p) for p in remove_prefixes) or e["path"] in add:
                    continue
                entries.append({"path": e["path"], "mode": e["mode"], "type": "blob", "sha": e["sha"]})
            base = None
        else:
            base = base_tree
        items = list(add.items())
        done = 0
        with ThreadPoolExecutor(max_workers=6) as ex:
            futs = {ex.submit(self.blob, data): path for path, data in items}
            for fut in futs:
                if cancelled():
                    raise RuntimeError("Przerwano")
                entries.append({"path": futs[fut], "mode": "100644", "type": "blob", "sha": fut.result()})
                done += 1
                if progress:
                    progress(done, len(items), "Wysyłanie na GitHub")
        body = {"tree": entries}
        if base:
            body["base_tree"] = base
        tree = self.req("POST", "/git/trees", body)["sha"]
        commit = self.req("POST", "/git/commits", {"message": message, "tree": tree, "parents": [parent]})["sha"]
        self.req("PATCH", f"/git/refs/heads/{branch}", {"sha": commit, "force": False})

    def squash(self, branch: str):
        """Jedna świeża wersja bez historii (zwalnia miejsce po usuniętych galeriach)."""
        _, tree = self.head(branch)
        commit = self.req("POST", "/git/commits", {"message": "Galerie: porządki", "tree": tree, "parents": []})["sha"]
        self.req("PATCH", f"/git/refs/heads/{branch}", {"sha": commit, "force": True})

    def ensure_pages(self, branch: str):
        try:
            self.req("GET", "/pages")
            return
        except GitHubError as e:
            if "404" not in str(e):
                return
        try:
            self.req("POST", "/pages", {"source": {"branch": branch, "path": "/"}}, ok=(201, 409))
        except GitHubError as e:
            raise GitHubError("Nie udało się włączyć GitHub Pages automatycznie. Włącz ręcznie: repo → Settings → "
                              f"Pages → Branch: {branch} / (root) → Save.\n({e})")

    def pages_status(self) -> str:
        try:
            return self.req("GET", "/pages/builds/latest").get("status", "")
        except GitHubError:
            return ""


def _short(s: str) -> str:
    try:
        return json.loads(s).get("message", s)[:200]
    except ValueError:
        return s[:200]


# --------------------------------------------------------------------------- wysokopoziomowo
def publish(settings: dict, items, info: dict, wm: str, progress=None, cancelled=lambda: False) -> dict:
    settings["owner"], settings["repo"] = parse_repo(settings["owner"], settings["repo"])
    gh = GitHub(settings["owner"], settings["repo"], settings["token"])
    branch = gh.check()
    gid, key, files = build_files(items, info, wm, progress, cancelled)
    files["index.html"] = viewer_html()
    files["robots.txt"] = b"User-agent: *\nDisallow: /\n"
    gh.commit_files(branch, files, [], f"Galeria {gid}", progress, cancelled)
    gh.ensure_pages(branch)
    return {"id": gid, "key": key, "title": info.get("title", ""), "count": len(items),
            "created": int(time.time()), "max": info.get("max", 0),
            "names": [n for _, _, n in items]}


def delete(settings: dict, gid: str):
    gh = GitHub(settings["owner"], settings["repo"], settings["token"])
    branch = gh.check()
    gh.commit_files(branch, {"index.html": viewer_html()}, [f"g/{gid}/"], f"Usunięcie galerii {gid}")


def cleanup_history(settings: dict):
    gh = GitHub(settings["owner"], settings["repo"], settings["token"])
    gh.squash(gh.check())


def parse_selection(text: str, names: list[str]) -> tuple[list[str], list[str]]:
    """Z tekstu od klienta wyciąga nazwy zdjęć. Zwraca (znalezione, nieznane_tokeny)."""
    import re
    stems = {os.path.splitext(n)[0].lower(): n for n in names}
    found, unknown = [], []
    for tok in re.findall(r"[A-Za-z0-9_\-]{3,}", text):
        t = tok.lower()
        if t in stems:
            if stems[t] not in found:
                found.append(stems[t])
        elif any(ch.isdigit() for ch in t) and ("_" in t or "-" in t):
            unknown.append(tok)
    return found, unknown
