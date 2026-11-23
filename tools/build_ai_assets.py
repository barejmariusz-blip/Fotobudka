"""GitHub Actions: przygotowuje modele AI i pliki dla programu.

* modele (YOLOv8n, DINOv2-small, CLIP-B/32 vision) trzymane w wydaniu `models-v2` tego repo
  (jeśli wydanie już jest - pobierane stamtąd, inaczej budowane i wysyłane),
* fotobudka/_model.py - adresy, rozmiary, sumy SHA-256,
* fotobudka/data/labels.npz - embeddingi opisów ras (CLIP text).
"""
import hashlib, json, os, subprocess, sys, urllib.request
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from labels_pl import LABELS, TEMPLATES

REPO = os.environ.get("GITHUB_REPOSITORY", "barejmariusz-blip/Fotobudka")
TAG = "models-v2"
CACHE = "ai_cache"
os.makedirs(CACHE, exist_ok=True)
FILES = {"yolo": "yolov8n.onnx", "dino": "dinov2-small.onnx", "clip": "clip-b32-vision-q.onnx"}
HF = "https://huggingface.co/"


def sh(*a, check=True):
    print("+", " ".join(a), flush=True)
    return subprocess.run(a, check=check, capture_output=True, text=True)


def fetch(url, dst):
    print("download", url, flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Fotobudka-build"})
    with urllib.request.urlopen(req, timeout=300) as r, open(dst, "wb") as f:
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b)


have_release = sh("gh", "release", "view", TAG, "--repo", REPO, check=False).returncode == 0
if have_release:
    sh("gh", "release", "download", TAG, "--repo", REPO, "--dir", CACHE, "--clobber")
missing = [f for f in FILES.values() if not os.path.exists(os.path.join(CACHE, f))]
if missing:
    # CLIP vision
    p = os.path.join(CACHE, FILES["clip"])
    if not os.path.exists(p):
        fetch(HF + "Xenova/clip-vit-base-patch32/resolve/main/onnx/vision_model_quantized.onnx", p)
    # DINOv2-small
    p = os.path.join(CACHE, FILES["dino"])
    if not os.path.exists(p):
        for repo in ("Xenova/dinov2-small", "onnx-community/dinov2-small"):
            try:
                fetch(HF + repo + "/resolve/main/onnx/model.onnx", p)
                break
            except Exception as e:
                print("fail", repo, e)
    # YOLOv8n -> ONNX
    p = os.path.join(CACHE, FILES["yolo"])
    if not os.path.exists(p):
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch", "torchvision",
                        "--index-url", "https://download.pytorch.org/whl/cpu"], check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics", "onnx"], check=True)
        from ultralytics import YOLO
        out = YOLO("yolov8n.pt").export(format="onnx", imgsz=640, opset=12, simplify=False)
        os.replace(out, p)
    files = [os.path.join(CACHE, f) for f in FILES.values()]
    if have_release:
        sh("gh", "release", "upload", TAG, *files, "--repo", REPO, "--clobber")
    else:
        sh("gh", "release", "create", TAG, *files, "--repo", REPO, "--title", "Modele AI (nie usuwać)",
           "--notes", "Modele AI pobierane przez Fotobudkę przy pierwszym użyciu rozpoznawania zwierząt.",
           "--latest=false", "--prerelease")

models = {}
for k, f in FILES.items():
    p = os.path.join(CACHE, f)
    models[k] = {"file": f, "url": f"https://github.com/{REPO}/releases/download/{TAG}/{f}",
                 "size": os.path.getsize(p), "sha256": hashlib.sha256(open(p, "rb").read()).hexdigest()}
with open("fotobudka/_model.py", "w") as fh:
    fh.write("MODELS = " + json.dumps(models, indent=1) + "\n")
print(json.dumps(models, indent=1))

# embeddingi tekstu (CLIP) dla nazw ras
import onnxruntime as ort
from tokenizers import Tokenizer
tp = os.path.join(CACHE, "text_model_quantized.onnx")
tk = os.path.join(CACHE, "tokenizer.json")
if not os.path.exists(tp):
    fetch(HF + "Xenova/clip-vit-base-patch32/resolve/main/onnx/text_model_quantized.onnx", tp)
if not os.path.exists(tk):
    fetch(HF + "Xenova/clip-vit-base-patch32/resolve/main/tokenizer.json", tk)
tok = Tokenizer.from_file(tk)
tok.no_padding()
tok.no_truncation()
ts = ort.InferenceSession(tp, providers=["CPUExecutionProvider"])


def text_emb(s):
    ids = np.array([tok.encode(s).ids], dtype=np.int64)
    feeds = {i.name: (ids if "ids" in i.name else np.ones_like(ids)) for i in ts.get_inputs()}
    outs = [o.name for o in ts.get_outputs()]
    v = ts.run(["text_embeds" if "text_embeds" in outs else outs[0]], feeds)[0].reshape(-1)
    return v / np.linalg.norm(v)


names, embs = [], []
for pl, en in LABELS:
    e = np.mean([text_emb(t.format(en)) for t in TEMPLATES], axis=0)
    embs.append(e / np.linalg.norm(e))
    names.append(pl)
os.makedirs("fotobudka/data", exist_ok=True)
np.savez("fotobudka/data/labels.npz", names=np.array(names), emb=np.stack(embs).astype(np.float16))
for k, f in FILES.items():
    s = ort.InferenceSession(os.path.join(CACHE, f), providers=["CPUExecutionProvider"])
    print(k, "inputs", [(i.name, i.shape) for i in s.get_inputs()], "outputs", [(o.name, o.shape) for o in s.get_outputs()])
