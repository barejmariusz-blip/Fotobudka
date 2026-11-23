"""Uruchamiane w GitHub Actions: pobiera model CLIP, liczy embeddingi opisów ras, zapisuje
fotobudka/data/labels.npz i fotobudka/_model.py (adres + suma kontrolna modelu)."""
import hashlib, os, sys, urllib.request
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from labels_pl import LABELS, TEMPLATES

BASE = "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/"
FILES = {"vision": "onnx/vision_model_quantized.onnx", "text": "onnx/text_model_quantized.onnx",
         "tok": "tokenizer.json"}
os.makedirs("ai_cache", exist_ok=True)
paths = {}
for k, f in FILES.items():
    p = os.path.join("ai_cache", os.path.basename(f))
    if not os.path.exists(p):
        print("download", f, flush=True)
        urllib.request.urlretrieve(BASE + f, p)
    paths[k] = p

import onnxruntime as ort
from tokenizers import Tokenizer

tok = Tokenizer.from_file(paths["tok"])
tok.no_padding()
tok.no_truncation()
ts = ort.InferenceSession(paths["text"], providers=["CPUExecutionProvider"])
print("text inputs", [(i.name, i.shape) for i in ts.get_inputs()], "outputs", [o.name for o in ts.get_outputs()])
vs = ort.InferenceSession(paths["vision"], providers=["CPUExecutionProvider"])
print("vision inputs", [(i.name, i.shape) for i in vs.get_inputs()], "outputs", [o.name for o in vs.get_outputs()])

def text_emb(s):
    ids = np.array([tok.encode(s).ids], dtype=np.int64)
    feeds = {}
    for i in ts.get_inputs():
        feeds[i.name] = ids if "ids" in i.name else np.ones_like(ids)
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

h = hashlib.sha256(open(paths["vision"], "rb").read()).hexdigest()
size = os.path.getsize(paths["vision"])
with open("fotobudka/_model.py", "w") as f:
    f.write(f'MODEL_URL = "{BASE + FILES["vision"]}"\nMODEL_SHA256 = "{h}"\nMODEL_SIZE = {size}\n')
print("model", size, h)
