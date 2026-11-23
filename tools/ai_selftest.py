"""Test w GitHub Actions: zdjęcia z Wikipedii (kilka ras), grupowanie i nazwy ras.
Wyniki jako adnotacje ::notice:: (widoczne w podsumowaniu builda)."""
import io, json, os, shutil, sys, urllib.error, urllib.parse, urllib.request
import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, os.getcwd())
from fotobudka import ai, analysis

import traceback

def main():
    os.makedirs(os.path.dirname(ai.model_path()), exist_ok=True)
    shutil.copy("ai_cache/vision_model_quantized.onnx", ai.model_path())
    UA = {"User-Agent": "FotobudkaSelfTest/1.0 (github actions; contact via github barejmariusz-blip)"}

    def api(params):
        url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode({**params, "format": "json"})
        return json.load(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30))

    def images(title, n):
        import time
        for attempt in range(4):
            try:
                q = api({"action": "query", "titles": title, "generator": "images", "gimlimit": 40,
                         "prop": "imageinfo", "iiprop": "url", "iiurlwidth": 900})
                break
            except urllib.error.HTTPError as e:
                if e.code != 429:
                    raise
                time.sleep(10 * (attempt + 1))
        pages = q.get("query", {}).get("pages", {}).values()
        urls = [ii["thumburl"] for p in pages if p["title"].lower().endswith((".jpg", ".jpeg"))
                for ii in p.get("imageinfo", []) if ii.get("thumburl")]
        out = []
        for u in urls[:n]:
            try:
                time.sleep(1)
                b = urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=30).read()
                out.append((u.rsplit("/", 1)[-1], Image.open(io.BytesIO(b)).convert("RGB")))
            except Exception as e:
                print("skip", u, e)
        return out

    SETS = {"Dalmatian": "Dalmatian dog", "Husky": "Siberian Husky", "BritishSH": "British Shorthair",
            "Pug": "Pug", "Golden": "Golden Retriever"}
    items = []
    for tag, title in SETS.items():
        for f, im in images(title, 4):
            items.append((tag, f, im))
            # ta sama scena po przycięciu / odbiciu (symulacja kolejnych kadrów tego samego zwierzęcia)
            w, h = im.size
            items.append((tag, f + " [crop]", ImageOps.mirror(im.crop((int(w * .08), int(h * .05), int(w * .95), int(h * .97))))))
    print(len(items), "zdjęć")
    datas = []
    for tag, f, im in items:
        buf = io.BytesIO(); im.save(buf, "JPEG", quality=90); datas.append(buf.getvalue())
    names, T = ai.labels()
    for mode in ("crop", "full", "mix"):
        E = np.stack([ai.embed_photo(d, 1, mode) for d in datas])
        top = [names[int(np.argmax(T @ e))] for e in E]
        exp = {"Dalmatian": "dalmatyńczyk", "Husky": "husky syberyjski", "BritishSH": "kot brytyjski",
               "Pug": "mops", "Golden": "golden retriever"}
        ok = sum(1 for (t, _, _), n in zip(items, top) if exp[t] == n)
        tags = np.array([t for t, _, _ in items])
        S = E @ E.T
        pair = [float(S[i, i + 1]) for i in range(0, len(items) - 1, 2)]
        same = S[(tags[:, None] == tags[None]) & ~np.eye(len(tags), dtype=bool)]
        diff = S[tags[:, None] != tags[None]]
        print(f"::notice::{mode}: rasa trafiona {ok}/{len(items)}; para kadrów śr {np.mean(pair):.3f} min {np.min(pair):.3f}; "
              f"ta sama rasa śr {same.mean():.3f}; różne rasy śr {diff.mean():.3f} p95 {np.percentile(diff, 95):.3f}")
        wrong = [f"{t}->{n}" for (t, _, _), n in zip(items, top) if exp[t] != n]
        print(f"::notice::{mode} błędy: " + ", ".join(wrong[:20]))


try:
    main()
except Exception:
    print("::error::SELFTEST " + traceback.format_exc().replace("\n", "%0A"))
    raise
