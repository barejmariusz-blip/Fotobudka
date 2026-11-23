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
    embs = []
    for tag, f, im in items:
        buf = io.BytesIO(); im.save(buf, "JPEG", quality=90)
        embs.append(ai.embed_photo(buf.getvalue(), 1))
    E = np.stack(embs)
    S = E @ E.T
    tags = np.array([t for t, _, _ in items])
    same = S[(tags[:, None] == tags[None]) & ~np.eye(len(tags), dtype=bool)]
    diff = S[tags[:, None] != tags[None]]
    print(f"::notice::Podobieństwo: ta sama rasa śr {same.mean():.3f} (min {same.min():.3f}), różne rasy śr {diff.mean():.3f} (max {diff.max():.3f})")
    for thr in (0.80, 0.83, 0.86, 0.89):
        lab = ai.cluster(embs, thr)
        groups = {}
        for (t, _, _), l in zip(items, lab):
            groups.setdefault(l, []).append(t)
        desc = "; ".join(f"G{l}: " + ",".join(f"{t}x{v.count(t)}" for t in sorted(set(v))) for l, v in sorted(groups.items()))
        print(f"::notice::próg {thr}: {len(groups)} grup -> {desc}")
    for tag in SETS:
        c = ai.centroid([e for e, (t, _, _) in zip(embs, items) if t == tag])
        print(f"::notice::{tag}: rozpoznano '{ai.name_for(c)[0]}' ({ai.name_for(c)[1]:.2f})")


try:
    main()
except Exception:
    print("::error::SELFTEST " + traceback.format_exc().replace("\n", "%0A"))
    raise
