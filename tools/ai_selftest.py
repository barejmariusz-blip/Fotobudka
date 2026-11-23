"""GitHub Actions: test rozpoznawania na zdjęciach z Wikipedii. Wyniki jako ::notice::."""
import io, json, os, shutil, sys, time, traceback, urllib.error, urllib.parse, urllib.request
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

sys.path.insert(0, os.getcwd())
from fotobudka import ai

UA = {"User-Agent": "FotobudkaSelfTest/2.0 (github actions; github.com/barejmariusz-blip)"}
SETS = {"Dalmatian": ("Dalmatian dog", "dalmatyńczyk"), "Husky": ("Siberian Husky", "husky syberyjski"),
        "BritishSH": ("British Shorthair", "kot brytyjski"), "Pug": ("Pug", "mops"),
        "Golden": ("Golden Retriever", "golden retriever"), "Corgi": ("Pembroke Welsh Corgi", "corgi"),
        "Maine": ("Maine Coon", "maine coon"), "Border": ("Border Collie", "border collie")}


def api(params):
    url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode({**params, "format": "json"})
    for attempt in range(5):
        try:
            return json.load(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30))
        except urllib.error.HTTPError as e:
            if e.code != 429:
                raise
            time.sleep(10 * (attempt + 1))
    raise RuntimeError("429")


def images(title, n):
    q = api({"action": "query", "titles": title, "generator": "images", "gimlimit": 40,
             "prop": "imageinfo", "iiprop": "url", "iiurlwidth": 1000})
    urls = [ii["thumburl"] for p in q.get("query", {}).get("pages", {}).values()
            if p["title"].lower().endswith((".jpg", ".jpeg")) for ii in p.get("imageinfo", []) if ii.get("thumburl")]
    out = []
    for u in urls:
        if len(out) >= n:
            break
        try:
            time.sleep(1)
            b = urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=30).read()
            out.append(Image.open(io.BytesIO(b)).convert("RGB"))
        except Exception as e:
            print("skip", u, e)
    return out


def augment(im, rng):
    """Symulacja innego kadru tego samego zwierzęcia: kadrowanie, obrót, światło, odbicie."""
    w, h = im.size
    s = rng.uniform(0.7, 0.92)
    cw, ch = int(w * s), int(h * s)
    x, y = rng.integers(0, w - cw + 1), rng.integers(0, h - ch + 1)
    im = im.crop((x, y, x + cw, y + ch)).rotate(rng.uniform(-8, 8), expand=False)
    im = ImageEnhance.Brightness(im).enhance(rng.uniform(0.8, 1.2))
    im = ImageEnhance.Color(im).enhance(rng.uniform(0.8, 1.2))
    if rng.random() < 0.5:
        im = ImageOps.mirror(im)
    return im


def main():
    os.makedirs(ai.data_dir(), exist_ok=True)
    for k, m in ai.MODELS.items():
        shutil.copy(os.path.join("ai_cache", m["file"]), ai.model_path(k))
    rng = np.random.default_rng(0)
    items = []   # (rasa, id zdjęcia, obraz)
    for tag, (title, _) in SETS.items():
        for j, im in enumerate(images(title, 5)):
            items.append((tag, f"{tag}{j}", im))
            for a in range(2):
                items.append((tag, f"{tag}{j}", augment(im, rng)))
    print(len(items), "obrazów", flush=True)
    t0 = time.time()
    res = [ai.analyze_image(im) for _, _, im in items]
    dt = (time.time() - t0) / max(1, len(items))
    det = sum(1 for r in res if r["a"] is not None)
    print(f"::notice::Wykryto zwierzę na {det}/{len(items)} obrazach; osobę na {sum(1 for r in res if r['p'] is not None)}; "
          f"{dt * 1000:.0f} ms/zdjęcie")
    ok = [i for i, r in enumerate(res) if r["a"] is not None]
    E = np.stack([res[i]["a"] for i in ok])
    tags = np.array([items[i][0] for i in ok])
    ids = np.array([items[i][1] for i in ok])
    S = E @ E.T
    off = ~np.eye(len(ok), dtype=bool)
    same_id = S[(ids[:, None] == ids[None]) & off]
    same_breed = S[(tags[:, None] == tags[None]) & (ids[:, None] != ids[None])]
    diff = S[tags[:, None] != tags[None]]
    q = lambda a, p: float(np.percentile(a, p)) if len(a) else float("nan")  # noqa: E731
    print(f"::notice::DINO podobieństwo: ten sam osobnik śr {same_id.mean():.3f} p10 {q(same_id, 10):.3f} | "
          f"ta sama rasa inny osobnik śr {same_breed.mean():.3f} p90 {q(same_breed, 90):.3f} | "
          f"inna rasa śr {diff.mean():.3f} p95 {q(diff, 95):.3f} p99 {q(diff, 99):.3f}")
    series = list(range(len(items)))
    for thr in (0.45, 0.50, 0.55, 0.60, 0.65, 0.70):
        lab, _ = ai.cluster(res, series, thr)
        L = np.array(lab)
        # czystość: ile zdjęć w grupie z dominującym osobnikiem / rasą
        groups = [np.where(L == c)[0] for c in set(lab) if c >= 0]
        pur_b = sum(max(np.unique([items[i][0] for i in g], return_counts=True)[1]) for g in groups) / max(1, (L >= 0).sum())
        # rozbicie: na ile grup średnio rozpada się jeden osobnik
        split = np.mean([len(set(L[[i for i in range(len(items)) if items[i][1] == idv and L[i] >= 0]]))
                         for idv in set(ids)])
        print(f"::notice::próg {thr}: {len(groups)} grup (osobników {len(set(ids))}, ras {len(SETS)}); "
              f"czystość rasy {pur_b:.2f}; osobnik w {split:.2f} grupach")
    exp = {k: v[1] for k, v in SETS.items()}
    hit = 0
    wrong = []
    for i in ok:
        nm, _ = ai.name_for([res[i]["c"]])
        if nm == exp[items[i][0]]:
            hit += 1
        else:
            wrong.append(f"{items[i][0]}->{nm}")
    print(f"::notice::Rasa (CLIP na wycinku zwierzęcia): trafiona {hit}/{len(ok)}. Błędy: " + ", ".join(wrong[:25]))
    gh = {}
    for i in ok:
        gh.setdefault(items[i][0], []).append(res[i]["c"])
    print("::notice::Rasa dla grupy: " + ", ".join(f"{t}->{ai.name_for(v)[0]}" for t, v in gh.items()))


try:
    main()
except Exception:
    print("::error::SELFTEST " + traceback.format_exc().replace("\n", "%0A"))
    raise
