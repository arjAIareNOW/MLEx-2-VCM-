"""Real-human material for HELLO ELKIT (.venv): spliced positives + real negatives, from
  MSWC English ($VCM_DATA/mswc: 1 s word clips from Common Voice, CC-BY 4.0; speaker-disjoint train/dev/test)
  MLS Spanish "el" ($VCM_DATA/mls_es/el_clips, cut by cut_el_mls.py; 9_hours -> train, dev -> val/dev, test -> test).

Positive = real "hello" + real "el" (MLS) or "elk" (MSWC) + real "kit" (MSWC), each trimmed and loudness-matched, with
0-100 ms / 0-30 ms gaps. Speakers are mostly MIXED inside a phrase (only 10 MSWC speakers said both hello and kit), so
these are augmentation material, not natural utterances; on the eval side they are reported separately ("real_splice").
Negatives = every other real word clip (kind "realneg" in training: its own share, see kws_data.py) (hello alone, kit alone, yellow, kid, ... + random words) and spliced hard
negatives: hello + other word, hello + el + other word, other word + el + kit.

HI ELKIT: MSWC has no "hi" (<3 letters) -> homophone "high" (selected_extra.csv); positives/hard negatives are made per
greeting (column greeting = hello / hi), so the merged ELKIT sets get both.

Usage: python wakeword/build_real.py train $VCM_DATA/elkit/kws       (writes index_real.csv; split train/val)
       python wakeword/build_real.py eval  $VCM_DATA/elkit/eval (appends to index.csv; split dev/test)
"""
import csv, random, sys, zlib
from pathlib import Path
import numpy as np, soundfile as sf
from scipy.signal import resample_poly

MODE, D = sys.argv[1], Path(sys.argv[2])
DATA = Path(os.environ.get("VCM_DATA", Path(__file__).resolve().parents[1] / "data")).expanduser()
MSWC, MLS = DATA / "mswc", DATA / "mls_es/el_clips"
SR = 16000
rng = random.Random(23101 if MODE == "train" else 4242)
SPLITS = {"train": {"train": "train", "dev": "val"}, "eval": {"dev": "dev", "test": "test"}}[MODE]  # MSWC split -> ours
MLS_SPLIT = {"train": "train", "9_hours": "train", "dev": "val" if MODE == "train" else "dev", "test": "test"}
N_POS = {"train": 6000, "val": 600, "dev": 400, "test": 400}
N_HARD = {"train": 3000, "val": 300, "dev": 300, "test": 300}


def load(path):
    x, sr = sf.read(path, dtype="float32", always_2d=True); x = x.mean(1)
    if sr != SR: x = resample_poly(x, SR, sr).astype(np.float32)
    e = np.abs(x); idx = np.flatnonzero(e > e.max() * 10 ** (-35 / 20))
    return x[max(0, idx[0] - 320): idx[-1] + 320] if len(idx) else x


def level(x, db):
    return x * (10 ** (db / 20) / (np.sqrt(np.mean(x ** 2)) + 1e-8))


def splice(*parts, gaps):
    out = []
    for i, p in enumerate(parts):
        out.append(level(p, -20 + rng.uniform(-3, 3)))
        if i < len(gaps): out.append(np.zeros(int(SR * rng.uniform(*gaps[i])), np.float32))
    y = np.concatenate(out); return y / max(1.0, np.abs(y).max() / 0.95)


mswc = list({r["LINK"]: r for f in ("selected.csv", "selected_extra.csv") for r in csv.DictReader(open(MSWC / f))
             if r["SPLIT"] in SPLITS}.values())
GREET = {"hello": "hello", "hi": "high"}  # greeting -> MSWC word
el_rows = [r for r in csv.DictReader(open(MLS / "index.csv")) if MLS_SPLIT[r["split"]] in SPLITS.values()]
by = {}
for r in mswc: by.setdefault((SPLITS[r["SPLIT"]], r["WORD"]), []).append(r)
out_dir = D / "real"; out_dir.mkdir(parents=True, exist_ok=True)
rows, cache = [], {}


def clip(r):
    if r["LINK"] not in cache: cache[r["LINK"]] = load(MSWC / "clips" / r["LINK"])
    return cache[r["LINK"]]


def el_clip(split):
    if rng.random() < 0.75 and any(MLS_SPLIT[r["split"]] == split for r in el_rows):
        r = rng.choice([r for r in el_rows if MLS_SPLIT[r["split"]] == split]); return load(MLS / r["file"]), "mls" + r["speaker"]
    r = rng.choice(by[(split, "elk")]); return clip(r)[: int(len(clip(r)) * rng.uniform(0.75, 1.0))], r["SPEAKER"][:8]


def add(name, y, kind, text, spk, split, gender="unknown", source="mswc", greeting=""):
    sf.write(out_dir / name, y, SR, subtype="PCM_16")
    if MODE == "train": rows.append(dict(file=f"real/{name}", kind=kind, text=text, speaker=spk, split=split, group="real_human", greeting=greeting))
    else: rows.append(dict(file=f"real/{name}", label=kind, text=text, speaker=spk, split=split, group="real_human",
                           gender=gender.lower() or "unknown", snr="clean", source=source, greeting=greeting))


for split in SPLITS.values():
    kit = by.get((split, "kit"), []) + by.get((split, "kits"), [])[:len(by.get((split, "kit"), [])) // 3]
    others = [r for (s, w), rs in by.items() if s == split and w not in ("hello", "high", "kit", "kits", "elk") for r in rs]
    for g, word in GREET.items():
        hello, gp = by[(split, word)], "" if g == "hello" else "hi_"  # hello names unchanged from the hello-only version
        for i in range(N_POS[split]):
            h, k = rng.choice(hello), rng.choice(kit); e, es = el_clip(split)
            y = splice(clip(h), e, clip(k), gaps=[(0.0, 0.10), (0.0, 0.03)])
            add(f"{gp}pos_{split}_{i:05d}.wav", y, "pos", f"{g} el kit", f"splice_{h['SPEAKER'][:8]}", split, h["GENDER"], "real_splice", g)
        for i in range(N_HARD[split]):
            h, o = rng.choice(hello), rng.choice(others); e, _ = el_clip(split); c = rng.random()
            if c < 0.4: y, t = splice(clip(h), clip(o), gaps=[(0.0, 0.15)]), f"{word} {o['WORD']}"
            elif c < 0.7: y, t = splice(clip(h), e, clip(o), gaps=[(0.0, 0.10), (0.0, 0.03)]), f"{word} el {o['WORD']}"
            else: y, t = splice(clip(o), e, clip(rng.choice(kit)), gaps=[(0.0, 0.10), (0.0, 0.03)]), f"{o['WORD']} el kit"
            add(f"{gp}hardneg_{split}_{i:05d}.wav", y, "neg", t, f"splice_{h['SPEAKER'][:8]}", split, source="real_splice", greeting=g)
    for r in [r for (s, w), rs in by.items() if s == split for r in rs]:
        add(f"neg_{split}_{zlib.crc32(r['LINK'].encode()):08x}.wav", clip(r), "realneg" if MODE == "train" else "neg", r["WORD"],
            r["SPEAKER"][:8], split, r["GENDER"])
    cache.clear(); print(split, sum(r["split"] == split for r in rows), "rows", flush=True)

idx = D / "index.csv"; old = [r for r in csv.DictReader(open(idx)) if not r["file"].startswith("real/")]  # idempotent
if MODE == "train": idx = D / "index_real.csv"  # training: index.csv stays TTS-only, index_real.csv = TTS + real material (used for training)
keys = list(old[0].keys())
with open(idx, "w", newline="") as f:
    wr = csv.DictWriter(f, keys, extrasaction="ignore"); wr.writeheader(); wr.writerows(old + [{k: r.get(k, "") for k in keys} for r in rows])
print(len(old), "existing +", len(rows), "real rows ->", idx, flush=True)
