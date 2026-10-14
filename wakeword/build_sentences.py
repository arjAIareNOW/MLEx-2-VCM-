"""Real continuous-speech negatives for the Elkit wake word -> $VCM_DATA/elkit_sent/{train,dev,test}/*.wav + index.csv.

  ai231 real-people clips (is_synthetic false: SLURP, FSC, SNIPS, real_voice, CommonVoice, ...):
      ai231 train split -> speakers hashed 85 % train / 15 % dev;  ai231 test split -> test (86 speakers, none in train)
      ai231 holdout is not used.
  MLS Spanish read speech (real continuous speech, other language):
      9_hours -> train (whole utterances);  dev -> dev, test -> test, cut into 4 s pieces (sentence-sized), 700 utterances each.
Inputs: $VCM_DATA/ai231/data (data_prep/download_data.py) and $VCM_DATA/mls_es (wakeword/download_wakeword_data.py mls).
Usage: python wakeword/build_sentences.py
"""
import csv, glob, hashlib, io, random
from pathlib import Path

import numpy as np, pyarrow.parquet as pq, soundfile as sf

import os
DATA = Path(os.environ.get("VCM_DATA", Path(__file__).resolve().parents[1] / "data")).expanduser()
OUT = DATA / "elkit_sent"
rows = []


def put(split, name, x, source, speaker):
    p = OUT / split / f"{name}.wav"; p.parent.mkdir(parents=True, exist_ok=True)
    sf.write(p, np.clip(x, -1, 1).astype(np.float32), 16000, subtype="PCM_16")
    rows.append(dict(file=f"{split}/{name}.wav", source=source, speaker=speaker, split=split, dur=round(len(x) / 16000, 3)))


def mono16k(b):
    x, sr = sf.read(io.BytesIO(b), dtype="float32", always_2d=True); x = x.mean(1)
    assert sr == 16000, sr
    return x


for s in ("train", "test"):
    for f in sorted(glob.glob(str(DATA / f"ai231/data/{s}-*.parquet"))):
        t = pq.read_table(f, columns=["audio", "source", "speaker_id", "is_synthetic"]).to_pylist()
        for i, r in enumerate(t):
            if r["is_synthetic"]: continue
            spk = str(r["speaker_id"])
            split = "test" if s == "test" else ("dev" if int(hashlib.md5(spk.encode()).hexdigest(), 16) % 100 < 15 else "train")
            put(split, f"ai231_{s}_{Path(f).stem}_{i:05d}", mono16k(r["audio"]["bytes"]), "ai231_" + r["source"], "ai231_" + spk)

rng = random.Random(0)
for part, split in (("9_hours", "train"), ("dev", "dev"), ("test", "test")):
    t = pq.read_table(DATA / f"mls_es/{part}.parquet", columns=["audio", "speaker_id", "id"]).to_pylist()
    if split != "train": rng.shuffle(t); t = t[:700]
    for r in t:
        x = mono16k(r["audio"]["bytes"])
        if split == "train": put(split, "mls_" + r["id"], x, "mls_es", "mls_" + r["speaker_id"])
        else:
            for k in range(0, len(x) - 32000 + 1, 64000):  # 4 s pieces, last piece >= 2 s
                put(split, f"mls_{r['id']}_{k // 64000}", x[k:k + 64000], "mls_es", "mls_" + r["speaker_id"])

assert len({r["file"] for r in rows}) == len(rows), "duplicate output filenames"
with open(OUT / "index.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, list(rows[0])); w.writeheader(); w.writerows(rows)
from collections import Counter
print(Counter((r["split"], r["source"].split("_")[0]) for r in rows))
print({s: len({r["speaker"] for r in rows if r["split"] == s}) for s in ("train", "dev", "test")}, "speakers")
