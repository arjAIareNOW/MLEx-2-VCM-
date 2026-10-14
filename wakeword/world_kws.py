"""KWS step 1b (.venv-aug): WORLD child / elderly variants of the KWS TRAINING clips (same recipe as make_conditions.py,
different random draws: seeded from crc32(file) + 9_000_000, eval uses 23101 + clip index).

pos  -> child + elderly copy each;  neg / lead / cmd -> one of child / elderly (alternating), so age-shifted voices
appear on both sides of the decision. 16 kHz, 0.25 s silence kept off (the trainer trims anyway).
Reads OUT/index_tts.csv, writes OUT/index.csv = original rows (group=adult) + variants.
"""
import csv, os, sys, zlib
from multiprocessing import Pool
from pathlib import Path
import numpy as np, soundfile as sf
from scipy.signal import resample_poly
sys.path.insert(0, str(Path(__file__).resolve().parent))
from world import analyze, transform, synth, FS

D = Path(sys.argv[1])


def work(r):
    rng = np.random.default_rng(9_000_000 + zlib.crc32(r["file"].encode()))
    x, sr = sf.read(D / r["file"]); x = x.mean(1) if x.ndim > 1 else x
    x = resample_poly(x, 320, 441) if sr == 22050 else x
    f0, sp, ap = analyze(x); voiced = f0[f0 > 0]
    if len(voiced) < 5: return []
    fm = float(np.median(voiced)); female = fm >= 165
    groups = ("child", "elderly") if r["kind"] == "pos" else (("child", "elderly")[zlib.crc32(r["file"].encode()) % 2],)
    out = []
    for g in groups:
        if g == "child":
            st = float(np.clip(12 * np.log2(rng.uniform(260, 320) / fm), 2, 14))
            y = synth(x, *transform(f0, sp, ap, st=st, alpha=rng.uniform(1.15, 1.25), r=rng.uniform(0.85, 1.0)))
        else:
            y = 0.5 * synth(x, *transform(f0, sp, ap, st=(-1.5 if female else 0.5), r=rng.uniform(0.75, 0.9), a=0.7,
                                          wob=rng.uniform(0.03, 0.06), wob_hz=rng.uniform(4, 7), alpha=rng.uniform(0.95, 1.0)))
        if not np.isfinite(y).all(): continue
        name = f"{Path(r['file']).stem}_{g}.wav"
        sf.write(D / name, y / max(1.0, np.abs(y).max() / 0.99), FS, subtype="PCM_16")
        out.append({**r, "file": name, "group": g})
    return out


if __name__ == "__main__":
    rows = [{**r, "group": "adult"} for r in csv.DictReader(open(D / "index_tts.csv"))]
    with Pool(int(os.environ.get("WORKERS", "8"))) as p:
        extra = [o for out in p.imap(work, rows, chunksize=16) for o in out]
    with open(D / "index.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, rows[0].keys()); wr.writeheader(); wr.writerows(rows + extra)
    print(len(rows), "tts clips +", len(extra), "WORLD variants", flush=True)
