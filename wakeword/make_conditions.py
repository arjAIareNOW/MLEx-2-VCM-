"""Step 2 (.venv-aug): 16 kHz resample + simulated age groups + held-out room noise.

Each TTS clip -> 3 voice groups:
  adult    unchanged (male / female from the TTS speaker; gender estimated from median F0 < / >= 165 Hz)
  child    WORLD: F0 raised to ~260-320 Hz, formants x1.15-1.25 (shorter vocal tract), slightly slower
  elderly  WORLD: slower (x0.75-0.9), 4-7 Hz pitch tremor, flatter intonation, formants x0.95-1.0, -6 dB
Each (clip, group) gets one noise level in rotation: clean, 20, 10, 5 dB SNR, using DEMAND rooms that the
slot/intent models never trained on (DLIVING, OMEETING, PCAFETER, STRAFFIC).
Also adds 400 real human ai231 test clips (no wake word) as negatives, clean.
"""
import csv, io, os, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np, soundfile as sf, pyarrow.parquet as pq
from scipy.signal import resample_poly
sys.path.insert(0, str(Path(__file__).resolve().parent))
from world import analyze, transform, synth, FS

SRCS, OUT = [Path(p) for p in sys.argv[1].split(",")], Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
DATA = Path(os.environ.get("VCM_DATA", Path(__file__).resolve().parents[1] / "data")).expanduser()
DEMAND = DATA / "noise/demand"
NOISES = [sf.read(next((DEMAND / f"{r}_16k").glob("*.wav")))[0] for r in ("DLIVING", "OMEETING", "PCAFETER", "STRAFFIC")]
LEVELS = [None, 20, 10, 5]


def add_noise(x, snr, k, rng):
    if snr is None: return x
    n = NOISES[k % len(NOISES)]; s = rng.integers(0, len(n) - len(x))
    seg = n[s:s + len(x)]; px, pn = np.mean(x ** 2) + 1e-10, np.mean(seg ** 2) + 1e-10
    return x + seg * np.sqrt(px / (pn * 10 ** (snr / 10)))


def write(name, y):
    y = y / max(1.0, np.abs(y).max() / 0.99); sf.write(OUT / name, y, FS, subtype="PCM_16")


def work(job):
    idx, src, r = job
    rng = np.random.default_rng(23101 + idx)
    x, sr = sf.read(Path(src) / r["file"]); x = resample_poly(x, 320, 441) if sr == 22050 else x
    x = np.concatenate([np.zeros(4000), x, np.zeros(4000)])  # 0.25 s lead/tail, as a VAD would cut
    f0, sp, ap = analyze(x); fm = float(np.median(f0[f0 > 0])) if (f0 > 0).any() else 150.0
    gender = "female" if fm >= 165 else "male"
    groups = {"adult": x}
    st = float(np.clip(12 * np.log2(rng.uniform(260, 320) / fm), 2, 14))
    groups["child"] = synth(x, *transform(f0, sp, ap, st=st, alpha=rng.uniform(1.15, 1.25), r=rng.uniform(0.85, 1.0)))
    groups["elderly"] = 0.5 * synth(x, *transform(f0, sp, ap, st=(-1.5 if gender == "female" else 0.5),
                                                  r=rng.uniform(0.75, 0.9), a=0.7, wob=rng.uniform(0.03, 0.06),
                                                  wob_hz=rng.uniform(4, 7), alpha=rng.uniform(0.95, 1.0)))
    out = []
    for gi, (g, y) in enumerate(groups.items()):
        k = 3 * idx + gi; snr = LEVELS[k % 4]
        name = f"{Path(r['file']).stem}_{g}_{'clean' if snr is None else f'{snr}db'}.wav"
        write(name, add_noise(y, snr, k, rng))
        out.append(dict(file=name, label=r["label"], text=r["text"], speaker=r["speaker"], split=r["split"],
                        group=g, gender=gender if g != "child" else "child", snr="clean" if snr is None else snr,
                        source="filipino_mms" if r["speaker"].startswith("mms_") else r["speaker"].rstrip("0123456789")))
    return out


jobs = [(i, str(src), r) for i, (src, r) in enumerate((src, r) for src in SRCS for r in csv.DictReader(open(src / "tts_index.csv")))]
with Pool(int(os.environ.get("WORKERS", "8"))) as pool:
    rows = [row for out in pool.imap(work, jobs, chunksize=8) for row in out]
t = pq.read_table(DATA / "ai231/data/test-00000-of-00001.parquet",
                  columns=["audio", "transcript", "speaker_id", "source", "out_of_scope"]).to_pylist()
real = [r for r in t if r["source"] == "real_voice"] if os.environ.get("GREETING") != "hi" else []  # hi add-on: hello set has them
for i, r in enumerate(real[:: max(1, len(real) // 400)][:400]):
    name = f"neg_real_{i:03d}.wav"; (OUT / name).write_bytes(r["audio"]["bytes"])
    rows.append(dict(file=name, label="neg", text=r["transcript"], speaker=f"ai231_{r['speaker_id']}",
                     split="dev" if i % 2 == 0 else "test", group="real_human", gender="unknown", snr="clean", source="ai231"))
with open(OUT / "index.csv", "w", newline="") as f:
    wr = csv.DictWriter(f, rows[0].keys()); wr.writeheader(); wr.writerows(rows)
print(len(rows), "clips")
