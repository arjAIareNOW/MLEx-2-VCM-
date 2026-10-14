"""On-the-fly training windows for the Elkit wake word (used by train_oww.py).

  positive  [lead]? + "Hello/Hi Elkit" + [command]? (random gaps), wake phrase fully inside a random 2 s window
  negative  TTS confusables (--neg-share 0.45) / leads / commands, real speech (ai231 train clips + real MSWC words),
            Speech Commands words, pure noise, and a command right after a cut-off wake phrase
Augment: speed 0.8-1.25 (pitch + vocal-tract shift, covers child-like voices), gain -10..+6 dB, and with p=0.8 noise at
0-20 dB SNR from the 13 DEMAND training rooms + Speech Commands background (the 4 evaluation rooms DLIVING, OMEETING,
PCAFETER, STRAFFIC are never used in training).
Note: Stream seeds its workers from the clock, so two training runs see different random windows.
"""
import csv, io, math, os, random, time
from pathlib import Path
import numpy as np, soundfile as sf, torch, torch.nn.functional as F
import pyarrow.parquet as pq

DATA_ROOT = Path(os.environ.get("VCM_DATA", Path(__file__).resolve().parents[1] / "data")).expanduser()

SR, WIN = 16000, 32000  # 2 s: "Hello Elkit" is ~0.9-1.5 s
EVAL_ROOMS = {"DLIVING", "OMEETING", "PCAFETER", "STRAFFIC"}


def load(path):
    x, sr = sf.read(path, dtype="float32", always_2d=True); x = x.mean(1)
    if sr != SR:
        x = F.interpolate(torch.from_numpy(x)[None, None], size=round(len(x) * SR / sr), mode="linear", align_corners=False)[0, 0].numpy()
    return trim(x)


def trim(x, db=-40):
    e = np.abs(x); th = e.max() * 10 ** (db / 20); idx = np.flatnonzero(e > th)
    return x[max(0, idx[0] - 800): idx[-1] + 800] if len(idx) else x


def speed(x, lo, hi):
    f = random.uniform(lo, hi)
    return F.interpolate(torch.from_numpy(x)[None, None], size=max(400, round(len(x) / f)), mode="linear", align_corners=False)[0, 0].numpy()


class Data:
    def __init__(self, kws_dir, split, n_real=3000, n_sc=6000, seed=0, neg_share=0.30, index="index.csv"):
        rng = random.Random(seed); self.neg_share = neg_share
        rows = [r for r in csv.DictReader(open(kws_dir / index)) if r["split"] == split]
        self.kind = {k: [load(kws_dir / r["file"]) for r in rows if r["kind"] == k] for k in ("pos", "lead", "cmd", "neg", "realneg")}
        real = []
        for p in sorted((DATA_ROOT / "ai231_filtered_lenient/data").glob("train-*.parquet")):
            real += pq.read_table(p, columns=["audio"]).to_pylist()
        rng.shuffle(real); cut = len(real) // 10
        real = real[cut:] if split == "train" else real[:cut]
        self.real = [trim(sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32", always_2d=True)[0].mean(1)) for r in real[:n_real]]
        sc = sorted(p for p in (DATA_ROOT / "speech_commands").glob("*/*.wav") if not p.parent.name.startswith("_"))
        rng.shuffle(sc); sc = sc[:n_sc] if split == "train" else sc[n_sc:n_sc + n_sc // 10]
        self.sc = [load(p) for p in sc]
        noise = [p for p in (DATA_ROOT / "noise/demand").glob("*_16k/*.wav") if p.parent.name[:-4] not in EVAL_ROOMS]
        noise += list((DATA_ROOT / "noise/speech_commands").glob("*.wav"))
        self.noise = [sf.read(p, dtype="float32", always_2d=True)[0].mean(1) for p in noise]
        print(split, {k: len(v) for k, v in self.kind.items()}, "real", len(self.real), "sc", len(self.sc), "noise", len(self.noise), flush=True)

    def window(self, x, keep=None):
        """random 1.5 s window; keep=(a, b) must lie fully inside."""
        if len(x) < WIN:
            pad = WIN - len(x); left = random.randint(0, pad); return np.pad(x, (left, pad - left))
        if keep is None: s = random.randint(0, len(x) - WIN)
        else:
            lo, hi = max(0, keep[1] - WIN), min(keep[0], len(x) - WIN)
            s = random.randint(lo, hi) if hi >= lo else max(0, min(keep[0], len(x) - WIN))
        return x[s:s + WIN]

    def positive(self, aug):
        w = random.choice(self.kind["pos"])
        if aug: w = speed(w, 0.8, 1.25)
        parts, a = [], 0
        if random.random() < 0.35:
            lead = random.choice(self.kind["lead"]); parts += [lead, np.zeros(random.randint(800, 4000), np.float32)]
        a = sum(len(p) for p in parts); parts.append(w); b = a + len(w)
        if random.random() < 0.35:
            parts += [np.zeros(random.randint(800, 4800), np.float32), random.choice(self.kind["cmd"])]
        return self.window(np.concatenate(parts), (a, b))

    def negative(self, aug):
        r = random.random(); q = (1 - self.neg_share) / 0.7  # other sources keep their v1 proportions (0.1/0.2/0.2/0.1/0.1)
        if r < self.neg_share: x = random.choice(self.kind["neg"])
        elif (r := (r - self.neg_share) / q) < 0.10: x = random.choice(self.kind["lead"] + self.kind["cmd"])
        elif r < 0.30: x = random.choice(self.kind["realneg"] if self.kind["realneg"] and random.random() < 0.5 else self.real)  # MSWC real words share the ai231 slot
        elif r < 0.50: x = random.choice(self.sc)
        elif r < 0.60:  # tail of a wake-word utterance: command after a cut-off HELLO ELKIT
            w = random.choice(self.kind["pos"]); c = random.choice(self.kind["cmd"])
            x = np.concatenate([w[int(len(w) * random.uniform(0.55, 0.8)):], np.zeros(random.randint(800, 4000), np.float32), c])
            x = x[:WIN]
        else: return self.noisy(np.zeros(WIN, np.float32), level=True)
        if aug: x = speed(x, 0.8, 1.25)
        return self.window(x)

    def noisy(self, x, level=False):
        n = random.choice(self.noise); s = random.randint(0, len(n) - WIN); n = n[s:s + WIN]
        if level: return n / (np.abs(n).max() + 1e-6) * 10 ** (random.uniform(-40, 0) / 20)
        ps, pn = np.mean(x ** 2) + 1e-8, np.mean(n ** 2) + 1e-8
        return x + n * np.sqrt(ps / (pn * 10 ** (random.uniform(0, 20) / 10)))

    def example(self, label, aug=True):
        x = self.positive(aug) if label else self.negative(aug)
        if aug:
            x = x * 10 ** (random.uniform(-10, 6) / 20)
            if random.random() < 0.8: x = self.noisy(x)
        return np.clip(x, -1, 1).astype(np.float32)


class Stream(torch.utils.data.IterableDataset):
    def __init__(self, data, steps, batch, ratio):
        self.d, self.steps, self.batch, self.ratio = data, steps, batch, ratio

    def __iter__(self):
        info = torch.utils.data.get_worker_info(); seed = (info.id if info else 0) + int(time.time() * 1000) % 100000
        random.seed(seed); np.random.seed(seed)
        n = self.steps // (info.num_workers if info else 1)
        for _ in range(n):
            y = (np.random.rand(self.batch) < 1 / (1 + self.ratio)).astype(np.float32)
            yield torch.from_numpy(np.stack([self.d.example(int(t)) for t in y])), torch.from_numpy(y)


def far_recall(p, y, cap=0.01):
    neg = np.sort(p[y == 0])[::-1]; k = int(math.floor(cap * len(neg)))
    th = neg[k] + 1e-6 if k < len(neg) else 0.0
    return float(np.mean(p[y == 1] >= th)), float(th)
