"""Elkit wake-word head = wakeword/train_oww.py recipe + real continuous-speech negatives ($VCM_DATA/elkit_sent). Released head: job 65684.

Why: 65357 saw mostly single words / TTS commands as negatives, so ordinary sentences score high and the deployed rule
had to go up to 0.94 x 2 (62.7 % real-splice recall). Changes, everything else identical:
  - negatives: with prob --sent-share a random 2 s window of a real sentence (ai231 real people / MLS Spanish;
    train speakers only; val uses the dev speakers)
  - positives: with prob --ctx-p a real sentence piece is placed before and/or after "Hello/Hi Elkit"
Build the sentences first (wakeword/build_sentences.py).
Usage: python wakeword/train_oww_sent.py --index index_real.csv --kws-data $VCM_DATA/elkit/kws --eval-clips $VCM_DATA/elkit/eval \
           --out runs/elkit --tag elkit --sent-share 0.3 --ctx-p 0.25   (other options as train_oww.py)
"""
import csv, random, sys
from pathlib import Path

import numpy as np, soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_oww as base
K = base.K
SENT = K.DATA_ROOT / "elkit_sent"
OPT = {"sent_share": 0.3, "ctx_p": 0.25}


class SentData(K.Data):
    def __init__(self, kws_dir, split, *args, **kw):
        super().__init__(kws_dir, split, *args, **kw)
        want = "train" if split == "train" else "dev"
        rows = [r for r in csv.DictReader(open(SENT / "index.csv")) if r["split"] == want]
        self.sent = [K.trim(sf.read(SENT / r["file"], dtype="float32", always_2d=True)[0].mean(1)) for r in rows]
        self.sent = [x for x in self.sent if len(x) > 4000]
        print(split, "sentences", len(self.sent), f"({sum(map(len, self.sent)) / 16000 / 3600:.1f} h)", flush=True)

    def piece(self, lo, hi):
        x = random.choice(self.sent); n = min(len(x), int(16000 * random.uniform(lo, hi))); s = random.randint(0, len(x) - n)
        return x[s:s + n]

    def negative(self, aug):
        if random.random() < OPT["sent_share"]:
            x = random.choice(self.sent)
            if aug: x = K.speed(x, 0.85, 1.15)
            return self.window(x)
        return super().negative(aug)

    def positive(self, aug):
        if random.random() >= OPT["ctx_p"]: return super().positive(aug)
        w = random.choice(self.kind["pos"])
        if aug: w = K.speed(w, 0.8, 1.25)
        gap = lambda: np.zeros(random.randint(0, 4000), np.float32)
        before = [self.piece(0.3, 1.5), gap()] if random.random() < 0.7 else []
        after = [gap(), self.piece(0.3, 1.5)] if random.random() < 0.7 else []
        a = sum(len(p) for p in before); b = a + len(w)
        return self.window(np.concatenate(before + [w] + after), (a, b))


if __name__ == "__main__":
    for k in list(OPT):
        flag = "--" + k.replace("_", "-")
        if flag in sys.argv:
            i = sys.argv.index(flag); OPT[k] = float(sys.argv[i + 1]); del sys.argv[i:i + 2]
    print("sentence options", OPT, flush=True)
    K.Data = SentData
    base.main()
