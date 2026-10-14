"""Score the wake-word evaluation sets with the DEPLOYED engine's streaming rule (deployment/vcm_engine.WakeWord:
2 s window every 80 ms, 1 s of silence before each clip), so thresholds and test numbers describe exactly what runs on
the device. Per clip it stores the whole sequence of window probabilities, so any rule "N windows in a row >= T" can be
evaluated later (wakeword/wake_metrics.py).

Sets: the Elkit eval clips ($VCM_DATA/elkit/eval: TTS positives / sound-alikes, real-speaker splices, real MSWC words,
real ai231 commands; dev + test voices) and, if present, real continuous sentences ($VCM_DATA/elkit_sent, dev + test).
Usage: python wakeword/score_eval.py HEAD.onnx OUT.npz
"""
import csv, os, sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO / "deployment"))
DATA = Path(os.environ.get("VCM_DATA", REPO / "data")).expanduser()
HEAD, _w = None, None


def init(head):
    global HEAD
    HEAD = head


def seq(item):
    global _w
    key, path = item
    from vcm_engine import WakeWord, read_wav
    from vcm_engine.models import _session
    if _w is None:
        _w = WakeWord(threads=1); _w.head = _session(Path(HEAD), 1)
    x = np.asarray(read_wav(path), np.float32).reshape(-1)  # same padding as WakeWord.score_clip
    x = np.concatenate([np.zeros(16000, np.float32), x, np.zeros(max(8000, 24000 - x.size), np.float32)])
    _w.cache = {}
    return key, np.array([_w._window(x[s:s + _w.WIN], s // _w.STRIDE) for s in range(0, x.size - _w.WIN + 1, _w.STRIDE)], np.float16)


if __name__ == "__main__":
    head, out = sys.argv[1], Path(sys.argv[2])
    ev = DATA / "elkit/eval"
    items = [("elkit/" + r["file"], ev / r["file"]) for r in csv.DictReader(open(ev / "index.csv"))]
    sent = DATA / "elkit_sent"
    if (sent / "index.csv").is_file():
        items += [("sent/" + r["file"], sent / r["file"]) for r in csv.DictReader(open(sent / "index.csv")) if r["split"] != "train"]
    res = {}
    with Pool(int(os.environ.get("SLURM_CPUS_PER_TASK", 8)), initializer=init, initargs=(head,)) as p:
        for i, (k, v) in enumerate(p.imap_unordered(seq, items, chunksize=32)):
            res[k] = v
            if i % 5000 == 0: print(i, "/", len(items), flush=True)
    out.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(out, **res)
    print("done", len(res), flush=True)
