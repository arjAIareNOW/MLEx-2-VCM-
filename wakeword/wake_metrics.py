"""Wake-word metrics from score_eval.py sequences: detection rate per clip kind, for rules "k windows in a row >= T".

Kinds: elkit_tts (generated "Hello/Hi Elkit"), elkit_real_splice (real speakers' recorded words joined), soundalike
(generated near-misses such as "Hi, el kid", allowed to wake), real_word (real MSWC single words), real_command (real
ai231 commands), real_sentence (real continuous speech: ai231 real people + MLS Spanish, if scored).
Thresholds are always chosen on DEV voices; TEST voices are only reported.
  default rule : <= 0.1 % of DEV unrelated real speech (real_word + real_command [+ real_sentence]) wakes it
Usage: python wakeword/wake_metrics.py SCORES.npz OUT.json [--k 1 2] [--extra 0.94:2 0.88:2]
"""
import argparse, csv, json, os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("VCM_DATA", REPO / "data")).expanduser()
KINDS = ("elkit_tts", "elkit_real_splice", "soundalike", "real_word", "real_command", "real_sentence")
UNRELATED = ("real_word", "real_command", "real_sentence")


def kind(r):
    if r["label"] == "pos": return "elkit_real_splice" if r["source"] == "real_splice" else "elkit_tts"
    return "real_word" if r["source"] == "mswc" else "real_command" if r["source"] == "ai231" else "soundalike"


def rows():
    out = [("elkit/" + r["file"], r["split"], kind(r)) for r in csv.DictReader(open(DATA / "elkit/eval/index.csv"))]
    sent = DATA / "elkit_sent/index.csv"
    if sent.is_file():
        out += [("sent/" + r["file"], r["split"], "real_sentence") for r in csv.DictReader(open(sent)) if r["split"] != "train"]
    return out


def clip_score(p, k):
    p = p.astype(np.float32)
    return float(np.lib.stride_tricks.sliding_window_view(p, k).min(1).max()) if p.size >= k else 0.0


def rates(score, rs, split, th):
    out = {}
    for kd in KINDS:
        v = np.array([score[key] for key, s, k in rs if s == split and k == kd])
        if v.size: out[kd] = [round(float(np.mean(v >= th)), 4), int((v >= th).sum()), int(v.size)]
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("scores"); ap.add_argument("out")
    ap.add_argument("--k", type=int, nargs="+", default=[1, 2]); ap.add_argument("--extra", nargs="*", default=[])
    a = ap.parse_args()
    with np.load(a.scores) as z: seqs = {k: z[k] for k in z.files}  # load once (npz access decompresses every time)
    rs = [r for r in rows() if r[0] in seqs]
    result = {"scores": a.scores, "rule": "threshold = <= 0.1 % of DEV unrelated real speech wakes it", "settings": {}}
    for k in a.k:
        score = {key: clip_score(seqs[key], k) for key, _, _ in rs}
        dev = np.sort([score[key] for key, s, kd in rs if s == "dev" and kd in UNRELATED])[::-1]
        th = float(dev[int(0.001 * len(dev))]) + 1e-6
        result["settings"][f"dev_0.1pct x {k}"] = {"threshold": round(th, 6), "consecutive": k,
                                                    "dev": rates(score, rs, "dev", th), "test": rates(score, rs, "test", th)}
    for e in a.extra:  # fixed settings, e.g. the one used in the field
        th, k = float(e.split(":")[0]), int(e.split(":")[1])
        score = {key: clip_score(seqs[key], k) for key, _, _ in rs}
        result["settings"][f"{th} x {k}"] = {"threshold": th, "consecutive": k,
                                             "dev": rates(score, rs, "dev", th), "test": rates(score, rs, "test", th)}
    Path(a.out).write_text(json.dumps(result, indent=1) + "\n")
    print(f"{'setting':18s}" + "".join(f"{k:>19s}" for k in KINDS))
    for name, m in result["settings"].items():
        print(f"{name:18s}" + "".join(f"{100 * m['test'][k][0]:9.1f}% ({m['test'][k][1]}/{m['test'][k][2]})".rjust(19) if k in m["test"] else " " * 19 for k in KINDS))


if __name__ == "__main__":
    main()
