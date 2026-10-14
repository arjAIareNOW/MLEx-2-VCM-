"""Step 4: pick the match threshold on DEV voices (false accepts <= 1 % of dev negatives, then max recall),
report on TEST voices only: recall by age/gender group, accent source and noise; false accepts on confusables
and on real human ai231 commands; Whisper latency on this CPU (2 threads)."""
import csv, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from elkoris_match import score

index = {r["file"]: r for r in csv.DictReader(open(sys.argv[1]))}
RES = Path(sys.argv[2]); report = {}
# transcripts_*.csv: Whisper text -> fuzzy ELKORIS match score; scores_*.csv: any detector's own score (higher = ELKORIS)
for path in sorted(RES.glob("transcripts_*.csv")) + sorted(RES.glob("scores_*.csv")):
    tag = path.stem.replace("transcripts_", "").replace("scores_", "")
    rows = []
    for t in csv.DictReader(open(path)):
        r = dict(index[t["file"]]); r["ms"] = float(t["ms"])
        if "text" in t: r["heard"] = t["text"]; r["score"] = score(t["text"])[0]
        else: r["heard"] = ""; r["score"] = float(t["score"])
        rows.append(r)
    REAL = {"mswc", "real_splice"}  # real-human extras: reported separately, never used for the threshold
    real_test = [r for r in rows if r["split"] == "test" and r["source"] in REAL]
    rows = [r for r in rows if r["source"] not in REAL]
    dev = [r for r in rows if r["split"] == "dev"]; test = [r for r in rows if r["split"] == "test"]
    dneg = np.array([r["score"] for r in dev if r["label"] == "neg"]); dpos = np.array([r["score"] for r in dev if r["label"] == "pos"])
    cands = [(float(th), float(np.mean(dpos >= th)), float(np.mean(dneg >= th))) for th in (np.round(np.arange(0.60, 1.001, 0.01), 2) if path.stem.startswith("transcripts_") else np.unique(np.concatenate([dneg, dpos])))]
    ok = [c for c in cands if c[2] <= 0.01]
    # if no threshold reaches FA <= 1 % (hotword prompts make Whisper write "Elkoris" for look-alikes), take the lowest FA
    best = max(ok, key=lambda c: c[1]) if ok else min(cands, key=lambda c: (c[2], -c[1]))
    th = best[0]
    def rate(rs, lab):
        rs = [r for r in rs if r["label"] == lab]
        return (round(float(np.mean([r["score"] >= th for r in rs])), 4), len(rs)) if rs else None
    pos = [r for r in test if r["label"] == "pos"]; neg = [r for r in test if r["label"] == "neg"]
    by = lambda key: {k: rate([r for r in test if r[key] == k], "pos") for k in sorted({r[key] for r in pos})}
    fa_by_text = defaultdict(list)
    for r in neg:
        if r["source"] != "ai231": fa_by_text[r["text"]].append(r["score"] >= th)
    report[tag] = {
        "threshold_from_dev": round(th, 4), "dev_recall": round(best[1], 4), "dev_false_accept_rate": round(best[2], 4), "dev_fa_target_met": bool(ok),
        "test_recall": rate(test, "pos"), "test_false_accept_rate": rate(test, "neg"),
        "test_false_accept_real_ai231": rate([r for r in neg if r["source"] == "ai231"], "neg"),
        "test_false_accept_tts_confusables": rate([r for r in neg if r["source"] != "ai231"], "neg"),
        "real_splice_recall": rate([r for r in real_test if r["source"] == "real_splice"], "pos"),
        "real_splice_hard_neg_FA": rate([r for r in real_test if r["source"] == "real_splice"], "neg"),
        "real_mswc_word_FA": rate([r for r in real_test if r["source"] == "mswc"], "neg"),
        "real_mswc_false_wakes_per_hour_of_words": (round(3600 * sum(r["score"] >= th for r in real_test if r["source"] == "mswc")
                                                          / max(1, sum(r["source"] == "mswc" for r in real_test)), 2)),
        "recall_by_group": by("group"), "recall_by_gender": by("gender"), "recall_by_source": by("source"),
        "recall_by_snr": by("snr"),
        "recall_by_greeting": by("greeting") if "greeting" in test[0] else None,
        "real_splice_recall_by_greeting": ({g: rate([r for r in real_test if r["source"] == "real_splice" and r["greeting"] == g], "pos")
                                            for g in ("hello", "hi")} if real_test and "greeting" in real_test[0] else None),
        "worst_confusables": sorted(((round(float(np.mean(v)), 3), k) for k, v in fa_by_text.items()), reverse=True)[:6],
        "latency_ms_p50_p95": [round(float(np.percentile([r["ms"] for r in rows], q)), 1) for q in (50, 95)],
        "missed_examples": [(r["text"], r["group"], r["source"], r["snr"], r["heard"]) for r in pos if r["score"] < th][:15],
        "false_accept_examples": [(r["text"], r["group"], r["heard"]) for r in neg if r["score"] >= th][:10],
    }
(RES / "report.json").write_text(json.dumps(report, indent=1))
for tag, m in report.items():
    print(f"{tag:18s} th={m['threshold_from_dev']} recall={m['test_recall']} FA={m['test_false_accept_rate']} lat={m['latency_ms_p50_p95']}")
