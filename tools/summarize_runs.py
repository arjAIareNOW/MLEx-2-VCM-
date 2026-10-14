"""Compare reproduced runs (RUNS/intent_conformer, RUNS/slot_conformer) with the released models in release/.

GPU training is not bit-exact (cuDNN autotuning, bf16), so a reproduction should land close to the released numbers,
not on them. Measured spread of the intent recipe: the same seed on an A6000 and on an A100 gave validation accuracy
97.99 % / 98.28 % and test accuracy 97.60 % / 98.01 %; another seed gave 97.77 % / 97.89 %.
Usage: python tools/summarize_runs.py RUNS            -> prints a table and writes RUNS/summary.md
"""
from __future__ import annotations

import json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def intent(d: Path) -> dict | None:
    f = d / "config.json"
    if not f.is_file(): return None
    c = json.loads(f.read_text())
    r = json.loads((d / "reject_threshold.json").read_text()) if (d / "reject_threshold.json").is_file() else {}
    return {"best epoch": c.get("best_epoch"), "val acc": c["validation_metrics"]["accuracy"], "val macro-F1": c["validation_metrics"]["macro_f1"],
            "test acc": c["test_metrics"]["accuracy"], "test macro-F1": c["test_metrics"]["macro_f1"],
            "holdout acc": c["holdout_metrics"]["accuracy"], "reject threshold": r.get("threshold")}


def slot(d: Path) -> dict | None:
    f = d / "config.json"
    if not f.is_file(): return None
    c = json.loads(f.read_text()); fin = c["final"]
    return {"best epoch": c.get("best_epoch"), "val acc": c["validation"]["accuracy"], "test acc": fin["test/clean"]["accuracy"],
            "test 5 dB noise": fin["test/demand_5db"]["accuracy"], "holdout acc": fin["holdout/clean"]["accuracy"]}


def table(name, ref, new):
    keys = list(ref)
    rows = [f"### {name}", "", "| | released | reproduced |", "|---|---:|---:|"]
    fmt = lambda v: "-" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))
    rows += [f"| {k} | {fmt(ref[k])} | {fmt(new.get(k)) if new else 'not run'} |" for k in keys]
    return "\n".join(rows) + "\n"


if __name__ == "__main__":
    runs = Path(sys.argv[1] if len(sys.argv) > 1 else REPO / "runs")
    md = "# Reproduction summary\n\n" + table("Intent Conformer", intent(REPO / "release/intent_conformer"), intent(runs / "intent_conformer")) + "\n" \
         + table("Slot Conformer", slot(REPO / "release/slot_conformer"), slot(runs / "slot_conformer"))
    print(md); runs.mkdir(parents=True, exist_ok=True); (runs / "summary.md").write_text(md)
