"""Out-of-scope reject threshold for the intent model, chosen on VALIDATION speakers only.

Rule (fixed in advance): threshold = 5th percentile of the max-softmax confidence of in-scope validation clips, i.e.
95 % of validation commands are accepted; anything below it is reported as "not a known command". Validation OOS clips
= ai231_v2 main-train OUT_OF_SCOPE rows of validation speakers (never used to train the 19-way head). The frozen
threshold is then applied to test and holdout (in-scope accept rate / accuracy, OOS false-accept rate).

Usage: python -m vcm.reject_threshold RUN_DIR      -> RUN_DIR/reject_threshold.json
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np
import pyarrow as pa
import torch
from torch.utils.data import DataLoader

from .common import AI231, AI231_V2, CLASSES
from .data import SUBSETS, IntentDataset, read
from .features import collate
from .models import load_intent

ACCEPT_TARGET = 0.95


@torch.inference_mode()
def scores(model, table, indices):
    mapping = {c: i for i, c in enumerate(CLASSES)}; mapping["OUT_OF_SCOPE"] = -1
    conf, pred, label = [], [], []
    for x, lengths, y in DataLoader(IntentDataset(table, indices, mapping), batch_size=64, collate_fn=collate):
        p = model(x, lengths).softmax(1)
        conf += p.max(1).values.tolist(); pred += p.argmax(1).tolist(); label += y.tolist()
    return np.array(conf), np.array(pred), np.array(label)


def report(conf, pred, label, t):
    ins, oos = label >= 0, label < 0; accepted = conf >= t
    return {"in_scope_clips": int(ins.sum()), "oos_clips": int(oos.sum()),
            "in_scope_accept_rate": float(accepted[ins].mean()) if ins.any() else None,
            "in_scope_accuracy_all": float((pred[ins] == label[ins]).mean()) if ins.any() else None,
            "in_scope_correct_and_accepted": float(((pred == label) & accepted)[ins].mean()) if ins.any() else None,
            "oos_false_accept_rate": float(accepted[oos].mean()) if oos.any() else None}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("run", type=Path)
    ap.add_argument("--data", type=Path, default=AI231_V2); ap.add_argument("--ai231", type=Path, default=AI231)
    a = ap.parse_args()
    out = a.run / "reject_threshold.json"
    if out.exists(): raise FileExistsError(out)
    torch.set_num_threads(8)
    model = load_intent(torch.load(a.run / "best.pt", map_location="cpu", weights_only=True))
    manifest = json.loads((a.run / "split_manifest.json").read_text())
    val_speakers = set(manifest["validation_speaker_ids"])
    parts = [read(sorted((a.data / sub).glob("train-*.parquet")), True) for sub, _ in SUBSETS]
    table = pa.concat_tables([p.cast(parts[0].schema) for p in parts])  # same row order as vcm.data.prepare_intent
    commands, speakers = table["command"].to_pylist(), table["speaker_id"].to_pylist()
    val_oos = [i for i in range(len(parts[0])) if commands[i] == "OUT_OF_SCOPE" and speakers[i] in val_speakers]
    conf, pred, label = scores(model, table, manifest["validation_indices"] + val_oos)
    threshold = float(np.quantile(conf[label >= 0], 1 - ACCEPT_TARGET))
    result = {"rule": f"5th percentile of in-scope validation max-softmax (accept {ACCEPT_TARGET:.0%} of validation commands)",
              "threshold": threshold, "validation": report(conf, pred, label, threshold)}
    for split in ("test", "holdout"):
        tab = read(sorted((a.ai231 / "data").glob(f"{split}-*.parquet")), True)
        idx = [i for i, c in enumerate(tab["command"].to_pylist()) if c in CLASSES or c == "OUT_OF_SCOPE"]
        result[split] = report(*scores(model, tab, idx), threshold)
    out.write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
