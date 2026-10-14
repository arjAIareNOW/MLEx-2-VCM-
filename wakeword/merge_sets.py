"""Merge the HELLO ELKIT and HI ELKIT sets into one 'ELKIT' set (either greeting = wake): a directory of symlinks to
both sources + one concatenated index (same columns; extra 'greeting' column = hello / hi).
Usage: python merge_sets.py OUT INDEX_NAME SRC_HELLO SRC_HI      (INDEX_NAME = index.csv for clips and for kws dirs)
"""
import csv, sys
from pathlib import Path

OUT, NAME, SRCS = Path(sys.argv[1]), sys.argv[2], [Path(p) for p in sys.argv[3:5]]
OUT.mkdir(parents=True, exist_ok=True)
rows = []
for greet, src in zip(("hello", "hi"), SRCS):
    for r in csv.DictReader(open(src / NAME)):
        link = OUT / r["file"]
        if not link.exists(): link.symlink_to(src / r["file"])
        rows.append({**r, "greeting": greet})
assert len({r["file"] for r in rows}) == len(rows), "file name collision between sets"
with open(OUT / NAME, "w", newline="") as f:
    wr = csv.DictWriter(f, rows[0].keys()); wr.writeheader(); wr.writerows(rows)
lab = "label" if "label" in rows[0] else "kind"
print(OUT, len(rows), "rows", {g: {k: sum(r["greeting"] == g and r[lab] == k for r in rows) for k in sorted({r[lab] for r in rows})} for g in ("hello", "hi")})
