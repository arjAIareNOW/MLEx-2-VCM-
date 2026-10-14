"""Install check: this machine must reproduce the engine outputs recorded on the build cluster.

Uses the first 40 clips of the ai231 holdout split (downloaded once, 15 MB, SHA-256 checked): intent, slot and
confidence of every clip, and the Elkit / Marvin wake scores of the first 10.
    python tests/verify.py              -> "N/N reference outputs reproduced - OK"  (exit code 1 on any mismatch)
    python tests/verify.py --record     -> (re)write tests/reference_outputs.json  (build cluster only)
"""
import json, sys
from pathlib import Path

import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent))
from benchmark_pi import decode, holdout  # noqa: E402
from vcm_engine import CommandRecognizer, make_wake  # noqa: E402

REF = HERE / "reference_outputs.json"
rows = pq.read_table(holdout(HERE.parent / "data/holdout-00000-of-00001.parquet"), columns=["file", "audio"]).to_pylist()[:40]
cr, elkit, marvin = CommandRecognizer(), make_wake("elkit"), make_wake("marvin")
out = []
for i, r in enumerate(rows):
    x = decode(r["audio"]["bytes"]); p = cr.predict(x)
    o = {"file": r["file"], "intent": p["intent"], "confidence": round(p["confidence"], 6), "slot": p["slot"]}
    if i < 10: o.update(elkit=round(elkit.score_clip(x), 6), marvin=round(marvin.score_clip(x), 6))
    out.append(o)
if "--record" in sys.argv:
    REF.write_text(json.dumps(out, indent=1) + "\n"); print("recorded", len(out)); sys.exit(0)
ref, bad, n = json.loads(REF.read_text()), 0, 0
for o, e in zip(out, ref):
    for k in ("intent", "slot", "confidence", "elkit", "marvin"):
        if k not in e: continue
        n += 1
        same = abs(o[k] - e[k]) <= 1e-3 if isinstance(e[k], float) else o[k] == e[k]
        if not same: bad += 1; print("mismatch", e["file"], k, o[k], "expected", e[k])
print(f"{n - bad}/{n} reference outputs reproduced" + (" - OK" if not bad else " - MISMATCH"))
sys.exit(1 if bad else 0)
