"""Step 2d - balanced subset of the ai231 `numerals` clips for the slot model -> $VCM_DATA/numerals_subset.parquet.

The numerals set holds real-human number-only recordings (MLEnd spoken numerals, Speech Commands digits); none of its
speakers is in test or holdout. For each of the 10 numbers that are slot values (6, 8, 9, 10, 30, 1, 18, 20, 60, 100)
up to 700 clips are drawn round-robin over speakers (seed 0) and labelled with the slot class, e.g. "6" -> ALARM|6:00 AM.
Usage: python data_prep/numerals_subset.py
"""
import sys
from pathlib import Path

import numpy as np, pyarrow as pa, pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vcm.common import AI231, DATA_ROOT  # noqa: E402

NUM_SLOT = {'6': 'ALARM|6:00 AM', '8': 'ALARM|8:00 AM', '9': 'ALARM|9:00 PM', '10': 'TIMER|10 seconds',
            '30': 'TIMER|30 seconds', '1': 'TIMER|1 minute', '18': 'TEMPERATURE|18 degrees',
            '20': 'BRIGHTNESS|20 percent', '60': 'BRIGHTNESS|60 percent', '100': 'BRIGHTNESS|100 percent'}


def iter_rows(files, cols):
    for f in files:
        for batch in pq.ParquetFile(f).iter_batches(batch_size=512, columns=cols):
            yield from batch.to_pylist()


def main():
    files = sorted((AI231 / 'data').glob('numerals-*.parquet'))
    assert len(files) == 6, files
    rows = [r for r in iter_rows(files, ['file', 'numerals', 'speaker_id', 'source', 'transcript'])
            if (r['numerals'] or '').strip() in NUM_SLOT]
    rng = np.random.default_rng(0); keep = []
    for n in NUM_SLOT:  # <= 700 per number, round-robin over speakers for speaker variety
        by = {}
        for r in rows:
            if r['numerals'].strip() == n: by.setdefault(r['speaker_id'], []).append(r['file'])
        spk = sorted(by); rng.shuffle(spk); picked = []
        while len(picked) < 700 and any(by[s] for s in spk):
            for s in spk:
                if by[s] and len(picked) < 700: picked.append(by[s].pop(int(rng.integers(len(by[s])))))
        keep += picked
    keep = set(keep); tabs = []
    for f in files:
        t = pq.read_table(f, columns=['file', 'numerals', 'speaker_id', 'source', 'transcript', 'audio'])
        tabs.append(t.filter(pa.array([fn in keep for fn in t['file'].to_pylist()])))
    sub = pa.concat_tables(tabs)
    slot = [NUM_SLOT[n.strip()] for n in sub['numerals'].to_pylist()]
    sub = sub.append_column('slot_class', pa.array(slot, pa.string()))
    DATA_ROOT.mkdir(parents=True, exist_ok=True); pq.write_table(sub, DATA_ROOT / 'numerals_subset.parquet')
    print('numerals subset', len(sub), dict(zip(*np.unique(slot, return_counts=True))), flush=True)


if __name__ == '__main__':
    main()
