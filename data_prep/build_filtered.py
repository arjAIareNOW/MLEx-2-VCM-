"""Step 2b - build $VCM_DATA/ai231_filtered_lenient: the ai231 training data kept by the Whisper label audit.

The audit (data_prep/whisper_audit.py + data_prep/analyze_whisper_labels.py) is frozen in
artifacts/whisper/label_audit.csv.gz, so this step needs no GPU.

A training clip is KEPT when
  - right transcription: Whisper agrees with the stored transcript (status OK, similarity >= 80), and
  - right words: for slotted commands the slot value (number / colour / task) is heard, with no opposite am/pm.
    Paraphrased open-source wording ("Make lights red" = COLOR) is kept: its label was assigned by the dataset and
    Whisper confirms the audio says that transcript. Column `exact` in filter_report.csv marks Option B wording;
  - slotted commands must carry one of the 3 schema slot values (e.g. "alarm for 7" is dropped);
  - out-of-scope clips are kept unless Whisper hears a clean in-scope command (OOS_LOOKS_LIKE_COMMAND).
--lenient (used for every model here): also keep real-human CLOSE_ACCENT_OR_WORDING clips whose command words and slot are right.
test/ and holdout/ are linked unchanged. The ai231 download is only read.

Usage: python data_prep/build_filtered.py [--audit artifacts/whisper/label_audit.csv.gz] [--out $VCM_DATA/ai231_filtered_lenient] [--strict]
"""
import argparse, collections, csv, gzip, os, sys
from pathlib import Path
import pyarrow as pa, pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vcm.common import AI231 as H, ARTIFACTS, DATA_ROOT  # noqa: E402
SLOTS = collections.defaultdict(set)
for _r in csv.DictReader((H / 'variations.csv').open()):
    if _r['value']: SLOTS[_r['label']].add(_r['value'])


LENIENT = True  # --strict turns this off


def decide(r):
    if r['command'] == 'OUT_OF_SCOPE':
        return r['status'] == 'OOS_OK', r['status']
    if r['command'] in SLOTS and r['slot_value'] not in SLOTS[r['command']]:
        return False, 'SLOT_NOT_IN_SCHEMA'  # slots must be one of the 3 schema values
    if r['status'] != 'OK':
        lenient_ok = (LENIENT and r['status'] == 'CLOSE_ACCENT_OR_WORDING' and str(r['is_synthetic']) == '0'
                      and (r['best_phrase_command'] == r['command'] or float(r['own_score'] or 0) >= 75)
                      and r['slot_heard'] != '0')
        if not lenient_ok:
            return False, r['status']
        return True, 'KEPT_LENIENT_ACCENT'
    if r['slot_heard'] == '0':
        return False, 'SLOT_NOT_HEARD'
    return True, 'KEPT'


def open_csv(path):
    return gzip.open(path, 'rt', newline='') if str(path).endswith('.gz') else open(path, newline='')


def main():
    global LENIENT
    p = argparse.ArgumentParser()
    p.add_argument('--audit', type=Path, default=ARTIFACTS / 'whisper/label_audit.csv.gz')
    p.add_argument('--out', type=Path, default=DATA_ROOT / 'ai231_filtered_lenient'); p.add_argument('--strict', action='store_true')
    a = p.parse_args(); LENIENT = not a.strict; out = a.out
    audit = {(r['split'], r['file']): r for r in csv.DictReader(open_csv(a.audit))}
    out.mkdir(parents=True, exist_ok=False)
    report = []
    for split, sub in [('train', 'data'), ('supplemental_fil', 'supplemental_fil')]:
        (out / sub).mkdir(exist_ok=True)
        for f in sorted((H / sub).glob('train-*.parquet')):
            t = pq.read_table(f)
            files = t['file'].to_pylist(); vmatch = t['variation_match'].to_pylist()
            keep, reasons = [], []
            for i, fn in enumerate(files):
                a = audit[(split, fn)]  # KeyError = clip missing from the audit -> stop, never guess
                k, why = decide(a)
                keep.append(k); reasons.append(why)
                report.append(dict(split=split, command=a['command'], source=a['source'], is_synthetic=a['is_synthetic'],
                                   exact=vmatch[i] == 'exact', kept=int(k), reason=why,
                                   file=fn, transcript=a['transcript'], whisper_text=a['whisper_text']))
            t = t.filter(pa.array(keep)).append_column('audit_whisper_text', pa.array(
                [audit[(split, fn)]['whisper_text'] for fn, k in zip(files, keep) if k], pa.string()))
            pq.write_table(t, out / sub / f.name)
            print(f'{sub}/{f.name}: kept {sum(keep)}/{len(keep)}', flush=True)
    for name in ['test-00000-of-00001.parquet', 'holdout-00000-of-00001.parquet']:
        os.symlink(H / 'data' / name, out / 'data' / name)
    os.symlink(H / 'variations.csv', out / 'variations.csv')
    with (out / 'filter_report.csv').open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(report[0])); w.writeheader(); w.writerows(report)
    (out / 'README.md').write_text(
        '# ai231 filtered (Whisper label audit)\n\nFiltered copy of the ai231 TRAIN data (data/train + supplemental_fil).\n'
        'Kept = Whisper large-v3-turbo agrees with the stored transcript (>=80) AND the words map to the labelled command '
        '(+ slot value heard for slotted commands). Slotted clips must carry one of the 3 schema slot values. Out-of-scope kept unless Whisper hears a clean command.\n'
        'test/holdout/variations.csv are symlinks to the unchanged originals in the ai231 download.\n'
        'Per-clip decisions: filter_report.csv. Built by data_prep/build_filtered.py.\n')
    summary(report)


def summary(report):
    for split in ['train', 'supplemental_fil']:
        rs = [r for r in report if r['split'] == split]
        reasons = sorted({r['reason'] for r in rs if not r['kept']})
        print(f'\n===== {split}: kept {sum(r["kept"] for r in rs)}/{len(rs)}')
        print(f'{"command":16s} {"before":>7s} {"kept":>6s} {"%kept":>6s} {"human":>11s} {"paraphr":>8s} ' + ' '.join(f'{x[:13]:>13s}' for x in reasons))
        by = collections.defaultdict(list)
        for r in rs: by[r['command']].append(r)
        for c in sorted(by):
            g = by[c]; k = sum(r['kept'] for r in g); hu = [r for r in g if str(r['is_synthetic']) == '0']
            cnt = collections.Counter(r['reason'] for r in g)
            print(f'{c:16s} {len(g):7d} {k:6d} {100*k/len(g):5.1f}% {sum(r["kept"] for r in hu):5d}/{len(hu):<5d} {sum(r["kept"] and not r.get("exact", True) for r in g):8d} '
                  + ' '.join(f'{cnt[x]:13d}' for x in reasons))


if __name__ == '__main__':
    main()
