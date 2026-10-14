"""Step 2c - build $VCM_DATA/ai231_v2 (the intent model's training set): the lenient-filtered ai231 train data
+ reviewed label fixes + Whisper-audited supplemental_synth + an oversampling list for LIGHT_ON / LIGHT_OFF.

Only reads the ai231 download, ai231_filtered_lenient and the frozen audit / review files; writes a NEW directory.
  data/train-*            ai231_filtered_lenient train with artifacts/review/label_fixes.csv applied (DROP / RELABEL rows)
  data/test, holdout      symlinks to the unchanged originals in the ai231 download
  supplemental_fil/       symlinks to ai231_filtered_lenient/supplemental_fil (unchanged)
  supplemental_synth/     train-voice clips only (voice_split == 'train'), kept by the same lenient rule as build_filtered_dataset.py
  synthetic_negatives/    symlinks to the ai231 download (train 1,000 / test 250 generated out-of-scope clips)
  oversample.csv          LIGHT_ON / LIGHT_OFF clips not worded exactly as a schema phrase ("lights off", "turn on lights", ...),
                          with a repeat factor for training; the parquet files themselves hold each clip once
  changes.csv, README.md
Usage: python data_prep/build_ai231_v2.py [OUT_DIR] [--repeat N]
"""
import collections, csv, os, sys
from pathlib import Path
import pyarrow as pa, pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_filtered as bfd
from vcm.common import AI231, ARTIFACTS, DATA_ROOT  # noqa: E402

ORIG, LEN, EXTRA = AI231, DATA_ROOT / 'ai231_filtered_lenient', AI231
FIXES = ARTIFACTS / 'review/label_fixes.csv'
AUDIT_EXTRA = ARTIFACTS / 'whisper/label_audit_supplemental_synth.csv.gz'
LIGHTS = ('LIGHT_ON', 'LIGHT_OFF')


def main():
    args = [x for x in sys.argv[1:] if not x.startswith('--')]
    repeat = int(sys.argv[sys.argv.index('--repeat') + 1]) if '--repeat' in sys.argv else 5
    args = [x for x in args if x != str(repeat)]
    out = Path(args[0]) if args else DATA_ROOT / 'ai231_v2'
    out.mkdir(parents=True, exist_ok=False)
    fixes = {r['file']: r for r in csv.DictReader(FIXES.open()) if r['action'] in ('DROP', 'RELABEL')}
    changes, oversample = [], []

    # 1. main train: lenient copy + label fixes
    (out / 'data').mkdir()
    applied = set()
    for f in sorted((LEN / 'data').glob('train-*.parquet')):
        t = pq.read_table(f); d = t.to_pydict(); keep = []
        for i, fn in enumerate(d['file']):
            fx = fixes.get(fn)
            if fx and fx['action'] == 'DROP':
                keep.append(False); applied.add(fn)
                changes.append(dict(file=fn, action='DROP', old_label=d['command'][i], new_label='', reason=fx['reason'])); continue
            if fx and fx['action'] == 'RELABEL':
                applied.add(fn)
                changes.append(dict(file=fn, action='RELABEL', old_label=d['command'][i], new_label=fx['new_label'], reason=fx['reason']))
                d['command'][i] = fx['new_label']; d['out_of_scope'][i] = 0; d['variation'][i] = ''
                d['variation_match'][i] = 'relabelled'; d['note'][i] = f"v2 relabel from {fx['label']}: {fx['reason']}"
            keep.append(True)
        t = pa.Table.from_pydict(d, schema=t.schema).filter(pa.array(keep))
        pq.write_table(t, out / 'data' / f.name)
        print(f'data/{f.name}: {t.num_rows}/{len(keep)} rows', flush=True)
    missing = set(fixes) - applied
    assert not missing, f'label fixes not found in lenient train: {sorted(missing)[:5]}'
    for name in ('test-00000-of-00001.parquet', 'holdout-00000-of-00001.parquet'):
        os.symlink(ORIG / 'data' / name, out / 'data' / name)
    os.symlink(ORIG / 'variations.csv', out / 'variations.csv')

    # 2. supplemental_fil: unchanged lenient copy
    (out / 'supplemental_fil').mkdir()
    for f in sorted((LEN / 'supplemental_fil').glob('*.parquet')):
        os.symlink(f, out / 'supplemental_fil' / f.name)

    # 3. supplemental_synth: train voices, same lenient keep rule as ai231_filtered_lenient
    bfd.LENIENT = True
    audit = {r['file']: r for r in csv.DictReader(bfd.open_csv(AUDIT_EXTRA))}
    src = EXTRA / 'supplemental_synth/train-00000-of-00001.parquet'
    t = pq.read_table(src); files, vs = t['file'].to_pylist(), t['voice_split'].to_pylist()
    keep, synth_report = [], []
    for fn, v in zip(files, vs):
        if v != 'train':
            keep.append(False); continue
        a = audit[fn]  # KeyError = clip missing from the audit -> stop, never guess
        k, why = bfd.decide(a); keep.append(k)
        synth_report.append(dict(file=fn, command=a['command'], kept=int(k), reason=why,
                                 transcript=a['transcript'], whisper_text=a['whisper_text']))
    t = t.filter(pa.array(keep))
    t = t.append_column('audit_whisper_text', pa.array([audit[fn]['whisper_text'] for fn in t['file'].to_pylist()], pa.string()))
    (out / 'supplemental_synth').mkdir()
    pq.write_table(t, out / 'supplemental_synth' / src.name)
    with (out / 'supplemental_synth/filter_report.csv').open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(synth_report[0])); w.writeheader(); w.writerows(synth_report)
    print(f'supplemental_synth: kept {t.num_rows}/{len(synth_report)} train-voice clips (of {len(files)})', flush=True)

    # 4. synthetic negatives: unchanged
    (out / 'synthetic_negatives').mkdir()
    for f in sorted((EXTRA / 'synthetic_negatives').glob('*.parquet')):
        os.symlink(f, out / 'synthetic_negatives' / f.name)

    # 5. oversample list: light clips worded differently from the schema phrases (the on/off minimal pairs)
    for sub in ('data', 'supplemental_fil', 'supplemental_synth'):
        for f in sorted((out / sub).glob('train-*.parquet')):
            r = pq.read_table(f, columns=['file', 'command', 'transcript', 'variation_match', 'source']).to_pylist()
            for x in r:
                if x['command'] in LIGHTS and x['variation_match'] != 'exact':
                    oversample.append(dict(subset=sub, file=x['file'], command=x['command'], source=x['source'],
                                           transcript=x['transcript'], repeat=repeat))
    for name, rows in (('changes.csv', changes), ('oversample.csv', oversample)):
        with (out / name).open('w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    c = collections.Counter((r['action'], r['old_label'], r['new_label']) for r in changes)
    o = collections.Counter(r['command'] for r in oversample)
    (out / 'README.md').write_text(
        '# ai231 v2 (lenient filter + label fixes + supplemental_synth)\n\n'
        'Built by data_prep/build_ai231_v2.py. The ai231 download and ai231_filtered_lenient are untouched.\n\n'
        '- data/train: ai231_filtered_lenient train with reviewed label fixes (changes.csv, from artifacts/review/label_fixes.csv).\n'
        '- data/test, data/holdout, variations.csv: symlinks to the unchanged originals in the ai231 download.\n'
        '- supplemental_fil: symlinks to ai231_filtered_lenient/supplemental_fil (unchanged).\n'
        '- supplemental_synth: ai231 supplemental_synth clips whose voice is a TRAIN voice, Whisper-audited '
        '(artifacts/whisper/label_audit_supplemental_synth.csv.gz) and kept by the lenient rule; per-clip decisions in supplemental_synth/filter_report.csv.\n'
        '- synthetic_negatives: symlinks to the downloaded ai231 synthetic_negatives (train/test).\n'
        f'- oversample.csv: {len(oversample)} LIGHT_ON/LIGHT_OFF clips not worded exactly as a schema phrase '
        f'({dict(o)}), repeat x{repeat} during training. Audio is stored once.\n\n'
        f'Label changes: {dict((" ".join(k).strip(), v) for k, v in c.items())}\n')
    print(f'changes: {dict(c)}\noversample: {dict(o)} x{repeat}\nwrote {out}')


if __name__ == '__main__':
    main()
