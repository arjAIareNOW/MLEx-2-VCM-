"""Step 2a (optional, GPU) - transcribe every ai231 clip with Whisper large-v3-turbo (faster-whisper, fp16, beam 5).

Sets: train, supplemental_fil, test, holdout, and supplemental_synth clips of TRAIN voices (voice_split == 'train').
Read-only on the dataset. Writes one JSONL per shard; data_prep/analyze_whisper_labels.py turns them into label_audit.csv.
The result used for every model is frozen in artifacts/whisper/ (label_audit.csv.gz = the four main sets,
label_audit_supplemental_synth.csv.gz = supplemental_synth); re-running Whisper is only needed to check the audit.
Usage: python data_prep/whisper_audit.py OUT_DIR --shard K --num-shards N [--synth]
"""
import argparse, io, json, sys, time
from pathlib import Path
import numpy as np, pyarrow.parquet as pq, soundfile as sf
from faster_whisper import WhisperModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vcm.common import AI231 as H  # noqa: E402

SETS = [('train', sorted((H / 'data').glob('train-*.parquet'))),
        ('supplemental_fil', sorted((H / 'supplemental_fil').glob('train-*.parquet'))),
        ('test', sorted((H / 'data').glob('test-*.parquet'))),
        ('holdout', sorted((H / 'data').glob('holdout-*.parquet')))]
META = ['file', 'transcript', 'command', 'variation', 'slot_value', 'out_of_scope', 'speaker_id', 'source',
        'is_synthetic', 'accent_group', 'duration_s', 'transcript_source', 'whisper_check']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('out', type=Path); p.add_argument('--shard', type=int, required=True)
    p.add_argument('--num-shards', type=int, required=True); p.add_argument('--synth', action='store_true')
    a = p.parse_args()
    if a.synth:  # supplemental_synth, train voices only (output split name 'supplemental_synth')
        SETS[:] = [('supplemental_synth', sorted((H / 'supplemental_synth').glob('train-*.parquet')))]
        META.append('voice_split'); a.out.mkdir(parents=True, exist_ok=True)
    out = a.out / f'shard_{a.shard:02d}_of_{a.num_shards:02d}.jsonl'
    done = set()
    if out.exists():
        done = {(r['split'], r['file']) for r in map(json.loads, out.open())}
    model = WhisperModel('large-v3-turbo', device='cuda', compute_type='float16')
    g, t0, n = -1, time.time(), 0
    with out.open('a') as fh:
        for split, files in SETS:
            for f in files:
                pf = pq.ParquetFile(f)
                for batch in pf.iter_batches(batch_size=256, columns=META + ['audio']):
                    for r in batch.to_pylist():
                        g += 1
                        if a.synth and r.get('voice_split') != 'train':
                            continue
                        if g % a.num_shards != a.shard or (split, r['file']) in done:
                            continue
                        wav, sr = sf.read(io.BytesIO(r.pop('audio')['bytes']), dtype='float32')
                        if wav.ndim > 1: wav = wav.mean(1)
                        assert sr == 16000, (r['file'], sr)
                        segs, _ = model.transcribe(wav, language='en', beam_size=5, condition_on_previous_text=False,
                                                   without_timestamps=True, vad_filter=False)
                        segs = list(segs)
                        r.update(split=split, parquet=f.name, whisper_text=' '.join(s.text.strip() for s in segs).strip(),
                                 avg_logprob=min((s.avg_logprob for s in segs), default=None),
                                 no_speech_prob=max((s.no_speech_prob for s in segs), default=None),
                                 peak=float(np.abs(wav).max()) if wav.size else 0.0)
                        fh.write(json.dumps(r) + '\n'); n += 1
                        if n % 500 == 0:
                            fh.flush(); print(f'{n} clips, {time.time()-t0:.0f}s', flush=True)
    print(f'DONE shard {a.shard}: {n} new clips in {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
