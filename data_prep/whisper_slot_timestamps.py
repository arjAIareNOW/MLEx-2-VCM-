"""Step 2e (optional, GPU) - word-level Whisper large-v3-turbo timestamps of the 6 slotted commands (train data).

The slot model uses them to start some training crops just before the slot word ("... for [30] seconds").
Frozen result: artifacts/whisper/slot_word_timestamps.jsonl.gz (made from the STRICT Whisper filter,
python data_prep/build_filtered.py --strict --out $VCM_DATA/ai231_filtered).
Usage: python data_prep/whisper_slot_timestamps.py OUT_DIR --data $VCM_DATA/ai231_filtered
"""
import argparse, io, json, sys, time
from pathlib import Path
import pyarrow.parquet as pq, soundfile as sf
from faster_whisper import WhisperModel

SLOTTED = {'TIMER', 'ALARM', 'TEMPERATURE', 'BRIGHTNESS', 'COLOR', 'CREATE_REMINDER'}
META = ['file', 'transcript', 'command', 'slot_value', 'speaker_id', 'source', 'is_synthetic', 'duration_s']


def main():
    p = argparse.ArgumentParser(); p.add_argument('out', type=Path)
    p.add_argument('--data', type=Path, required=True)
    a = p.parse_args(); out = a.out; out.mkdir(parents=True, exist_ok=True)
    wanted = SLOTTED
    sets = [('train', sorted((a.data / 'data').glob('train-*.parquet'))),
            ('supplemental_fil', sorted((a.data / 'supplemental_fil').glob('train-*.parquet')))]
    model = WhisperModel('large-v3-turbo', device='cuda', compute_type='float16')
    t0, n = time.time(), 0
    with (out / 'slot_word_timestamps.jsonl').open('w') as fh:
        for split, files in sets:
            for f in files:
                for batch in pq.ParquetFile(f).iter_batches(batch_size=256, columns=META + ['audio']):
                    for r in batch.to_pylist():
                        if r['command'] not in wanted: continue
                        wav, sr = sf.read(io.BytesIO(r.pop('audio')['bytes']), dtype='float32')
                        if wav.ndim > 1: wav = wav.mean(1)
                        segs, _ = model.transcribe(wav, language='en', beam_size=5, condition_on_previous_text=False,
                                                   word_timestamps=True, vad_filter=False)
                        words = [dict(w=x.word.strip(), s=round(x.start, 3), e=round(x.end, 3), p=round(x.probability, 3))
                                 for s in segs for x in (s.words or [])]
                        r.update(split=split, parquet=f.name, words=words)
                        fh.write(json.dumps(r) + '\n'); n += 1
                        if n % 250 == 0: fh.flush(); print(f'{n} clips, {time.time()-t0:.0f}s', flush=True)
    print(f'DONE {n} clips in {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
