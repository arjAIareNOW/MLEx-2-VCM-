"""Real human Spanish "el" clips for HELLO ELKIT: Whisper large-v3-turbo word timestamps on Multilingual LibriSpeech
Spanish (9_hours / dev / test, 69 speakers, CC-BY 4.0), cut every confidently recognised "el" (.venv-whisper, GPU).

Keeps words whose text is exactly "el" (case/punctuation stripped), probability >= 0.5 and length 80-450 ms, with
30 ms context each side. Output: OUT/<split>_<speaker>_<n>.wav (16 kHz) + OUT/index.csv.
Usage: python wakeword/cut_el_mls.py $VCM_DATA/mls_es $VCM_DATA/mls_es/el_clips
"""
import csv, io, re, sys, time
from pathlib import Path
import numpy as np, pyarrow.parquet as pq, soundfile as sf
from faster_whisper import WhisperModel

SRC, OUT = Path(sys.argv[1]), Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
model = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")
rows, t0, n_utt = [], time.time(), 0
for split in ("9_hours", "dev", "test"):
    for batch in pq.ParquetFile(SRC / f"{split}.parquet").iter_batches(batch_size=64, columns=["audio", "speaker_id", "id"]):
        for r in batch.to_pylist():
            x, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32"); x = x.mean(1) if x.ndim > 1 else x
            assert sr == 16000
            segs, _ = model.transcribe(x, language="es", beam_size=5, condition_on_previous_text=False, word_timestamps=True)
            for w in (w for s in segs for w in (s.words or [])):
                if re.sub(r"[^a-záéíóúñ]", "", w.word.lower()) != "el" or w.probability < 0.5: continue
                if not 0.08 <= w.end - w.start <= 0.45: continue
                a, b = max(0, int((w.start - 0.03) * sr)), min(len(x), int((w.end + 0.03) * sr))
                name = f"{split}_{r['speaker_id']}_{len(rows):05d}.wav"
                sf.write(OUT / name, x[a:b], sr, subtype="PCM_16")
                rows.append(dict(file=name, split=split, speaker=r["speaker_id"], utt=r["id"], prob=round(w.probability, 3),
                                 dur=round(w.end - w.start, 3)))
            n_utt += 1
            if n_utt % 500 == 0: print(f"{n_utt} utterances, {len(rows)} el, {time.time() - t0:.0f}s", flush=True)
with open(OUT / "index.csv", "w", newline="") as f:
    wr = csv.DictWriter(f, rows[0].keys()); wr.writeheader(); wr.writerows(rows)
print(f"DONE {n_utt} utterances, {len(rows)} el clips, {len({r['speaker'] for r in rows})} speakers, {time.time() - t0:.0f}s", flush=True)
