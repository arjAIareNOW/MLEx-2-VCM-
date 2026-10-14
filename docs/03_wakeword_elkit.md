# 3. Wake word "Hello Elkit" / "Hi Elkit"

One detector hears both phrases. It runs every 80 ms on the last 2 s of audio and is small enough for a Pi 5 core.

```
16 kHz audio ─► openWakeWord mel (frozen) ─► 16 speech embeddings × 96 (frozen) ─► Elkit head (trained) ─► P(Elkit)
                last 2 s, every 80 ms        one per 80 ms                         job 65684
wake = P ≥ 0.886 for 2 windows in a row, then 1 s cooldown
```

| | |
|---|---|
| Front end | openWakeWord v0.5.1 `melspectrogram.onnx` + `embedding_model.onnx` (Apache-2.0), unchanged |
| Head | `deployment/models/hello_elkit_oww_head.onnx`: [N,16,96] embeddings → logit (training: `release/elkit_wakeword/`) |
| Setting | threshold **0.886**, **2** windows in a row (`models/config.json`); `0.94 × 2` = fewer false wakes |
| Backup | "Marvin" (Speech Commands model, `--wake marvin`), used only if Elkit is not wanted |

The mel model clips quiet frames relative to the loudest frame of its input, so the mel is always computed on the whole
2 s window; embeddings whose input frames did not change are reused, so only 1 of the 16 is new per step.

## Data

Nobody recorded "Elkit" for ai231, so the positives are generated or stitched, and real speech is used for everything
the detector must NOT wake on. All sets are speaker-disjoint: evaluation voices are split into **dev** (thresholds) and
**test** (reported numbers), and no evaluation voice is used in training.

| Kind | Training (`$VCM_DATA/elkit/kws`) | Evaluation (`$VCM_DATA/elkit/eval`, dev / test) |
|---|---|---|
| "Hello/Hi Elkit", generated | other Piper speakers than eval (incl. accented and non-English voices reading English) + other Meta MMS Philippine-language voices (`gen_kws_tts.py`); WORLD child / elderly copies | 85 Piper voices + 8 MMS Philippine voices (`gen_tts.py`, `gen_mms.py`), adult / child / elderly, clean / 20 / 10 / 5 dB: **2,832 / 2,832** |
| "Hello/Hi Elkit", stitched from real people | real MSWC "hello" / "high" + MLS Spanish "el" / MSWC "elk" + MSWC "kit" (`build_real.py`) | same recipe, other speakers: **800 / 800** |
| Sound-alikes ("Hi, el kid", "yelo kit", "Hello Eric") | auto-filtered so no eval phrase is in training | **3,834 / 3,834** |
| Real single words | MSWC English words, Speech Commands | **6,262 / 6,194** MSWC words |
| Real commands | ai231 real-voice training clips | **90 / 89** ai231 clips |
| **Real sentences** | **10.3 h**: ai231 real-voice clips + MLS Spanish read speech (`build_sentences.py`) | **2,918 / 3,491** |
| Noise | 13 DEMAND rooms + Speech Commands background, 0-20 dB | 4 DEMAND rooms never used in training |

Training windows are built on the fly (`kws_data.py`): the wake phrase anywhere in a random 2 s window, sometimes with
"hey / okay" before it or a command after it; with p = 0.25 a piece of a real sentence before and/or after it
(`train_oww_sent.py`), so it still wakes when the phrase is said mid-sentence. Negatives: sound-alikes, commands, real
words, **real sentences (30 % of negatives)**, Speech Commands, noise. Augmentation: speed 0.8-1.25, gain, noise.

## Training

| | |
|---|---|
| Script | `wakeword/train_oww_sent.py` (= `train_oww.py` + sentence negatives, `--sent-share 0.3 --ctx-p 0.25`) |
| Head | 16 × 96 embeddings → flatten → 2 × [Linear 128, LayerNorm, ReLU, dropout] → 1 logit; binary cross-entropy, labels smoothed to 0.025 / 0.975 |
| Data per epoch | 400,000 training windows (3 negatives per positive), 12,000 validation windows |
| Run | Slurm job **65684**, **CPU only** (16 cores, 64 GB), 40 epochs, **56 min**; best epoch 25 (validation recall at 1 % false accepts = 0.868) |
| Seed | 23101 (data-loader workers are seeded from the clock: a rerun is close, not bit-identical) |
| Log | `release/elkit_wakeword/train.log`, `train_history.json` |

Why sentences: the previous head (job 65357) saw mostly single words and commands as negatives, so ordinary running
speech often woke it: **16.8 %** of real test sentences at its field setting 0.94 × 2. Adding real sentences fixes that.

## Choosing the setting

All clips are scored with the deployed engine's exact streaming rule (`score_eval.py`), so the numbers describe what runs
on the device. `wake_metrics.py` evaluates any rule "N windows in a row ≥ T". The released setting **0.886 × 2** was chosen
on **dev** voices: it is the threshold at which the new head wakes on as many dev "Hello/Hi Elkit" phrases as the previous
head did at 0.94 × 2. The strictest rule (≤ 0.1 % of dev real speech wakes it) is also reported, but it misses most
real-speaker phrases, so it is not used.

## Test results (test voices, never used for training or for choosing the setting)

| Test set | **job 65684 @ 0.886 × 2 (released)** | 65684 @ 0.94 × 2 | previous 65357 @ 0.94 × 2 |
|---|---|---|---|
| Generated "Hello/Hi Elkit" (2,832) | **86.6 %** | 77.9 % | 86.2 % |
| Stitched from real speakers (800) | **65.6 %** | 42.9 % | 62.7 % |
| False wakes: **real sentences** (3,491) | **1.5 %** | 0.6 % | 16.8 % |
| False wakes: real single words (6,194) | **0** | 0 | 0 |
| False wakes: real ai231 commands (89) | **1** | 0 | 0 |
| Sound-alike phrases that wake it (3,834) | 8.8 % | 5.0 % | 7.7 % |

Full table, including dev numbers and the strict rule: `release/elkit_wakeword/wake_metrics.json`.

**Extra check (not part of the reproducible pipeline):** 176 "Hi/Hello Elkit" phrases from 10 commercial ElevenLabs
voices (mostly Filipino), never used for training: 99 % wake clean, 93 % / 84 % / 62 % with held-out room noise at
10 / 5 / 0 dB SNR (151 of the phrases).

## Limitations

- Positives are generated or stitched; nobody said "Elkit" naturally into the device for the test. Real voices are the
  weak spot (65.6 % of stitched real-speaker phrases), and loud rooms (0 dB) miss about 4 in 10.
- Sound-alikes such as "Hi, el kid" may wake it; this was accepted when the setting was chosen.
- Licences: Meta MMS-TTS voices are CC-BY-NC-4.0, so the head is for research / education only.

## Reproduce

```bash
bash wakeword/reproduce_wakeword.sh          # Slurm chain: download -> generate eval + training audio -> real splices
                                             # -> real sentences -> train -> score + wake_metrics.json
bash wakeword/reproduce_wakeword.sh --local  # same, one step after another
```
CPU except the Whisper "el" cutting step (one GPU). Several hours, about 60 GB under `$VCM_DATA`.
