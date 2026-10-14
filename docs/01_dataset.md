# 1. The ai231 dataset

All intent and slot training, validation and testing uses one dataset: **ai231 ME2 voice-command dataset**
([airimonda/ai231-me2-voice-commands](https://huggingface.co/datasets/airimonda/ai231-me2-voice-commands) on Hugging Face),
pinned to revision **`5abbe539a46b9b26ff24e73d2860d1698d44e81f`**. `data_prep/download_data.py ai231` downloads that
revision and checks every file against `artifacts/checksums/ai231.sha256` (the dataset has newer revisions; do not use them
to reproduce these numbers).

- **License:** `other` on the dataset card. Research and education use only. Some parts (e.g. `supplemental_fil`) do not
  record speaker consent or license; see the dataset card. **DOI: [10.57967/hf/10723](https://doi.org/10.57967/hf/10723)**;
  cite the DOI together with the revision above.
- **Audio:** 16 kHz, mono, 16-bit PCM WAV stored inside Parquet files.

## What is in it

| Part | Clips | Hours | Speakers / voices | Used for |
|---|---:|---:|---:|---|
| `train` | 10,733 | 6.1 | 308 | intent + slot training, validation (15 % of speakers held out) |
| `test` | 4,443 | 2.45 | 115 | **final test only**, speakers never seen in training or validation |
| `holdout` | 202 | 0.18 | 5 | **Raspberry Pi benchmark set**, speakers never seen anywhere else |
| `numerals` | 66,390 | 21.8 | 2,547 | slot model only (number words; a balanced 7,000-clip subset) |
| `supplemental_fil` | 14,120 | 8.7 | 17 Filipino-accented cloned voices | training only |
| `supplemental_synth` | 5,856 | 2.7 | group synthetic voices | training only, and only voices that belong to train (3,983 clips) |
| `synthetic_negatives` | 1,250 | — | generated | out-of-scope examples (not used as an intent target) |

No speaker or synthetic voice appears in more than one of train / test / holdout (the dataset's own split).

## Labels

- **19 intents:** ALARM, BRIGHTNESS, CALL, COLOR, CREATE_REMINDER, LIGHT_OFF, LIGHT_ON, LIST_REMINDERS, MESSAGE, NEXT,
  PAUSE, PLAY_MUSIC, STOP, TEMPERATURE, TIME, TIMER, VOLUME_DOWN, VOLUME_UP, WEATHER, plus `OUT_OF_SCOPE`.
- **18 slot values**, 3 for each of the 6 slotted intents (from `variations.csv`, the 93 command variations):

  | Intent | Slot | Values |
  |---|---|---|
  | TIMER | duration | 10 seconds · 30 seconds · 1 minute |
  | ALARM | time | 6:00 AM · 8:00 AM · 9:00 PM |
  | TEMPERATURE | degrees | 18 · 22 · 26 degrees |
  | BRIGHTNESS | percent | 20 · 60 · 100 percent |
  | COLOR | color | Red · Blue · Green |
  | CREATE_REMINDER | reminder | Drink water · Study · Exercise |

## Who is in it

Real people (group recordings, SLURP, Fluent Speech Commands, SNIPS, Common Voice and other open corpora, Filipino and
other accents) and synthetic voices (the group's synthetic set and Filipino-accented voice clones). In `train`, 2,359 of
10,733 clips are real human speech; in `test` 824 of 4,443; the `holdout` has 96 real clips from one speaker
(`202520785`) and 106 synthetic clips.

## How the splits are used here

| Set | Source | Role |
|---|---|---|
| training | ai231_v2 `train` (non-validation speakers) + `supplemental_fil` + `supplemental_synth` (train voices) | fit the weights |
| validation | 15 % of the `train` speakers (43 speakers, 1,392 clips), see `artifacts/splits/` | **every choice**: epoch, raw vs EMA weights, which run, the reject threshold |
| test | official `test`, in-scope clips (4,367) + 76 out-of-scope | reported once, never used for a choice |
| holdout | official `holdout` (186 in-scope + 16 out-of-scope) | reported once; also the Raspberry Pi benchmark |

How ai231_v2 is made from the download is in [02_preprocessing.md](02_preprocessing.md).
