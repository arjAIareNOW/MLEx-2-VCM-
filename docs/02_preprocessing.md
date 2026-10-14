# 2. Preprocessing

Two kinds of preprocessing: **data cleaning** (which clips to train on, with which label), done once to build
`ai231_v2`, and the **audio front end + augmentation** applied to every clip at training time.

```
ai231 download ──► 2a Whisper label audit ──► 2b filter (lenient) ──► 2c ai231_v2 ──► intent model
   (pinned)          (frozen CSV in repo)       ai231_filtered_lenient     + label fixes
                                                        │                  + supplemental_synth
                                                        │                  + LIGHT oversampling list
                                                        └──► + 2d numerals subset + 2e slot-word timings ──► slot model
```

Every step only reads its inputs and writes a new folder under `$VCM_DATA` (default `./data`).
The repo scripts were checked against the original runs: they rebuild `ai231_v2` **byte for byte** and the filtered set
and numerals subset with identical contents.

## 2a. Whisper label audit (GPU, optional to re-run)

`data_prep/whisper_audit.py` transcribes every clip of train, supplemental_fil, test, holdout and the train-voice part
of supplemental_synth with **Whisper large-v3-turbo** (faster-whisper, fp16, beam 5, English).
`data_prep/analyze_whisper_labels.py` compares each transcript with the stored transcript and with the 93 command
phrases (RapidFuzz) and gives every clip a status:

| Status | Meaning |
|---|---|
| `OK` | Whisper agrees with the stored transcript (similarity ≥ 80) |
| `CLOSE_ACCENT_OR_WORDING` | similar (≥ 55) or close to one of its own command's phrases (≥ 75) |
| `OTHER_COMMAND_HEARD` | the audio is clearly a different command |
| `AUDIO_DOES_NOT_MATCH` / `NO_SPEECH_HEARD` | wrong or empty audio |
| `OOS_OK` / `OOS_LOOKS_LIKE_COMMAND` | out-of-scope clip that does / does not sound like a command |
| `slot_heard` | for slotted commands: is the slot value (number, colour, task) actually spoken |

The results used for every model are frozen in `artifacts/whisper/` (`label_audit.csv.gz`,
`label_audit_supplemental_synth.csv.gz`), so the rest of the pipeline needs no GPU and gives the same data.
Test and holdout were audited too, only to check their labels (all holdout speech matches its labels); **they are never
filtered or changed**.

## 2b. Filter → `ai231_filtered_lenient` (`data_prep/build_filtered.py`)

A training clip is kept when Whisper confirms its words and its label:

- status `OK`, or (lenient rule) a **real-human** clip with `CLOSE_ACCENT_OR_WORDING` whose words still map to its own
  command and whose slot is heard. This keeps accented real speakers, the rarest and most valuable training data;
- slotted commands must carry one of the 3 schema slot values ("alarm for 7" is dropped) and the value must be heard;
- out-of-scope clips are kept unless they sound like a clean command.

Result: train **10,118 of 10,733** kept, supplemental_fil **13,851 of 14,120** kept. Per-clip reasons are in
`filter_report.csv`. Test, holdout and `variations.csv` are links to the untouched originals.

## 2c. `ai231_v2` (`data_prep/build_ai231_v2.py`), the intent model's training set

1. **Reviewed label fixes** (`artifacts/review/label_fixes.csv`, checked by listening + Whisper): 4 clips relabelled to
   LIGHT_OFF (e.g. "off all the lights …" labelled LIGHT_ON), 20 dropped (19 out-of-scope clips that are really light
   commands or ambiguous, 1 brightness request without a schema value); 25 reviewed clips kept as they were.
2. **supplemental_synth**: only clips whose voice already belongs to train (`voice_split == train`), Whisper-audited
   with the same lenient rule: **3,894 of 3,983** kept. Test/holdout voices are never added.
3. **Oversampling list** `oversample.csv`: the 184 LIGHT_ON / LIGHT_OFF clips that are *not* worded exactly like a
   schema phrase ("light up", "lights off please", …) are repeated ×5 during training. These are the on/off minimal pairs
   the model confused most. Audio is stored once; only the training index repeats.
4. `synthetic_negatives` and test/holdout are links to the download, unchanged.

Training data seen by the intent model: **25,620 unique clips (14.8 h, 246 speakers/voices)**, 26,268 with
oversampling. Validation: **1,392 clips (0.73 h, 43 speakers)**. Validation speakers are excluded from every
supplemental set (588 clips dropped for that reason).

### Speaker-disjoint validation split (`vcm/data.py`)

15 % of the speakers of the original train split, drawn separately inside each (source, real/synthetic) group with
seed 23101 (at least one speaker when a group has ≥ 2). Speakers who also appear in supplemental_fil always stay in
training. The list is computed from the original download, checked against `artifacts/splits/validation_speakers.json`,
and the code asserts that no speaker and no identical audio is in both training and validation.

## 2d. Numerals subset (slot model only, `data_prep/numerals_subset.py`)

7,000 real-human clips from the ai231 `numerals` part: up to 700 for each of the 10 numbers that are slot values
(6, 8, 9, 10, 30, 1, 18, 20, 60, 100), drawn round-robin over speakers (seed 0) and labelled with their slot
(e.g. "30" → TIMER | 30 seconds).

## 2e. Slot-word timings (slot model only)

Word-level Whisper timestamps for the 6 slotted commands (`data_prep/whisper_slot_timestamps.py`, frozen in
`artifacts/whisper/slot_word_timestamps.jsonl.gz`). The slot model uses them to sometimes start a training clip just
before the slot word.

## Audio front end (`vcm/features.py`, same for intent and slot)

| Step | Setting |
|---|---|
| input | 16 kHz mono; first **5 s** (80,000 samples) |
| spectrogram | power mel, n_fft 512, **25 ms** Hann window (400), **10 ms** hop (160), **64 mels**, 20–8,000 Hz |
| scale | dB, top 80 dB |
| normalisation | per utterance: subtract mean, divide by std |
| output | `[1, 64, T]`, T ≤ 501 frames; batches padded, true lengths passed to the model (padding never changes the result) |

The Raspberry Pi uses a NumPy copy of this front end (`deployment/vcm_engine/features.py`); it matches the PyTorch
version to within 4e-7 in output confidence.

## Training-time augmentation (`vcm/augment.py`), training clips only

| Step | Probability | Setting |
|---|---|---|
| silence trim / re-pad | 0.5 | cut to the speech (energy VAD) + 0.05–0.25 s margin, re-pad to a random length up to 5 s (lead ≤ 1 s) with −90…−55 dBFS noise; with p 0.3 rescale peak to 0.03–0.9. Teaches the model that test/holdout clips have long silences and different loudness |
| gain | 0.5 | ±6 dB |
| speed | 0.5 | 0.9–1.1× (tempo and pitch) |
| time shift | 0.5 | ±100 ms |
| background noise | 0.5 | 10–25 dB SNR, Speech Commands background-noise files |
| extra noise | 0.3 | 5–10 dB SNR |
| SpecAugment | always | 2 frequency masks ≤ 8 mels, 2 time masks ≤ 10 % of the clip (≤ 40 frames) |
| mixup | 0.5 | α = 0.3 |

The slot model uses gain / speed / shift / noise (with DEMAND room noise added to the pool), frequency masks, placement
of the clip inside a 5 s window, and the slot-word crop; see [05_slot_model.md](05_slot_model.md).

Noise sources (downloaded by `data_prep/download_data.py`): the 6 background-noise files of Speech Commands v0.02
(CC-BY-4.0) and DEMAND (CC-BY-4.0, [philgzl/demand](https://huggingface.co/datasets/philgzl/demand) @ `29f46e61`).
They are noise only, not speech.
