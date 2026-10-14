# ME2 Voice-Controlled Smart Device

Always-on wake word + spoken-command recognition that runs **offline on a Raspberry Pi 5**: say
**"Hello Elkit"** (or "Hi Elkit"), then a command such as *"set a timer for 30 seconds"*. The device answers with one of
**19 intents** and, for six of them, one of **18 slot values**. No cloud round-trip.

```
mic 16 kHz ─► wake word "Hello / Hi Elkit" ─► command audio (≤ 5 s) ─► log-mel 64 ─► intent Conformer (19)
                (Marvin = backup)                                                  └► slot model (18 values)
```

This repository shows **how every model was built, from the raw dataset to the Pi**, and lets anyone reproduce it.

| Step | Doc | Code |
|---|---|---|
| 1. The ai231 dataset | [docs/01_dataset.md](docs/01_dataset.md) | `data_prep/download_data.py` |
| 2. Preprocessing (Whisper label audit, filtering, label fixes, features, augmentation) | [docs/02_preprocessing.md](docs/02_preprocessing.md) | `data_prep/`, `vcm/features.py`, `vcm/augment.py` |
| 3. Wake word "Hello / Hi Elkit" (+ Marvin backup) | [docs/03_wakeword_elkit.md](docs/03_wakeword_elkit.md) | `wakeword/` |
| 4. Intent model (Conformer) | [docs/04_intent_model.md](docs/04_intent_model.md) | `vcm/train_intent.py` |
| 5. Slot model | [docs/05_slot_model.md](docs/05_slot_model.md) | `vcm/train_slot.py` |
| 6. Raspberry Pi deployment and benchmarks | [docs/06_deployment_pi.md](docs/06_deployment_pi.md) | `deployment/` |
| Model card / poster | [POSTER.md](POSTER.md) | |

## Results at a glance

| | |
|---|---|
| Intent accuracy, test (4,367 clips, 104 unseen speakers) | **98.01 %** (macro-F1 0.971) |
| Intent accuracy, holdout (186 clips, Pi benchmark set) | **98.39 %** |
| Slot accuracy, test / holdout | **99.25 %** / 100 % |
| Whole command (intent and slot right), test / holdout | **97.60 %** / 98.39 % |
| Intent model | Conformer, 4 × 96-dim, **644 k parameters**, 2.7 MB ONNX, trained in 27 min on 1 × A100 |
| Slot model | same Conformer with 18 outputs, 644 k parameters, 2.7 MB ONNX, trained in 25 min on 1 × A100 |
| Wake word "Hello / Hi Elkit", test (0.886 × 2) | **86.6 %** generated voices, **65.6 %** real-speaker phrases; false wakes on **1.5 %** of real sentences, 0 / 6,194 real words ([docs/03](docs/03_wakeword_elkit.md)) |

Every choice (epoch, weights, run, thresholds) was made on **validation speakers only**; test and holdout were scored
once at the end.

## Quick start: use the released models (no training)

```bash
git clone https://github.com/arjAIareNOW/MLEx-2-VCM-.git && cd MLEx-2-VCM-/deployment
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt pyarrow
python tests/verify.py              # checks this machine reproduces the reference outputs
python -m vcm_engine listen         # microphone: "Hello Elkit ... turn on the lights"
python benchmark_pi.py --threads 1  # accuracy + latency on the ai231 holdout (downloads 15 MB)
```

## Reproduce the models from scratch

```bash
git clone https://github.com/arjAIareNOW/MLEx-2-VCM-.git && cd MLEx-2-VCM-
bash env/setup.sh                   # Python environments (exact versions in env/freeze/)
bash reproduce.sh                   # Slurm: download -> preprocess -> train intent + slot -> ONNX -> summary
bash reproduce.sh --local           # same, without Slurm, on one CUDA GPU
```

`reproduce.sh` downloads the pinned data (SHA-256 checked), rebuilds the training sets, trains both models, exports
ONNX, picks the reject threshold and writes `runs/summary.md` comparing your numbers with the released ones.
Time: ~30 min (intent, A100) + ~25 min (slot, A100) of GPU, about 12 GB of disk.
The wake word has its own pipeline: `bash wakeword/reproduce_wakeword.sh` (CPU, several hours, ~60 GB).

Checks done before release: the repo's training code gives **bit-identical** losses and weights to the code that
produced the released models (same seed, CPU), its ONNX export of the released checkpoint is byte-identical, and the
preprocessing scripts rebuild the training set byte for byte. GPU training is not bit-exact, so a fresh GPU run lands
close to, not exactly on, the released numbers.

Fresh-clone check (2026-10-14): `git clone` + `bash reproduce.sh` on the build cluster (1 × RTX A6000) completed end to end.
The retrained intent model reached 97.3 % test / 98.9 % holdout accuracy (released: 98.0 % / 98.4 %).

## Repository layout

```
data_prep/      download + preprocessing (Whisper audit, filter, ai231_v2, numerals subset)
vcm/            intent + slot models: features, augmentation, data split, models, training, ONNX export, reject threshold
wakeword/       Elkit wake word: TTS / real-speech data generation, training, evaluation
deployment/     Raspberry Pi engine (ONNX Runtime + NumPy), models/, benchmark_pi.py, tests/
release/        released checkpoints, ONNX, configs, training logs and curves, per-clip predictions
artifacts/      frozen inputs: Whisper audit, label fixes, validation speaker lists, checksums
env/            requirements and exact frozen environments
reproduce.sh    one-command reproduction (intent + slot)
POSTER.md       model card
```

## Licences

| What | Terms |
|---|---|
| Code in this repository | MIT ([LICENSE](LICENSE)) |
| ai231 dataset ([airimonda/ai231-me2-voice-commands](https://huggingface.co/datasets/airimonda/ai231-me2-voice-commands) @ `5abbe539`) | `other`: research and education only; not redistributed here, downloaded by script; DOI [10.57967/hf/10723](https://doi.org/10.57967/hf/10723) |
| Released intent / slot weights (`release/`, `deployment/models/`) | trained on ai231: research and education use only |
| Elkit wake-word head | trained partly on Meta MMS-TTS audio (CC-BY-NC-4.0): non-commercial |
| openWakeWord front end (`melspectrogram.onnx`, `embedding_model.onnx`) | Apache-2.0 (openWakeWord v0.5.1) |
| Noise / extra speech used in training | DEMAND, Speech Commands, MSWC, MLS: CC-BY-4.0; Piper voices: per-voice licences |
