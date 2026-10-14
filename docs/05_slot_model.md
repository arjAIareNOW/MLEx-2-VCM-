# 5. Slot model (values for the 6 slotted commands)

Six commands carry a value: TIMER (10 s / 30 s / 1 min), ALARM (6:00 AM / 8:00 AM / 9:00 PM), TEMPERATURE (18 / 22 / 26
degrees), BRIGHTNESS (20 / 60 / 100 %), COLOR (red / blue / green) and CREATE_REMINDER (drink water / study / exercise).
A separate model reads the **same log-mel features** as the intent model and outputs 18 logits; at inference only
the 3 values of the intent the intent model predicted compete (softmax over those 3).

## Architecture (`vcm/models.py: IntentConformer` with 18 outputs)

The released slot model is the **same network as the intent model** ([04](04_intent_model.md)): Conformer encoder,
4 layers, d_model 96, 4 heads, **643,506 parameters** (ONNX 2.7 MB), with an 18-way output layer. `--arch tcdscnn`
trains the earlier, smaller slot model (`SlotTCDSCNN`, 202,898 parameters, ONNX 0.9 MB, kept in
`release/slot_tcdscnn/` for comparison).

## Training recipe (`vcm/train_slot.py`)

| | |
|---|---|
| Data | `ai231_filtered_lenient` slotted clips (train + supplemental_fil) + **7,000 real-human numerals** clips: 19,160 training items |
| Validation | 806 clips from 43 speakers not in training (`artifacts/splits/slot_validation_speakers.json`) |
| Objective | cross-entropy over the 3 values of the clip's intent, label smoothing 0.1 |
| Optimiser | AdamW lr 2e-3, weight decay 0.05, 5-epoch warm-up + cosine, batch 128, bf16, gradient clip 5, EMA weights |
| Length | 150 epochs, no early stopping, every epoch checkpointed (149 steps per epoch, 22,350 steps) |
| Augmentation | gain, speed, shift, noise (Speech Commands background + 13 DEMAND rooms, equal weight), p 0.5 placement in a 5 s window, p 0.2 crop starting just before the slot word (Whisper word timings), 2 frequency masks |
| Selection | validation score = mean(accuracy on real-human clips, accuracy on synthetic clips); best epoch = centre of the best 5-epoch moving average (one lucky epoch cannot win) |
| Seed | 23101 |

### The released run (`release/slot_conformer/`)

1 × NVIDIA A100 80GB (Slurm job 65703), 25 min of training for 150 epochs; **selected epoch 123** (5-epoch average
validation score 0.9849; 0.9815 at epoch 123). The folder holds `best.pt`, `slot.onnx`, `config.json`, `history.csv`, the training log and
per-clip predictions. It was trained by a copy of `vcm/train_slot.py` that differs from the repo's only by
experiment options left at their defaults (the repo keeps just `--arch`, default `conformer`).
Checked: with the same seed on CPU (`--smoke`), the repo code and that copy give identical logs and identical weights.
Re-exporting `best.pt` on another machine gives an ONNX file that differs only by float rounding of the folded
convolution weights (≤ 2.4e-7; logits ≤ 1.5e-6); `slot.onnx` here is the one exported inside job 65703.

### Choosing the model (validation only)

| Slot model | Parameters | Validation score, 5-epoch average (selection) | at the selected epoch |
|---|---:|---:|---:|
| TC-DSCNN, job 65162 (previous release) | 203 k | 0.9649 | 0.9712 |
| Conformer, A6000, job 65699 | 644 k | 0.9794 | 0.9794 |
| **Conformer, A100, job 65703 (released)** | 644 k | **0.9849** | **0.9815** |
| TC-DSCNN with real-human clips sampled 2× (job 65700) | 203 k | 0.9546 | 0.9547 |

65699 and 65703 are the same recipe and seed on different GPUs. On test they are practically tied (whole-command test
accuracy 97.64 % vs 97.60 %, 2 clips apart).

## Results (slot accuracy given the correct intent)

| Split | Clips | Conformer 65703 | Real-human clips | TC-DSCNN 65162 (previous) |
|---|---:|---:|---:|---:|
| validation | 806 | 99.63 % | 96.3 % (81 clips) | 98.76 % |
| **test** | 2,538 | **99.25 %** | 93.4 % (259 clips) | 98.50 % |
| test + unseen room noise, 20 / 10 / 5 dB SNR | 2,538 | 99.25 / 98.94 / 98.46 % | 93.4 / 90.7 / 87.3 % | 98.35 / 98.03 / 97.04 % |
| **holdout** | 108 | **100 %** | 100 % (47 clips) | 100 % |
| holdout, 5 dB SNR | 108 | 97.22 % | 95.7 % | 100 % |

Room noise for testing comes from 4 DEMAND environments never used in training (DLIVING, OMEETING, PCAFETER, STRAFFIC).
Real human speakers are harder than synthetic voices. The 19 clean test errors are COLOR (10, mostly blue heard as
green), ALARM (5, mostly 9:00 PM heard as 8:00 AM), TEMPERATURE (2) and BRIGHTNESS (2).

### Whole command (intent and slot both right)

Measured through the deployment engine (`deployment/vcm_engine`, NumPy log-mel + ONNX), intent model 65672 with
reject threshold 0.6934:

| | with TC-DSCNN 65162 | **with Conformer 65703** |
|---|---:|---:|
| Test, all commands (4,367) | 97.18 % | **97.60 %** |
| Test, real human voices (777) | 85.97 % | **87.00 %** |
| Test, correct and accepted (reject on) | 94.78 % | 95.12 % |
| Holdout, all commands (186) | 98.39 % | 98.39 % |

## Reproduce

```bash
python -m vcm.train_slot --output runs/slot_conformer    # GPU, ~25 min on an A100
python -m vcm.export_onnx slot runs/slot_conformer        # slot.onnx + PyTorch/ONNX check
```
