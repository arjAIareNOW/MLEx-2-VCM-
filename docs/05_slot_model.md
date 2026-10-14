# 5. Slot model (values for the 6 slotted commands)

Six commands carry a value: TIMER (10 s / 30 s / 1 min), ALARM (6:00 AM / 8:00 AM / 9:00 PM), TEMPERATURE (18 / 22 / 26
degrees), BRIGHTNESS (20 / 60 / 100 %), COLOR (red / blue / green) and CREATE_REMINDER (drink water / study / exercise).
A separate small model reads the **same log-mel features** as the intent model and outputs 18 logits; at inference only
the 3 values of the intent the intent model predicted compete (softmax over those 3).

## Architecture (`vcm/models.py: SlotTCDSCNN`)

Temporal depthwise-separable CNN, **202,898 parameters** (ONNX 0.9 MB):
stem Conv2D(1→64, 3×3, stride 2) → 2 depthwise-separable blocks (64→128→128 channels, frequency stride 2) →
6 dilated temporal blocks (dilations 1, 2, 4, 8, 16, 32; squeeze-excitation; ~2.6 s receptive field) →
frequency mean → attentive statistics pooling → Linear(256 → 18). Padding-masked like the intent model.

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

### The released run (`release/slot_tcdscnn/`)

1 × NVIDIA RTX A6000 (Slurm array job 65162, task 0), 72 min of training for 150 epochs; **selected epoch 112**
(5-epoch average validation score 0.965). The folder holds `best.pt`, `slot.onnx`, `config.json`, `history.csv`, the
training log and per-clip predictions.

The repo's `vcm/train_slot.py` was checked against the code that trained this model: the data (19,160 / 806 items,
12,918 word timings) is identical and, with the same seed on CPU, the first epoch is bit-identical. The original run also
scored an extra report-only validation set each epoch, which used random numbers; that report is not part of this repo,
so later epochs of a re-run follow a different (equally valid) random sequence.

## Results (slot accuracy given the correct intent)

| Split | Clips | Accuracy | Real-human clips |
|---|---:|---:|---:|
| validation | 806 | 98.76 % | 95.1 % (81 clips) |
| **test** | 2,538 | **98.50 %** | 90.0 % (259 clips) |
| test + unseen room noise, 20 / 10 / 5 dB SNR | 2,538 | 98.35 / 98.03 / 97.04 % | 89.6 / 86.9 / 83.4 % |
| **holdout** | 108 | **100 %** | 100 % (47 clips) |

Room noise for testing comes from 4 DEMAND environments never used in training (DLIVING, OMEETING, PCAFETER, STRAFFIC).
Real human speakers are harder than synthetic voices. The 38 clean test errors are mostly BRIGHTNESS 100 % heard as 20 %
(10) and colour mix-ups (20 among red / blue / green).

## Reproduce

```bash
python -m vcm.train_slot --output runs/slot_tcdscnn      # GPU, ~75 min on an A6000
python -m vcm.export_onnx slot runs/slot_tcdscnn          # slot.onnx + PyTorch/ONNX check
```
