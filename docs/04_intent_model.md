# 4. Intent model: Conformer (19 commands)

After the wake word, the command audio (up to 5 s) goes to the intent model, which picks one of the 19 commands or
rejects the clip as "not a known command".

## Architecture (`vcm/models.py: IntentConformer`)

```
audio 16 kHz ─► log-mel 64 × T (10 ms hop, ≤ 5 s → T ≤ 501)        vcm/features.py
            ─► conv stem: 2 × [Conv 3×3 stride 2 – BatchNorm – SiLU] (32 ch)  → 4× fewer frames (40 ms), 16 mel bins
            ─► Linear(32·16 → 96)
            ─► Conformer encoder × 4 (d_model 96, 4 attention heads, FFN 192, depthwise conv kernel 15, dropout 0.1)
            ─► attentive statistics pooling (weighted mean + std over time, padding masked) → 192
            ─► dropout 0.3 ─► Linear(192 → 19) ─► softmax ─► intent (+ reject if max probability < 0.693)
```

| | |
|---|---|
| Parameters | **643,699** (2.6 MB float32; ONNX 2.7 MB) |
| Encoder | non-causal; it sees the whole command clip at once (no streaming / KV cache needed: it runs once per command, after the wake word) |
| Padding | attention uses a key-padding mask and padded frames are zeroed before every depthwise convolution, so a clip gives the same logits alone or in a padded batch |
| Output | 19 logits (raw); out-of-scope is handled by the reject threshold, not by a 20th class |

## Training recipe (`vcm/train_intent.py`)

| | |
|---|---|
| Data | `ai231_v2` training clips: 25,620 unique (14.8 h, 246 speakers/voices), 26,268 per epoch with the LIGHT oversampling ([02](02_preprocessing.md)) |
| Validation | 1,392 clips, 43 speakers not in training |
| Objective | cross-entropy with class weights (inverse training frequency, mean 1) and label smoothing 0.1; mixup (α 0.3, p 0.5) |
| Optimiser | AdamW, peak lr 2e-3, weight decay 0.05 (weights only), 5-epoch linear warm-up then cosine to 1e-5 per step |
| Batch / length | 128 clips, 120 epochs max (206 steps per epoch), early stop after 30 epochs without improvement |
| Precision | bf16 autocast on the GPU, gradient clipping at 5 |
| Weights | exponential moving average (decay min(0.999, (1+n)/(10+n))); each epoch keeps the better of raw / EMA weights on validation |
| Augmentation | silence trim/pad, gain, speed, shift, noise, SpecAugment, mixup ([02](02_preprocessing.md)) |
| Selection | **validation macro-F1 only** (best epoch); test and holdout are scored once at the end |
| Seed | 23102 (training); 23101 (speaker split) |

### The released run (`release/intent_conformer/`)

| | |
|---|---|
| Hardware | 1 × **NVIDIA A100 80GB PCIe** (node `uperdfi-a100gpu0`), Slurm job 65672, process capped at 50 % of GPU memory, 5 CPU cores |
| Wall-clock | **26.7 min** training (29.8 min for the whole job, including data loading and test scoring) |
| Steps | 120 epochs × 206 = 24,720 optimiser steps; **best epoch 112** (raw weights) |
| Loss at the best epoch | train 0.879 (label-smoothed, with mixup) / validation 0.262 |

`release/intent_conformer/` holds the checkpoint (`best.pt`), its ONNX export, `config.json` (all settings, versions,
hashes), the per-epoch `training_history.csv`, the training log, the split manifest and per-clip test/holdout predictions.

The training code in this repo was checked against the code that produced the released model: with the same seed on
CPU it gives **bit-identical** losses and weights, and the ONNX export of the released checkpoint is byte-identical.
A new GPU run will not be bit-identical (cuDNN autotuning, bf16), but lands within normal seed-to-seed variation.

## Results

| Split | Clips | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| validation (used for selection) | 1,392 | 98.28 % | 0.978 |
| **test** (104 unseen speakers / voices) | 4,367 in-scope | **98.01 %** | **0.971** |
| **holdout** (Pi benchmark set, 4 unseen speakers) | 186 in-scope | **98.39 %** | 0.976 |

Hardest classes on test (F1): VOLUME_UP 0.90, VOLUME_DOWN 0.93, LIGHT_OFF 0.93, PLAY_MUSIC 0.94; most frequent
confusions VOLUME_UP ↔ VOLUME_DOWN (10), LIGHT_ON → LIGHT_OFF (4), STOP → PLAY_MUSIC (4): commands that differ by one word.

### Out-of-scope rejection (`vcm/reject_threshold.py`)

Rule fixed in advance: reject when the top softmax probability is below the 5th percentile of in-scope validation
confidences → **threshold 0.693** (keeps 95 % of validation commands).

| Split | In-scope accepted | Correct and accepted | Out-of-scope falsely accepted |
|---|---:|---:|---:|
| validation | 94.97 % | 94.18 % | 33.3 % (6 / 18) |
| test | 96.34 % | 95.47 % | 31.6 % (24 / 76) |
| holdout | 96.77 % | 96.24 % | 31.3 % (5 / 16) |

The out-of-scope numbers come from very few clips; in use the intent model only runs after the wake word, so most
unrelated speech never reaches it.

### Baseline of comparable size

One baseline, trained on exactly the same data, split, augmentation and selection rule, for scale:

| Model | Parameters | Validation acc. | Test acc. / macro-F1 | Holdout acc. |
|---|---:|---:|---:|---:|
| **Conformer (this model)** | 644 k | **98.28 %** | **98.01 % / 0.971** | 98.39 % |
| CNN + BiGRU (CRNN) baseline | 405 k | 97.20 % | 97.41 % / 0.963 | 99.46 % |

Choice by validation: the Conformer. (On the 186-clip holdout the baseline is 2 clips better; the holdout is too small
to decide on and was not used for selection.)

## Reproduce

```bash
python -m vcm.train_intent --output runs/intent_conformer --seed 23102      # GPU, ~30 min on an A100
python -m vcm.export_onnx intent runs/intent_conformer                       # intent.onnx + PyTorch/ONNX check
python -m vcm.reject_threshold runs/intent_conformer                         # reject_threshold.json
python -m vcm.train_intent --output /tmp/smoke --smoke                       # 2-epoch CPU check of the pipeline
```
