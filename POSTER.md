# ME2 — Voice Controlled Smart Device

**Always-on keyword + intent recognition at the edge · no cloud round-trip**

14 October 2026 · Arjay Arnao, AI231 ME2 · Model · Dataset · Training on A100 · Validation on Raspberry Pi 5

## Model

```
Mic · 16 kHz → log-mel 64 × ≤501 (10 ms hop, ≤ 5 s)
Wake word "Hello / Hi Elkit": frozen openWakeWord embedding → Elkit head (2 s window, every 80 ms)   [Marvin = backup]
Conformer encoder × 4 · 96-dim  (non-causal; runs once per command after the wake word, so no KV cache / streaming)
Intent head (19 + reject)         Slot head (separate 203 k TC-DSCNN, 18 values, 3 per slotted intent)
Actuator: the app acts on {"intent", "slot"} events          Raspberry Pi 5 (8 GB), ONNX Runtime, CPU
```

| Item | Value |
|---|---|
| Parameters / weights | intent **0.64 M · 2.7 MB** (ONNX) · slot 0.20 M · 0.9 MB · wake head 0.21 M · 0.9 MB (+ frozen 0.6 M front end, 2.4 MB) |

## Dataset

| Item | Value |
|---|---|
| Source | [ai231 ME2 voice commands](https://huggingface.co/datasets/airimonda/ai231-me2-voice-commands) @ `5abbe539` (Hugging Face; licence "other", research/education; no DOI) |
| Hours / utts | intent training **14.8 h / 25,620** clips (+ 0.73 h / 1,392 validation); test 4,367 · holdout 186 |
| Speakers | **246** training speakers/voices, 43 validation, 104 test, 4 holdout (all disjoint) |
| Labels | **19 intents** · **18 slots** (3 values × 6 slotted intents) |

## Training on the A100 cluster

| Item | Value |
|---|---|
| Cluster | 1 × NVIDIA A100 80GB PCIe (node uperdfi-a100gpu0), 1 GPU, 5 CPU cores, **26.7 min** (job 65672), seed 23102 |
| Objective | cross-entropy, class-weighted, label smoothing 0.1, mixup (not CTC / attention decoder: utterance classification) |
| Optimiser | AdamW, lr 2e-3, wd 0.05, 5-epoch warm-up + cosine, batch 128, bf16, EMA |
| Steps / loss | 24,720 steps (120 epochs; best epoch 112) / train 0.879, validation 0.262 |

## Validation on the Raspberry Pi 5

| Item | Value |
|---|---|
| Keyword / intent acc | WAKE_TODO % / **98.39 %** (holdout, intent) |
| False-accept rate | WAKE_FA_TODO |
| Latency p95 / RTF | PI_TODO ms / PI_TODO |
| Runtime | onnxruntime · 1 thr |

---

## To Be Submitted

| | |
|---|---|
| GitHub repository | [github.com/arjAIareNOW/MLEx-2-VCM-](https://github.com/arjAIareNOW/MLEx-2-VCM-) — public, MIT (code) |
| Dataset location | Hugging Face `airimonda/ai231-me2-voice-commands`, revision `5abbe539a46b9b26ff24e73d2860d1698d44e81f`; licence "other" (research/education); no DOI yet |
| A100 cluster | uperdfi-a100gpu0, 1 × A100 80 GB, 29.8 min job (26.7 min training), seeds 23102 (training) / 23101 (split) |
| Model weights | `release/` and `deployment/models/` in the repository; research/education use (ai231 terms); Elkit head non-commercial (MMS-TTS data) |

## What the reviewer will look for

| # | Item | Status |
|---|---|---|
| 1 | Repo public, one-command reproduction | `bash reproduce.sh` (Slurm) or `bash reproduce.sh --local`; run end to end from a fresh clone before release |
| 2 | Dataset licensed and citable (DOI) | partial: licensed and pinned to a revision with SHA-256 checksums; no DOI |
| 3 | Training logs + final checkpoint committed | done: `release/intent_conformer/` (best.pt, ONNX, config, history, log), `release/slot_tcdscnn/` |
| 4 | Pi latency reproduced by the posted script | `deployment/benchmark_pi.py` + live `vcm-benchmark` run (results in `deployment/results/`) |
| 5 | Held-out test set, unseen speakers | done: test (104 speakers) and holdout (4 speakers) never seen in training or validation |
| 6 | Baseline of comparable size compared | done: CNN + BiGRU 405 k params, test 97.41 % vs Conformer 98.01 % (same data and recipe) |
