# ME2 — Voice Controlled Smart Device

**Always-on keyword + intent recognition at the edge · no cloud round-trip**

14 October 2026 · Arjay Arnao, AI231 ME2 · Model · Dataset · Training on A100 · Validation on Raspberry Pi 5

Printable two-page version: [poster/poster.pdf](poster/poster.pdf) (drawn by `poster/make_poster.py`).

## Model

```
Mic · 16 kHz → log-mel 64 × ≤501 (10 ms hop, ≤ 5 s)
Wake word "Hello / Hi Elkit": frozen openWakeWord embedding → Elkit head (2 s window, every 80 ms)   [Marvin = backup]
Conformer encoder × 4 · 96-dim  (non-causal; runs once per command after the wake word, so no KV cache / streaming)
Intent head (19 + reject)         Slot head (separate 644 k Conformer, 18 values, 3 per slotted intent)
Actuator: the app acts on {"intent", "slot"} events          Raspberry Pi 5 (8 GB), ONNX Runtime, CPU
```

| Item | Value |
|---|---|
| Parameters / weights | intent **0.64 M · 2.7 MB** (ONNX) · slot 0.64 M · 2.7 MB · wake head 0.21 M · 0.9 MB (+ frozen 0.6 M front end, 2.4 MB) |

## Dataset

| Item | Value |
|---|---|
| Source | [ai231 ME2 voice commands](https://huggingface.co/datasets/airimonda/ai231-me2-voice-commands) @ `5abbe539` (Hugging Face; licence "other", research/education; DOI [10.57967/hf/10723](https://doi.org/10.57967/hf/10723)) |
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
| Keyword / intent acc | **86.6 %** generated voices, **65.6 %** real-speaker phrases (wake word, test, 0.886 × 2) / **98.39 %** (holdout, intent) · whole command (intent + slot) **98.39 %** holdout, **97.60 %** test |
| False-accept rate | wake word: **1.5 %** of real sentences, **0 / 6,194** real words, 1 / 89 real commands (test); intent out-of-scope: 31.6 % (test) |
| Latency p95 / RTF | **36.9 ms** / **0.01** per command (log-mel + intent + slot, p50 18.8 ms) · wake word 70.0 ms per 80 ms step / 0.22 (Pi 5, 1 thread, 202 holdout clips) |
| Runtime | onnxruntime · 1 thr |

### Live test on the Pi 5 (speaker → mic)

Class `vcm-benchmark`, 14 Oct 2026: the laptop plays the 202 holdout clips through its speaker, about 1 m from the Pi's
USB microphone (58 min). Wake word bypassed and actions not carried out (ELKIT recognised and logged each command only),
so slow actions could not affect later trials; the wake word is evaluated separately above.

| Item | Value |
|---|---|
| Intent / command | **91.1 %** / **90.6 %** (real voices 86.5 %, synthetic 95.3 %; 95 % CI 86-94 %) · slot exact **99.0 %** |
| Errors | wrong command fired **0.5 %** · command ignored 4.3 % · out-of-scope accepted 9 / 16 (56 %, 95 % CI 33-77 %) |
| Latency / RTF | inference **20.8 ms** mean, 42.5 ms p95 / **0.008** · response p50 / p95 0.99 / 1.21 s (includes the end-of-speech wait) |
| Pi load | 53.5 °C max, no throttling · ELKIT 5.7 % CPU, 157 MB RAM |

---

## To Be Submitted

| | |
|---|---|
| GitHub repository | [github.com/arjAIareNOW/MLEx-2-VCM-](https://github.com/arjAIareNOW/MLEx-2-VCM-) — public, MIT (code) |
| Dataset location | Hugging Face `airimonda/ai231-me2-voice-commands`, revision `5abbe539a46b9b26ff24e73d2860d1698d44e81f`; licence "other" (research/education); DOI 10.57967/hf/10723 |
| A100 cluster | uperdfi-a100gpu0, 1 × A100 80 GB, 29.8 min job (26.7 min training), seeds 23102 (training) / 23101 (split) |
| Model weights | `release/` and `deployment/models/` in the repository; research/education use (ai231 terms); Elkit head non-commercial (MMS-TTS data) |

## What the reviewer will look for

| # | Item | Status |
|---|---|---|
| 1 | Repo public, one-command reproduction | done: `bash reproduce.sh` (Slurm) or `bash reproduce.sh --local`; fresh-clone run checked 2026-10-14 on the build cluster (RTX A6000, intent + slot) |
| 2 | Dataset licensed and citable (DOI) | done: licence "other" (research/education), DOI 10.57967/hf/10723, pinned to a revision with SHA-256 checksums |
| 3 | Training logs + final checkpoint committed | done: `release/intent_conformer/` (best.pt, ONNX, config, history, log), `release/slot_conformer/` |
| 4 | Pi latency reproduced by the posted script | done: `deployment/benchmark_pi.py` on the Pi 5 → 36.9 ms p95, RTF 0.01 (1 thread); live class `vcm-benchmark`: 91.1 % intent, 21 ms inference, 1.21 s p95 response |
| 5 | Held-out test set, unseen speakers | done: test (104 speakers) and holdout (4 speakers) never seen in training or validation |
| 6 | Baseline of comparable size compared | done: CNN + BiGRU 405 k params, test 97.41 % vs Conformer 98.01 % (same data and recipe) |
