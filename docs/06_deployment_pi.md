# 6. Running on the Raspberry Pi 5

The device is a **Raspberry Pi 5 (8 GB)** with a USB microphone. Everything runs on its CPU with ONNX Runtime;
no internet, no PyTorch.

```
mic 16 kHz ─► wake word "Hello / Hi Elkit"  (every 80 ms, last 2 s of audio)
                 │ fires (probability ≥ threshold for N windows in a row)
                 ▼
             record the command until 0.55 s of silence (max 5 s)
                 ▼
             log-mel (NumPy) ─► intent Conformer ─► reject if confidence < 0.693 ─► slot model (6 slotted intents)
                 ▼
             event {"intent": "TIMER", "slot": "30 seconds", ...}  → printed, and logged for vcm-benchmark
```

## Files (`deployment/`)

| File | What |
|---|---|
| `models/melspectrogram.onnx`, `models/embedding_model.onnx` | openWakeWord v0.5.1 front end (frozen, Apache-2.0) |
| `models/hello_elkit_oww_head.onnx` | Elkit wake-word head ([03](03_wakeword_elkit.md)) |
| `models/marvin_v4_backup.onnx` | **backup** wake word "Marvin" (`--wake marvin`) |
| `models/intent_conformer.onnx` | intent model ([04](04_intent_model.md)) |
| `models/slot_conformer.onnx` | slot model ([05](05_slot_model.md)) |
| `models/config.json` | thresholds, class lists, front-end settings, SHA-256 of every model |
| `vcm_engine/` | the engine: audio, features (NumPy copy of the training front end), models, assistant loop, CLI |
| `benchmark_pi.py` | offline Pi benchmark (accuracy + latency on the ai231 holdout) |
| `tests/verify.py` | install check against outputs recorded on the build cluster |

## Install on the Pi

```bash
# on the Pi (Raspberry Pi OS, Python 3.11)
sudo apt install -y libportaudio2 git
git clone https://github.com/arjAIareNOW/MLEx-2-VCM-.git && cd MLEx-2-VCM-/deployment
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt pyarrow
python tests/verify.py                 # -> "N/N reference outputs reproduced - OK"
python -m vcm_engine listen            # say "Hello Elkit" ... "set a timer for 30 seconds"
```

Other commands: `python -m vcm_engine listen --wake marvin` (backup wake word), `--mic-rate 48000` for 48 kHz USB
microphones, `python -m vcm_engine command FILE.wav`, `python -m vcm_engine wake FILE.wav`, `python -m vcm_engine devices`.

## Benchmark 1: offline (same numbers on any machine)

```bash
python benchmark_pi.py --threads 1     # writes results/pi_benchmark_<host>_elkit_1t.json
python benchmark_pi.py --threads 4
```

Runs the 202 holdout clips through log-mel + intent + slot (batch 1, after warm-up) and 60 s of audio through the
streaming wake word. Reports device model, intent / slot accuracy, out-of-scope false accepts, latency p50 / p95 and
real-time factor. Accuracy must equal the numbers in [04](04_intent_model.md) / [05](05_slot_model.md) (it does on the
build cluster: intent 98.39 %, slot 100 %, out-of-scope false accepts 31.25 %); latency depends on the CPU.

## Benchmark 2: live, over the air ([airimonda/vcm-benchmark](https://github.com/airimonda/vcm-benchmark))

The class benchmark plays "wake word + command" from a laptop speaker to the Pi microphone and reads what the Pi logged.
`python -m vcm_engine listen` writes that log automatically to `~/vcm_benchmark/vcm_<date-time>.log`, one JSON line per
event: `{"event": "wake", "probability": ...}` and `{"intent": ..., "slot": ..., "infer_ms": ..., "audio_ms": ...,
"confidence": ...}` (a rejected command is logged as `OUT_OF_SCOPE`). `infer_ms` = log-mel + intent + slot time.

1. **On the Pi:** `cd MLEx-2-VCM-/deployment && . .venv/bin/activate && python -m vcm_engine listen`
   (leave it running).
2. **On the laptop:**
   ```bash
   git clone https://github.com/airimonda/vcm-benchmark.git && cd vcm-benchmark
   python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   python benchmark.py                               # record "Hello Elkit" 3 times when asked; size "full"
   ```
3. The report is saved in `vcm-benchmark/runs/<date-time>/`: wake-detect rate, intent and slot accuracy over the 196
   holdout clips, false-wake rate on commands played without the wake word, latency (p50 / p95), RTF, CPU temperature
   and RAM. These are the numbers on the poster ([POSTER.md](../POSTER.md)).

## Wake-word threshold

`models/config.json` → `wake.streaming_threshold` and `wake.consecutive_hits` (override with
`python -m vcm_engine listen --wake-threshold 0.94 --wake-consecutive 2` for fewer false wakes; the default is 0.886 × 2). See [03](03_wakeword_elkit.md) for what each setting catches and
how often unrelated speech wakes it.
