"""Raspberry Pi benchmark of the deployed pipeline (ONNX Runtime, CPU). Needs numpy, onnxruntime, pyarrow.

  commands : the ai231 HOLDOUT split (202 clips: 186 commands + 16 out-of-scope, speakers never used in training).
             Per clip: WAV decode + NumPy log-mel + intent model (+ slot model for slotted commands), batch 1.
             Reports intent accuracy, slot accuracy, out-of-scope false accepts with the reject threshold, latency
             p50 / p95 and real-time factor (processing time / audio length).
  wake     : streaming wake word on 60 s of audio (holdout clips joined): time per 80 ms step (Elkit) or 100 ms step
             (Marvin backup) and real-time factor.
The holdout file (15 MB) is downloaded once from the pinned dataset revision and checked by SHA-256.

Usage (from deployment/):  python benchmark_pi.py [--threads 1] [--wake elkit|marvin]
Writes results/pi_benchmark_<host>_<wake>_<threads>t.json
"""
import argparse, hashlib, io, json, os, platform, sys, time, urllib.request, wave
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from vcm_engine import CONFIG, CommandRecognizer, make_wake  # noqa: E402

HOLDOUT_URL = ("https://huggingface.co/datasets/airimonda/ai231-me2-voice-commands/resolve/"
               "5abbe539a46b9b26ff24e73d2860d1698d44e81f/data/holdout-00000-of-00001.parquet")
HOLDOUT_SHA = "c36227c81eee4eac63de62cee94631ee9e73952ccf8e16c27912eb049508eb6e"


def holdout(path: Path) -> Path:
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        print("downloading the ai231 holdout split (15 MB) ...", flush=True)
        urllib.request.urlretrieve(HOLDOUT_URL, path)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == HOLDOUT_SHA, f"checksum mismatch: {path}"
    return path


def decode(b: bytes) -> np.ndarray:
    with wave.open(io.BytesIO(b)) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16000, 1, 2)
        return np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768.0


def device_model() -> str:
    for p in ("/proc/device-tree/model", "/sys/firmware/devicetree/base/model"):
        try: return Path(p).read_text().strip("\x00\n ")
        except OSError: pass
    return platform.processor() or platform.machine()


def cpu_temp():
    try: return int(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000
    except (OSError, ValueError): return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threads", type=int, default=1); ap.add_argument("--wake", choices=("elkit", "marvin"), default="elkit")
    ap.add_argument("--holdout", type=Path, default=HERE / "data/holdout-00000-of-00001.parquet")
    ap.add_argument("--warmup", type=int, default=20)
    a = ap.parse_args()
    t = pq.read_table(holdout(a.holdout), columns=["audio", "command", "slot_value"]).to_pylist()
    audio = [decode(r["audio"]["bytes"]) for r in t]
    cr = CommandRecognizer(threads=a.threads)
    for x in audio[:a.warmup]: cr.predict(x)

    temp0 = cpu_temp(); ms, rtf, ok_i, n_i, ok_s, n_s, oos_acc, n_oos = [], [], 0, 0, 0, 0, 0, 0
    for r, x in zip(t, audio):
        t0 = time.perf_counter(); out = cr.predict(x); dt = time.perf_counter() - t0  # features + intent (+ slot)
        ms.append(dt * 1000); rtf.append(dt / (min(len(x), 80000) / 16000))
        if r["command"] == "OUT_OF_SCOPE":
            n_oos += 1; oos_acc += out["accepted"]
        else:
            n_i += 1; ok_i += out["intent"] == r["command"]
            if r["slot_value"] and out["intent"] == r["command"]:
                n_s += 1; ok_s += out["slot"] == r["slot_value"]

    wake = make_wake(a.wake, threads=a.threads)
    stream = np.concatenate(audio)[:60 * 16000]; step = wake.STRIDE if a.wake == "elkit" else wake.stride
    for i in range(0, 50 * step, step): wake.push(stream[i:i + step])  # warm-up
    wake.reset(); step_ms = []
    for i in range(0, len(stream) - step + 1, step):
        t0 = time.perf_counter(); wake.push(stream[i:i + step]); step_ms.append((time.perf_counter() - t0) * 1000)

    q = lambda v, p: round(float(np.percentile(v, p)), 2)
    res = {"device": device_model(), "machine": platform.machine(), "os": platform.platform(), "python": platform.python_version(),
           "runtime": "onnxruntime", "onnxruntime_version": __import__("onnxruntime").__version__, "threads": a.threads,
           "models": {k: CONFIG[k]["onnx"] for k in ("wake", "wake_backup", "intent", "slot")},
           "commands": {"clips": len(t), "intent_accuracy": round(ok_i / n_i, 4), "intent_clips": n_i,
                        "slot_accuracy_given_intent": round(ok_s / n_s, 4) if n_s else None, "slot_clips": n_s,
                        "reject_threshold": cr.reject_threshold, "oos_false_accept_rate": round(oos_acc / n_oos, 4) if n_oos else None,
                        "oos_clips": n_oos,
                        "latency_ms": {"p50": q(ms, 50), "p95": q(ms, 95), "mean": round(float(np.mean(ms)), 2),
                                       "scope": "log-mel + intent + slot (slotted intents), batch 1, after warm-up"},
                        "rtf": {"p50": q(rtf, 50), "p95": q(rtf, 95)}},
           "wake": {"model": a.wake, "step_ms": step / 16, "audio_s": len(stream) / 16000,
                    "latency_ms_per_step": {"p50": q(step_ms, 50), "p95": q(step_ms, 95)},
                    "rtf": round(float(np.sum(step_ms)) / 1000 / (len(stream) / 16000), 4)},
           "cpu_temp_c": {"start": temp0, "end": cpu_temp()}}
    out = HERE / "results" / f"pi_benchmark_{platform.node()}_{a.wake}_{a.threads}t.json"; out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res, indent=2) + "\n"); print(json.dumps(res, indent=2)); print("wrote", out)


if __name__ == "__main__":
    main()
