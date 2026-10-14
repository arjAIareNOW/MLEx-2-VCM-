"""Command line:
  python -m vcm_engine listen [--device N] [--mic-rate 48000]   live microphone, prints events as JSON lines
  python -m vcm_engine command FILE.wav                          intent + slot for a recorded command
  python -m vcm_engine wake FILE.wav                             Elkit score of a recording (streaming rule)
  python -m vcm_engine devices                                   list audio input devices
"""
import argparse
import json

from . import Assistant, CommandRecognizer, make_wake, read_wav


def main():
    ap = argparse.ArgumentParser(prog="vcm_engine")
    sub = ap.add_subparsers(dest="cmd", required=True)
    li = sub.add_parser("listen"); li.add_argument("--device", default=None); li.add_argument("--mic-rate", type=int)
    li.add_argument("--wake-threshold", type=float); li.add_argument("--wake-consecutive", type=int, help="windows in a row above the threshold"); li.add_argument("--show-levels", action="store_true")
    li.add_argument("--wake", choices=("elkit", "marvin"), default="elkit", help="marvin = backup wake word")
    li.add_argument("--bench-log", default="~/vcm_benchmark", help="folder for the vcm-benchmark log ('' = off)")
    li.add_argument("--bench-id", default="vcm", help="name prefix of the vcm-benchmark log file")
    sub.add_parser("command").add_argument("wav")
    wk = sub.add_parser("wake"); wk.add_argument("wav"); wk.add_argument("--wake", choices=("elkit", "marvin"), default="elkit")
    sub.add_parser("devices")
    a = ap.parse_args()
    if a.cmd == "devices":
        import sounddevice as sd
        print(sd.query_devices())
    elif a.cmd == "command":
        r = CommandRecognizer().predict(read_wav(a.wav)); print(json.dumps(r, indent=1))
    elif a.cmd == "wake":
        w = make_wake(a.wake); s = w.score_clip(read_wav(a.wav))
        print(json.dumps({"wake": a.wake, "score": s, "threshold": w.threshold, "detected": s >= w.threshold}))
    else:
        dev = int(a.device) if a.device is not None and str(a.device).isdigit() else a.device

        log = None
        if a.bench_log:  # one JSON line per command, the format read by github.com/airimonda/vcm-benchmark
            import datetime, os
            d = os.path.expanduser(a.bench_log); os.makedirs(d, exist_ok=True)
            log = open(os.path.join(d, f"{a.bench_id}_{datetime.datetime.now():%Y%m%d-%H%M%S}.log"), "a", buffering=1)
            print("vcm-benchmark log:", log.name, flush=True)

        def show(e):
            if e["type"] != "level" or a.show_levels:
                print(json.dumps({k: v for k, v in e.items() if k != "audio_wav"}), flush=True)
            if log is not None and e["type"] == "wake":
                print(json.dumps({"event": "wake", "probability": round(e["probability"], 4)}), file=log, flush=True)
            if log is not None and e["type"] == "command":
                print(json.dumps({"intent": e["intent"] if e["accepted"] else "OUT_OF_SCOPE",
                                  "slot": e["slot"] if e["accepted"] else None,
                                  "infer_ms": round(e["total_ms"], 1), "audio_ms": round(e["audio_s"] * 1000),
                                  "confidence": round(e["confidence"], 4)}), file=log, flush=True)

        asst = Assistant(on_event=show, device=dev, mic_rate=a.mic_rate, wake_threshold=a.wake_threshold, wake_consecutive=a.wake_consecutive, wake=a.wake)
        print(('Say "Marvin"' if a.wake == "marvin" else 'Say "Hello Elkit" or "Hi Elkit"') + ', pause briefly, then a command (e.g. "set a timer for 30 seconds"). Ctrl+C to quit.', flush=True)
        try:
            asst.run_microphone()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
