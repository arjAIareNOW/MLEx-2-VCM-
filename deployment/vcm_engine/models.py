"""The deployed models, run with ONNX Runtime on CPU.

WakeWord           "Hello / Hi Elkit"   openWakeWord embeddings (frozen) + Elkit head; 2 s window every 80 ms
MarvinWakeWord     "Marvin" (backup)    1 s log-mel window every 100 ms (use when Elkit is not suitable)
CommandRecognizer  intent Conformer     19 intents, out-of-scope reject threshold
                   slot TC-DSCNN        18 slot values; only the 3 values of the predicted intent compete
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .features import WAKE_SAMPLES, command_features, wake_features

BUNDLE = Path(__file__).resolve().parents[1]
CONFIG = json.loads((BUNDLE / "models" / "config.json").read_text())


def _session(path: Path, threads: int) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads, so.inter_op_num_threads = threads, 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def _softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


class WakeWord:
    """Streaming "Hello Elkit" / "Hi Elkit" detector (one model for both). Feed 16 kHz float32 audio of any chunk size
    with push().

    Front end = openWakeWord v0.5.1 melspectrogram.onnx + embedding_model.onnx (frozen, Apache-2.0). Every 80 ms the last
    2 s of audio is turned into 197 mel frames -> 16 embeddings (one per 8 frames) -> Elkit head, exactly as
    in training. The mel model clips quiet frames relative to the loudest frame of its input, so the mel is always
    computed on the whole 2 s window; embeddings whose 76 input frames did not change are reused (with normal microphone
    audio only 1 of the 16 is new per step)."""

    WIN = 32000        # 2 s window
    STRIDE = 1280      # 80 ms = 8 mel frames = 1 embedding step
    EMB_FRAMES = 76    # mel frames per embedding
    EMB_STEP = 8       # mel frames between embeddings

    def __init__(self, threshold: float | None = None, consecutive: int | None = None, cooldown_s: float | None = None,
                 threads: int = 2):
        c = CONFIG["wake"]
        m = BUNDLE / "models"
        self.mel = _session(m / c["melspectrogram_onnx"], threads)
        self.emb = _session(m / c["embedding_onnx"], threads)
        self.head = _session(m / c["onnx"], threads)
        self.threshold = float(threshold if threshold is not None else c["streaming_threshold"])
        self.consecutive = int(consecutive or c["consecutive_hits"])
        self.cooldown = int(16000 * (cooldown_s if cooldown_s is not None else c["cooldown_ms"] / 1000))
        self.reset()

    def reset(self):
        self.buffer = np.zeros(self.WIN, np.float32)  # silence prefill: scoring starts immediately
        self.pending = np.zeros(0, np.float32)
        self.samples, self.hits, self.last_event = 0, 0, -10 ** 9
        self.cache = {}  # absolute step index of an embedding -> (its 76 mel frames, embedding)
        self.last_probability, self.last_infer_ms = 0.0, 0.0

    def probability(self, embeddings: np.ndarray) -> np.ndarray:
        """P(Elkit) for one or more runs of 16 embeddings ([16, 96] or [N, 16, 96])."""
        e = np.asarray(embeddings, np.float32).reshape(-1, 16, 96)
        return 1 / (1 + np.exp(-self.head.run(None, {"embeddings": e})[0].reshape(-1)))

    def _window(self, x: np.ndarray, step: int) -> float:
        """P(Elkit) of one 2 s window whose first sample is step * 1280 samples into the stream."""
        mel = self.mel.run(None, {"input": (x[None] * 32767).astype(np.float32)})[0][:, 0][0] / 10 + 2  # [197, 32]
        embs, todo = [None] * 16, []
        for j in range(16):
            w = mel[j * self.EMB_STEP: j * self.EMB_STEP + self.EMB_FRAMES]
            hit = self.cache.get(step + j)
            if hit is not None and np.allclose(hit[0], w, atol=1e-4): embs[j] = hit[1]
            else: todo.append(j)
        if todo:
            ins = np.stack([mel[j * self.EMB_STEP: j * self.EMB_STEP + self.EMB_FRAMES] for j in todo])[..., None]
            out = self.emb.run(None, {"input_1": ins.astype(np.float32)})[0].reshape(len(todo), 96)
            for j, e in zip(todo, out):
                embs[j] = e; self.cache[step + j] = (ins[todo.index(j), ..., 0], e)
        for k in [k for k in self.cache if k < step]: del self.cache[k]
        return float(self.probability(np.stack(embs))[0])

    def push(self, audio: np.ndarray) -> list[dict]:
        """Returns a list of wake events (usually empty). Each: {"time_s", "probability", "infer_ms"}."""
        self.pending = np.concatenate([self.pending, np.asarray(audio, np.float32).reshape(-1)])
        events = []
        while self.pending.size >= self.STRIDE:
            chunk, self.pending = self.pending[:self.STRIDE], self.pending[self.STRIDE:]
            self.buffer = np.concatenate([self.buffer[self.STRIDE:], chunk]); self.samples += self.STRIDE
            t = time.perf_counter(); p = self._window(self.buffer, self.samples // self.STRIDE)
            ms = (time.perf_counter() - t) * 1000
            self.last_probability, self.last_infer_ms = p, ms
            self.hits = self.hits + 1 if p >= self.threshold else 0
            if self.hits >= self.consecutive and self.samples - self.last_event >= self.cooldown:
                self.last_event, self.hits = self.samples, 0
                events.append({"time_s": self.samples / 16000, "probability": p, "infer_ms": ms})
        return events

    def score_clip(self, audio: np.ndarray) -> float:
        """Offline score of a short recording with the streaming rule: 1 s of silence before the clip, >= 2.5 s in total,
        best run of `consecutive` windows on the 80 ms grid: max over steps of the lowest P(Elkit) in the run, so
        score >= threshold exactly when push() would wake on this recording (consecutive=1: max P over all windows)."""
        x = np.asarray(audio, np.float32).reshape(-1)
        x = np.concatenate([np.zeros(16000, np.float32), x, np.zeros(max(8000, 24000 - x.size), np.float32)])
        saved = self.cache; self.cache = {}
        seq = np.array([self._window(x[s:s + self.WIN], s // self.STRIDE) for s in range(0, x.size - self.WIN + 1, self.STRIDE)])
        self.cache = saved
        n = min(max(self.consecutive, 1), seq.size)
        return float(np.lib.stride_tricks.sliding_window_view(seq, n).min(1).max())


class MarvinWakeWord:
    """Backup wake word "Marvin" (the earlier final wake word). Streaming: a 1 s window every 100 ms, fires after
    2 consecutive windows above the threshold. Same push() / score_clip() interface as WakeWord."""

    def __init__(self, threshold: float | None = None, consecutive: int | None = None, cooldown_s: float | None = None,
                 threads: int = 2):
        c = CONFIG["wake_backup"]
        self.session = _session(BUNDLE / "models" / c["onnx"], threads)
        self.input = self.session.get_inputs()[0].name
        self.threshold = float(threshold if threshold is not None else c["streaming_threshold"])
        self.consecutive = int(consecutive or c["consecutive_hits"])
        self.cooldown = int(16000 * (cooldown_s if cooldown_s is not None else c["cooldown_ms"] / 1000))
        self.stride = int(16000 * c["stride_ms"] / 1000)
        self.reset()

    def reset(self):
        self.buffer = np.zeros(WAKE_SAMPLES, np.float32)
        self.pending = np.zeros(0, np.float32)
        self.samples, self.hits, self.last_event = 0, 0, -10 ** 9
        self.last_probability, self.last_infer_ms = 0.0, 0.0

    def probability(self, windows: np.ndarray) -> np.ndarray:
        logits = self.session.run(None, {self.input: wake_features(windows)})[0]
        return 1 / (1 + np.exp(-logits.reshape(-1)))

    def push(self, audio: np.ndarray) -> list[dict]:
        self.pending = np.concatenate([self.pending, np.asarray(audio, np.float32).reshape(-1)])
        events = []
        while self.pending.size >= self.stride:
            chunk, self.pending = self.pending[:self.stride], self.pending[self.stride:]
            self.buffer = np.concatenate([self.buffer[self.stride:], chunk]); self.samples += self.stride
            t = time.perf_counter(); p = float(self.probability(self.buffer)[0]); ms = (time.perf_counter() - t) * 1000
            self.last_probability, self.last_infer_ms = p, ms
            self.hits = self.hits + 1 if p >= self.threshold else 0
            if self.hits >= self.consecutive and self.samples - self.last_event >= self.cooldown:
                self.last_event, self.hits = self.samples, 0
                events.append({"time_s": self.samples / 16000, "probability": p, "infer_ms": ms})
        return events

    def score_clip(self, audio: np.ndarray) -> float:
        x = np.concatenate([np.zeros(8000, np.float32), np.asarray(audio, np.float32), np.zeros(8000, np.float32)])
        if x.size < WAKE_SAMPLES:
            x = np.pad(x, (0, WAKE_SAMPLES - x.size))
        p = self.probability(np.stack([x[s:s + WAKE_SAMPLES] for s in range(0, x.size - WAKE_SAMPLES + 1, self.stride)]))
        return float(np.max(np.minimum(p[:-1], p[1:]))) if p.size > 1 else float(p[0])


def make_wake(name: str = "elkit", **kw):
    """"elkit" (default) or "marvin" (backup)."""
    return {"elkit": WakeWord, "marvin": MarvinWakeWord}[name](**kw)


class CommandRecognizer:
    """Intent + slot from the audio after the wake word (16 kHz float32, up to 5 s; longer audio is cut at 5 s)."""

    def __init__(self, reject_threshold: float | None = None, threads: int = 4):
        ci, cs = CONFIG["intent"], CONFIG["slot"]
        self.intent = _session(BUNDLE / "models" / ci["onnx"], threads)
        self.slot = _session(BUNDLE / "models" / cs["onnx"], threads)
        self.intents, self.slot_classes = ci["classes"], cs["classes"]
        self.slot_names = cs["slot_names"]
        self.reject_threshold = reject_threshold if reject_threshold is not None else ci.get("reject_threshold")
        self.groups = {}
        for i, c in enumerate(self.slot_classes):
            self.groups.setdefault(c.split("|")[0], []).append(i)

    def _run(self, session, feats):
        return session.run(None, {"log_mel": feats, "frame_lengths": np.array([feats.shape[-1]], np.int64)})[0][0]

    def predict(self, audio: np.ndarray) -> dict:
        t0 = time.perf_counter()
        feats = command_features(audio)
        t1 = time.perf_counter()
        p = _softmax(self._run(self.intent, feats))
        t2 = time.perf_counter()
        k = int(p.argmax()); intent = self.intents[k]
        out = {"intent": intent, "confidence": float(p[k]),
               "accepted": bool(self.reject_threshold is None or p[k] >= self.reject_threshold),
               "top3": [(self.intents[i], float(p[i])) for i in np.argsort(-p)[:3]],
               "slot_name": None, "slot": None, "slot_confidence": None,
               "audio_s": round(min(len(audio), 80000) / 16000, 3),
               "features_ms": (t1 - t0) * 1000, "intent_ms": (t2 - t1) * 1000, "slot_ms": 0.0}
        if intent in self.groups:  # slotted intent: pick one of its 3 values
            g = self.groups[intent]; z = self._run(self.slot, feats)[g]; q = _softmax(z); j = int(q.argmax())
            out.update(slot_name=self.slot_names[intent], slot=self.slot_classes[g[j]].split("|", 1)[1],
                       slot_confidence=float(q[j]), slot_ms=(time.perf_counter() - t2) * 1000)
        out["total_ms"] = (time.perf_counter() - t0) * 1000
        return out
