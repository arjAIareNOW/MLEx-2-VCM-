"""Train the "Hello / Hi Elkit" wake-word head on a FROZEN pretrained speech embedding (openWakeWord recipe), CPU only.

Why: the from-scratch KWS learns its acoustics from TTS alone and does not transfer to unseen voices/accents. openWakeWord's
front end (melspectrogram.onnx + embedding_model.onnx, a port of Google's speech_embedding, trained on large amounts of REAL
speech) is kept fixed; only a small classifier on 16 x 96 embeddings (= 2 s window, one embedding per 80 ms) is trained.
Models: deployment/models/melspectrogram.onnx + embedding_model.onnx (openWakeWord v0.5.1 release, Apache-2.0).

Training windows come from kws_data.Data (TTS + real-word data, augmentation, DEMAND training rooms, real ai231 + Speech Commands
negatives) with a 2 s window. Features for N augmented windows are precomputed once (single process, threaded ORT), then the classifier trains
for a few epochs on them. Checkpoint = best recall at FAR <= 1 % on the val speakers. Eval clips are scored in streaming
fashion (mel over the clip, classifier on every run of 16 embeddings, max) -> OUT/scores_<tag>.csv.
Usage: python wakeword/train_oww.py --kws-data $VCM_DATA/elkit/kws --eval-clips $VCM_DATA/elkit/eval --index index_real.csv --out runs/elkit
"""
import argparse, csv, json, os, random, sys, time
from pathlib import Path
import numpy as np, soundfile as sf, torch, torch.nn as nn, torch.nn.functional as F
import onnxruntime as ort

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import kws_data as K

K.WIN = WIN = 32000  # 2 s: 197 mel frames -> 16 embeddings
OWW = HERE.parent / "deployment/models"
_S = {}


def sessions(threads=1):
    if not _S:
        so = ort.SessionOptions(); so.intra_op_num_threads = threads; so.inter_op_num_threads = 1
        _S["mel"] = ort.InferenceSession(str(OWW / "melspectrogram.onnx"), so, providers=["CPUExecutionProvider"])
        _S["emb"] = ort.InferenceSession(str(OWW / "embedding_model.onnx"), so, providers=["CPUExecutionProvider"])
    return _S["mel"], _S["emb"]


def embed(x):
    """float audio [-1, 1] at 16 kHz (B, n) -> (B, T, 96) embeddings, one per 8 mel frames (80 ms)."""
    mel, emb = sessions()
    s = mel.run(None, {"input": (x * 32767).astype(np.float32)})[0][:, 0] / 10 + 2  # (B, F, 32)
    idx = np.arange(0, s.shape[1] - 76 + 1, 8)
    w = np.stack([s[:, i:i + 76] for i in idx], 1)  # (B, T, 76, 32)
    B, T = w.shape[:2]
    return emb.run(None, {"input_1": w.reshape(B * T, 76, 32, 1).astype(np.float32)})[0].reshape(B, T, 96)


DATA = {}


def features(split, total, ratio, aug_p, seed, batch=256):
    """augmented windows are built in this process (no fork: torch + fork deadlocked); ORT uses intra-op threads."""
    random.seed(seed); np.random.seed(seed % 2**32)
    d, xs, ys = DATA[split], [], []
    for i in range(0, total, batch):
        y = (np.random.rand(min(batch, total - i)) < 1 / (1 + ratio)).astype(np.float32)
        x = np.stack([d.example(int(t), aug=random.random() < aug_p) for t in y])
        xs.append(embed(x).astype(np.float16)); ys.append(y)
        if (i // batch) % 200 == 0: print(f"  {split} features {i + len(y)}/{total}", flush=True)
    return np.concatenate(xs), np.concatenate(ys)


class Head(nn.Module):
    def __init__(self, h=128, p=0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(16 * 96, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(p),
                                 nn.Linear(h, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(p), nn.Linear(h, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def score_clip(head, x):
    pad = np.zeros(16000, np.float32)
    tail = max(8000, 40000 - 16000 - len(x))  # >= 2.5 s total so short real word clips still give >= 16 embedding frames
    x = np.concatenate([pad, x.astype(np.float32), np.zeros(tail, np.float32)])
    e = torch.from_numpy(embed(x[None])[0])  # (T, 96)
    runs = e.unfold(0, 16, 1).permute(0, 2, 1)  # (R, 16, 96)
    with torch.inference_mode():
        return float(head(runs).sigmoid().max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kws-data", type=Path, default=K.DATA_ROOT / "elkit/kws")
    ap.add_argument("--eval-clips", type=Path, default=K.DATA_ROOT / "elkit/eval")
    ap.add_argument("--out", type=Path, required=True); ap.add_argument("--tag", default="elkit")
    ap.add_argument("--n-train", type=int, default=400000); ap.add_argument("--n-val", type=int, default=12000)
    ap.add_argument("--ratio", type=float, default=3); ap.add_argument("--neg-share", type=float, default=0.45)
    ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--index", default="index_real.csv")
    ap.add_argument("--score-only", action="store_true", help="skip training, score eval clips with OUT/best.pt")
    a = ap.parse_args()
    if a.smoke: a.n_train, a.n_val, a.epochs = 2048, 1024, 2
    a.out.mkdir(parents=True, exist_ok=True); torch.manual_seed(23101); random.seed(23101); np.random.seed(23101)
    t0 = time.time()
    if not a.score_only: train(a, t0)
    score(a, t0)


def train(a, t0):
    DATA["train"] = K.Data(a.kws_data, "train", n_real=200 if a.smoke else 3000, n_sc=200 if a.smoke else 6000, neg_share=a.neg_share, index=a.index)
    DATA["val"] = DATA["train"] if a.smoke else K.Data(a.kws_data, "val", n_real=3000, n_sc=6000, seed=0, neg_share=a.neg_share, index=a.index)
    sessions(a.workers); torch.set_num_threads(1)
    xt, yt = features("train", a.n_train, a.ratio, 1.0, 1000)
    xv, yv = features("val", a.n_val, a.ratio, 0.5, 9_000_000)
    print(f"features train {xt.shape} val {xv.shape} pos {yt.mean():.2f} in {time.time() - t0:.0f}s", flush=True)
    torch.set_num_threads(a.workers)
    head = Head(a.hidden); opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.01)
    xt_t, yt_t, xv_t = torch.from_numpy(xt), torch.from_numpy(yt), torch.from_numpy(xv).float()
    steps = a.epochs * ((len(xt) + 511) // 512); sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps)
    best, hist = -1, []
    for ep in range(a.epochs):
        head.train(); perm = torch.randperm(len(xt)); tl = 0
        for i in range(0, len(xt), 512):
            b = perm[i:i + 512]
            loss = F.binary_cross_entropy_with_logits(head(xt_t[b].float()), yt_t[b] * 0.95 + 0.025)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); tl += float(loss) * len(b)
        head.eval()
        with torch.inference_mode(): p = head(xv_t).sigmoid().numpy()
        rec, th = K.far_recall(p, yv)
        hist.append(dict(epoch=ep + 1, train_loss=tl / len(xt), val_recall_at_far1=rec, val_threshold=th)); print(json.dumps(hist[-1]), flush=True)
        if rec > best: best = rec; torch.save(dict(state=head.state_dict(), epoch=ep + 1, val_recall=rec, hidden=a.hidden), a.out / "best.pt")
    json.dump(dict(args={k: str(v) for k, v in vars(a).items()}, history=hist), open(a.out / "train_history.json", "w"), indent=1)


def score(a, t0):

    ck = torch.load(a.out / "best.pt", weights_only=True); head = Head(ck["hidden"]); head.load_state_dict(ck["state"]); head.eval()
    torch.set_num_threads(1); _S.clear(); sessions(2)  # time scoring with 2 threads, like the KWS
    rows = list(csv.DictReader(open(a.eval_clips / "index.csv")))
    if a.smoke: rows = rows[:40]
    res = []
    for r in rows:
        x, sr = sf.read(a.eval_clips / r["file"], dtype="float32", always_2d=True); x = x.mean(1); assert sr == 16000
        t = time.perf_counter(); s = score_clip(head, x); res.append((r["file"], s, (time.perf_counter() - t) * 1000))
    with open(a.out / f"scores_{a.tag}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["file", "score", "ms"]); w.writerows(res)
    torch.onnx.export(head, torch.zeros(1, 16, 96), str(a.out / "hello_elkit_oww_head.onnx"), input_names=["embeddings"],
                      output_names=["logit"], dynamic_axes={"embeddings": {0: "batch"}}, opset_version=17)
    print("best epoch", ck["epoch"], "val recall@FAR1%", ck["val_recall"], "scored", len(res), f"total {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
