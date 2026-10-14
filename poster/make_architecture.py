"""Draw poster/conformer_architecture.png (intent model architecture, same layout as the reference CRNN figure).
Usage: python poster/make_architecture.py   (needs matplotlib, torch/torchaudio for the example spectrogram)"""
import sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))
OUT = REPO / "poster/conformer_architecture.png"

# example log-mel from a synthetic chirp-like "speech" signal (no dataset audio is stored in the figure)
t = np.linspace(0, 2.0, 32000); rng = np.random.default_rng(0)
env = np.clip(np.sin(np.pi * t / 2.0), 0, 1) ** 0.5 * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))
x = env * sum(np.sin(2 * np.pi * f * t * (1 + 0.15 * np.sin(2 * np.pi * 0.7 * t))) / k for k, f in enumerate((180, 360, 540, 900, 1400, 2300), 1))
x = (x + 0.02 * rng.standard_normal(t.size)).astype(np.float32) / 3
import torch
from vcm.features import LogMel
mel = LogMel()(torch.from_numpy(x)[None])[0].numpy()

C = dict(audio=("#e8f1fb", "#9fc3e9"), mel=("#e9f6e8", "#9fd39a"), stem=("#fdecef", "#f2a7b5"), proj=("#fff3e0", "#f5c47a"),
         enc=("#f0ebfa", "#b9a6e3"), pool=("#e6f4f8", "#95cbdc"), cls=("#e8f1fb", "#9fc3e9"), out=("#e9f6e8", "#9fd39a"))
fig = plt.figure(figsize=(17, 4.6), dpi=200); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 170); ax.set_ylim(0, 46); ax.axis("off")


def panel(x0, w, key, title, dashed=False):
    f, e = C[key]
    ax.add_patch(FancyBboxPatch((x0, 2), w, 36, boxstyle="round,pad=0,rounding_size=1.2", fc=f, ec=e, lw=1.4, ls="--" if dashed else "-"))
    ax.text(x0 + w / 2, 35.2, title, ha="center", va="top", fontsize=11.5, weight="bold")


def box(x0, y0, w, h, lines, key, bold_first=True, fs=9.2, step=2.35):
    f, e = C[key]
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0,rounding_size=0.8", fc="white" if key != "enc" else "#f7f3ff", ec=e, lw=1.3))
    for i, l in enumerate(lines):
        ax.text(x0 + w / 2, y0 + h - 1.2 - i * step, l, ha="center", va="top", fontsize=fs, weight="bold" if (i == 0 and bold_first) else "normal")


def arrow(x0, x1, y=20):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=14, lw=1.6, color="black"))


ax.text(1, 44.5, "Conformer Intent Model Architecture", fontsize=17, weight="bold", va="top")
ax.text(169, 44.5, "644 k parameters · 2.7 MB ONNX · 19 intents (+ reject)", fontsize=10.5, ha="right", va="top", color="#444")

# 1 audio
panel(1, 15, "audio", "Audio Input")
tt = np.linspace(0, 1, 400); wave = np.sin(2 * np.pi * 9 * tt) * np.exp(-((tt - .5) / .22) ** 2) * (0.6 + 0.4 * np.sin(2 * np.pi * 37 * tt))
ax.plot(2.5 + 12 * tt, 21 + 5 * wave, color="#2f6db5", lw=0.9)
ax.text(8.5, 11.5, "Mono, 16 kHz\n(after the wake word,\n≤ 5 s)", ha="center", fontsize=9.2)
arrow(16, 18.5)

# 2 log-mel
panel(18.5, 19, "mel", "Log-Mel\nSpectrogram")
ax.imshow(mel, origin="lower", aspect="auto", cmap="viridis", extent=(20.5, 35.5, 18.5, 28.5), zorder=3)
ax.set_xlim(0, 170); ax.set_ylim(0, 46)
ax.text(28, 16.8, "64 mel bands", ha="center", fontsize=9.5)
ax.text(20.3, 14.3, "· FFT: 512\n· Window: 25 ms\n· Hop: 10 ms\n· 20–8,000 Hz, dB\n· Per-utterance norm.", va="top", fontsize=8.3, linespacing=1.25)
arrow(37.5, 40)

# 3 conv stem
panel(40, 30, "stem", "Conv Stem\n(subsampling 4×)", dashed=True)
box(42, 8, 12.5, 18, ["Conv1", "3 × 3, stride 2", "1 → 32", "", "BatchNorm", "SiLU"], "stem")
box(56, 8, 12.5, 18, ["Conv2", "3 × 3, stride 2", "32 → 32", "", "BatchNorm", "SiLU"], "stem")
arrow(54.5, 56, 17)
arrow(70, 72.5)

# 4 projection
panel(72.5, 15, "proj", "Reshape +\nProjection")
ax.text(80, 27.5, "Flatten frequency\nand channels", ha="center", va="top", fontsize=9)
box(74.5, 13.5, 11, 7, ["512 features", "(32 × 16)"], "proj", bold_first=False, fs=8.8, step=2.6)
ax.text(80, 11.3, "Linear 512 → 96\n(40 ms / frame)", ha="center", va="top", fontsize=9)
arrow(87.5, 90)

# 5 conformer encoder
panel(90, 27, "enc", "Conformer Encoder\n× 4 layers (d = 96)")
for i, (name, sub) in enumerate((("Feed-forward ½", "96 → 192 → 96"), ("Self-attention", "4 heads, masked"), ("Conv module", "depthwise k = 15"), ("Feed-forward ½", "+ LayerNorm"))):
    box(93, 25.2 - i * 5.9, 21, 5.2, [name, sub], "enc", fs=8.2, step=2.0)
ax.text(103.5, 3.4, "non-causal, padding-masked", ha="center", fontsize=8.3, color="#555")
arrow(117, 119.5)

# 6 pooling
panel(119.5, 15, "pool", "Attentive\nStats Pooling")
ax.text(127, 27.5, "Attention weights\nover time", ha="center", va="top", fontsize=9)
box(121.5, 13.5, 11, 7, ["mean ⊕ std", "96 + 96 = 192"], "pool", bold_first=False, fs=8.8, step=2.6)
ax.text(127, 11, "valid frames only", ha="center", va="top", fontsize=8.5, color="#555")
arrow(134.5, 137)

# 7 classifier
panel(137, 16, "cls", "Classifier")
box(139, 24.5, 12, 5.5, ["Dropout", "(p = 0.3)"], "cls", bold_first=False, fs=8.6, step=2.2)
box(139, 16, 12, 5.5, ["Linear", "(192 → 19)"], "cls", bold_first=False, fs=8.6, step=2.2)
box(139, 7.5, 12, 5.5, ["Softmax", "reject < 0.693"], "cls", bold_first=False, fs=8.6, step=2.2)
for y0, y1 in ((24.5, 21.5), (16, 13)):
    ax.add_patch(FancyArrowPatch((145, y0), (145, y1), arrowstyle="-|>", mutation_scale=10, lw=1.2))
arrow(153, 155.5)

# 8 output
panel(155.5, 13.5, "out", "Output")
for i, h in enumerate((4, 6.5, 5, 9, 7.5)):
    ax.add_patch(plt.Rectangle((157.5 + i * 1.9, 20), 1.3, h, color="#3f9b48"))
ax.text(162.2, 17.5, "19 intents", ha="center", va="top", fontsize=10.5, weight="bold")
ax.text(162.2, 13.5, "e.g. TIMER\n+ slot model:\n\"30 seconds\"", ha="center", va="top", fontsize=8.6)

fig.savefig(OUT, dpi=200, facecolor="white"); print("wrote", OUT)
