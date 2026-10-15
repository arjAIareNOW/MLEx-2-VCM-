"""Draw the two-page ME2 poster (course template layout) from the numbers in POSTER.md.
Writes poster/poster.pdf (2 pages) and poster/poster_page1.png, poster/poster_page2.png.
Usage: python poster/make_poster.py   (needs matplotlib)"""
import textwrap
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, Rectangle

OUT = Path(__file__).resolve().parent
W, H = 15.0, 10.0                      # page size in inches (template aspect)
NAVY, BLUE, ROW, ROW2 = "#1c2a43", "#1f5fd1", "#ffffff", "#f3f5f9"
SANS, MONO = "DejaVu Sans", "DejaVu Sans Mono"


def page():
    fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off"); fig.patch.set_facecolor("white")
    return fig, ax


def wrap(text, width_in, size, mono=False):
    chars = max(8, int(width_in / ((0.60 if mono else 0.54) * size / 72)))
    return "\n".join(textwrap.wrap(text, chars)) or " "


def heading(ax, x, y, text, size=19, color=BLUE):
    ax.text(x, y, text, fontsize=size, weight="bold", color=color, va="top", family=SANS)
    return y + size / 72 * 1.45


def table(ax, x, y, widths, header, rows, size=12, pad=0.07):
    """Template table: dark header row, light rows; long values wrap. Returns the y below the table."""
    lh = size / 72 * 1.22
    def draw(cells, y, fc, color, weight):
        texts = [wrap(c, w - 2 * pad, size) for c, w in zip(cells, widths)]
        h = max(t.count("\n") + 1 for t in texts) * lh + 2 * pad
        cx = x
        for t, w in zip(texts, widths):
            ax.add_patch(Rectangle((cx, y), w, h, fc=fc, ec="none"))
            ax.text(cx + pad, y + pad, t, fontsize=size, color=color, weight=weight, va="top", family=SANS, linespacing=1.1)
            cx += w
        ax.plot([x, x + sum(widths)], [y + h, y + h], color="#d5dbe6", lw=0.8)
        return y + h
    y = draw(header, y, NAVY, "white", "bold")
    for i, r in enumerate(rows):
        y = draw(r, y, ROW if i % 2 == 0 else ROW2, "#111111", "normal")
    return y


def block(ax, x, y, w, h, text, fc, ec, size=13.5, color="#14361a"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.08", fc=fc, ec=ec, lw=1.6))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, weight="bold", color=color, family=SANS)


# ------------------------------------------------------------------ page 1: model card
fig1, ax = page()
ax.text(0.45, 0.35, "ME2 — Voice Controlled Smart Device", fontsize=30, weight="bold", va="top", family=SANS)
ax.text(0.45, 1.0, "Always-on keyword + intent recognition at the edge · no cloud round-trip", fontsize=16.5, weight="bold", va="top", family=SANS)
ax.text(0.45, 1.45, "14 October 2026 · Arjay Arnao, AI231 ME2 · Model · Dataset · Training on A100 · Validation on Raspberry Pi 5",
        fontsize=14, va="top", family=SANS)

L, LW = 0.45, 7.05; R, RW = 7.85, 6.7
y = heading(ax, L, 2.0, "Model")
bh, gap = 0.40, 0.07
block(ax, L, y, LW, bh, "Mic · 16 kHz → log-mel 64 × ≤501 (10 ms hop, ≤ 5 s)", "#e8f0fb", "#8fb0e0", color="#16325c"); y += bh + gap
block(ax, L, y, LW, bh, "Wake word \"Hello / Hi Elkit\" · openWakeWord + Elkit head", "#e4f3f6", "#86c3d1", 12.5, "#0f4250"); y += bh + gap
block(ax, L, y, LW, bh + 0.14, "Conformer encoder × 4 · 96-dim\n", "#e6f4e4", "#7cc47a"); \
    ax.text(L + LW / 2, y + bh - 0.02, "non-causal · once per command after the wake word (no KV cache)", ha="center", va="center",
            fontsize=10.5, color="#2b5a2e", family=SANS); y += bh + 0.14 + gap
hw = (LW - 0.12) / 2
block(ax, L, y, hw, bh, "Intent head (19 + reject)", "#fdebd3", "#e6a85c", 12.5, "#6b3d05")
block(ax, L + hw + 0.12, y, hw, bh, "Slot head (18 values)", "#fdebd3", "#e6a85c", 12.5, "#6b3d05"); y += bh + gap
block(ax, L, y, hw, bh, "Actuator {intent, slot}", "#ece6f7", "#a08bd6", 12.5, "#3b2a6b")
block(ax, L + hw + 0.12, y, hw, bh, "Raspberry Pi 5 (8 GB)", "#e6f4e4", "#7cc47a", 12.5); y += bh + 0.18

y = table(ax, L, y, [2.2, LW - 2.2], ["Item", "Value"],
          [["Parameters / weights", "intent 0.64 M · 2.7 MB · slot 0.64 M · 2.7 MB · wake head 0.21 M · 0.9 MB (+ frozen 0.6 M front end, 2.4 MB)"]])
y = heading(ax, L, y + 0.18, "Validation on the Raspberry Pi 5")
table(ax, L, y, [2.2, LW - 2.2], ["Item", "Value"], [
    ["Keyword / intent acc", "wake word 86.6 % generated / 65.6 % real-speaker phrases (test, 0.886 × 2) · whole command "
                             "(intent + slot) 98.39 % holdout, 97.60 % test"],
    ["False-accept rate", "wake word 1.5 % of real sentences, 0 / 6,194 real words · out-of-scope commands accepted 31.6 % (test)"],
    ["Latency p95 / RTF", "36.9 ms / 0.01 per command (p50 18.8 ms) · wake word 70.0 ms per 80 ms step, RTF 0.22"],
    ["Runtime", "onnxruntime · 1 thr (Pi 5 CPU, 202 holdout clips)"],
])

y = heading(ax, R, 2.0, "Dataset")
y = table(ax, R, y, [1.75, RW - 1.75], ["Item", "Value"], [
    ["Source", "ai231 ME2 voice commands (Hugging Face airimonda/ai231-me2-voice-commands @ 5abbe539); licence \"other\", research/education"],
    ["Hours / utts", "14.8 h / 25,620 training clips (+ 0.73 h / 1,392 validation); test 4,367 · holdout 186"],
    ["Speakers", "246 training speakers/voices · 43 validation · 104 test · 4 holdout (all disjoint)"],
    ["Labels", "19 intents · 18 slots (3 values × 6 slotted intents)"],
])
y = heading(ax, R, y + 0.18, "Training on the A100 cluster")
y = table(ax, R, y, [1.75, RW - 1.75], ["Item", "Value"], [
    ["Cluster", "1 × NVIDIA A100 80GB (uperdfi-a100gpu0), 26.7 min training (job 65672), seed 23102"],
    ["Objective", "cross-entropy, class-weighted, label smoothing 0.1, mixup (utterance classification, not CTC / attention)"],
    ["Optimiser", "AdamW lr 2e-3, wd 0.05, 5-epoch warm-up + cosine, batch 128, bf16, EMA"],
    ["Steps / loss", "24,720 steps (120 epochs, best epoch 112) / train 0.879 · validation 0.262"],
])
y = heading(ax, R, y + 0.18, "Live test on the Pi 5 (speaker → mic)")
table(ax, R, y, [1.75, RW - 1.75], ["Item", "Value"], [
    ["Intent / command", "91.1 % / 90.6 % · slot 99.0 % (202 clips)"],
    ["Errors", "wrong 0.5 % · ignored 4.3 % · out-of-scope 9/16"],
    ["Latency / RTF", "21 ms (p95 43 ms) / 0.008 · reply p95 1.21 s"],
])

# ------------------------------------------------------------------ page 2: submission + reviewer checklist
fig2, ax = page()
ax.text(0.45, 0.45, "To Be Submitted", fontsize=30, weight="bold", va="top", family=SANS)
ax.plot([0.45, W - 0.45], [1.18, 1.18], color="#7cc242", lw=2.5)
x0, lw_, vw = 0.55, 3.2, W - 0.55 - 0.55 - 3.2
y, size, pad = 1.5, 13.5, 0.13
rows = [["GitHub repository", "github.com/arjAIareNOW/MLEx-2-VCM- — public, MIT (code)"],
        ["Dataset location", "Hugging Face airimonda/ai231-me2-voice-commands @ 5abbe539a46b9b26ff24e73d2860d1698d44e81f; "
                             "licence \"other\" (research/education); DOI 10.57967/hf/10723"],
        ["A100 cluster", "uperdfi-a100gpu0, 1 × A100 80 GB, 29.8 min job (26.7 min training), seeds 23102 (training) / 23101 (split)"],
        ["Model weights", "release/ and deployment/models/ in the repository; research/education use (ai231 terms); "
                          "Elkit head non-commercial (MMS-TTS data)"]]
top = y
for i, (k, v) in enumerate(rows):
    t = wrap(v, vw - 2 * pad, size - 1, mono=True); h = (t.count("\n") + 1) * size / 72 * 1.4 + 2 * pad
    ax.add_patch(Rectangle((x0, y), lw_, h, fc="#fdf1e1", ec="none"))
    ax.text(x0 + pad, y + h / 2, k, fontsize=size + 1, weight="bold", va="center", family=SANS)
    ax.text(x0 + lw_ + pad, y + pad, t, fontsize=size - 1, va="top", family=MONO, linespacing=1.25)
    if i: ax.plot([x0, x0 + lw_ + vw], [y, y], color="#ecd9bd", lw=0.9)
    y += h
ax.add_patch(FancyBboxPatch((x0, top), lw_ + vw, y - top, boxstyle="round,pad=0,rounding_size=0.06", fc="none", ec="#d99a3c", lw=2))
ax.plot([x0 + lw_, x0 + lw_], [top, y], color="#ecd9bd", lw=0.9)

y = heading(ax, 0.45, y + 0.4, "What the reviewer will look for", 19)
table(ax, 0.45, y, [0.55, 5.0, W - 0.9 - 5.55], ["#", "Item", "Status"], [
    ["1", "Repo public, one-command reproduction", "done: bash reproduce.sh (Slurm) or --local; fresh-clone run checked on the build cluster"],
    ["2", "Dataset licensed and citable (DOI)", "done: licensed (\"other\", research/education), DOI 10.57967/hf/10723, pinned revision + SHA-256"],
    ["3", "Training logs + final checkpoint committed", "done: release/intent_conformer/, release/slot_conformer/ (best.pt, ONNX, config, history, log)"],
    ["4", "Pi latency reproduced by the posted script", "done: deployment/benchmark_pi.py on the Pi 5 → 36.9 ms p95, RTF 0.01 (1 thread); live class vcm-benchmark: 91.1 % intent, 1.21 s p95 response"],
    ["5", "Held-out test set, unseen speakers", "done: test (104 speakers) and holdout (4 speakers) never seen in training or validation"],
    ["6", "Baseline of comparable size compared", "done: CNN + BiGRU 405 k params, test 97.41 % vs Conformer 98.01 % (same data and recipe)"],
], size=13)

with PdfPages(OUT / "poster.pdf", metadata={"Title": "ME2 — Voice Controlled Smart Device", "Author": "Arjay Arnao"}) as pdf:
    for f in (fig1, fig2): pdf.savefig(f)
fig1.savefig(OUT / "poster_page1.png", dpi=110); fig2.savefig(OUT / "poster_page2.png", dpi=110)
print("wrote", OUT / "poster.pdf", OUT / "poster_page1.png", OUT / "poster_page2.png")
