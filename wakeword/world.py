"""WORLD vocoder helpers (pyworld) used to simulate child / elderly voices for the wake-word data
(make_conditions.py, world_kws.py): F0 shift, formant (vocal-tract) warp, tempo change, intonation reshaping."""
import numpy as np
import pyworld

FS, FP = 16000, 5.0  # sample rate, WORLD frame period (ms)


def analyze(x):
    f0, t = pyworld.dio(x, FS, frame_period=FP); f0 = pyworld.stonemask(x, f0, t, FS)
    return f0, pyworld.cheaptrick(x, f0, t, FS), pyworld.d4c(x, f0, t, FS)


def frame_interp(a, idx):
    lo = np.floor(idx).astype(int); hi = np.minimum(lo + 1, len(a) - 1); w = (idx - lo)[:, None]
    return a[lo] * (1 - w) + a[hi] * w


def transform(f0, sp, ap, st=0.0, r=1.0, a=1.0, slope=0.0, wob=0.0, wob_hz=1.0, alpha=1.0):
    f0, sp, ap = f0.copy(), sp.copy(), ap.copy()
    v = f0 > 0
    if v.sum() > 2 and (a != 1.0 or slope or wob):  # intonation: reshape the log-F0 contour around its mean
        tn = np.arange(len(f0)) / max(1, len(f0) - 1); ts = np.arange(len(f0)) * FP / 1000
        lf = np.log(f0[v]); m = lf.mean()
        lf = m + a * (lf - m) + slope * (tn[v] - 0.5) + wob * np.sin(2 * np.pi * wob_hz * ts[v])
        f0[v] = np.exp(lf)
    if st: f0[v] *= 2 ** (st / 12)
    f0[v] = np.clip(f0[v], 50, 600)
    if alpha != 1.0:  # formant / vocal-tract warp of the spectral envelope
        src = np.clip(np.arange(sp.shape[1]) / alpha, 0, sp.shape[1] - 1)
        lo = np.floor(src).astype(int); hi = np.minimum(lo + 1, sp.shape[1] - 1); w = src - lo
        sp = sp[:, lo] * (1 - w) + sp[:, hi] * w
    if r != 1.0:  # tempo without pitch change: resample the frame sequence
        n = len(f0); idx = np.linspace(0, n - 1, max(2, int(round(n / r))))
        near = np.rint(idx).astype(int); f0n = np.interp(idx, np.arange(n), f0); f0n[f0[near] == 0] = 0
        f0, sp, ap = f0n, frame_interp(sp, idx), frame_interp(ap, idx)
    return np.ascontiguousarray(f0), np.ascontiguousarray(sp), np.ascontiguousarray(ap)


def synth(x, f0, sp, ap):
    y = pyworld.synthesize(f0, sp, ap, FS, FP)
    rx, ry = np.sqrt(np.mean(x ** 2)) + 1e-8, np.sqrt(np.mean(y ** 2)) + 1e-8
    return y * (rx / ry)
