"""Training-only augmentation (docs/02_preprocessing.md). Validation / test / holdout audio is never augmented.

Waveform chain (IntentAugmentor.__call__, in this order):
  1. silence trim/pad   p=0.5  cut to the energy-VAD speech span + 0.05-0.25 s margin, re-pad to a random length in
                               [speech, 5 s] (lead <= 1 s) with -90..-55 dBFS noise; then p=0.3 rescale to peak 0.03-0.9
  2. gain               p=0.5  +/-6 dB
  3. speed              p=0.5  0.9-1.1x (tempo + pitch, linear resampling)
  4. time shift         p=0.5  +/-100 ms, zero-filled
  5. background noise   p=0.5  10-25 dB SNR
  6. extra noise        p=0.3  5-10 dB SNR
Feature level (on the GPU batch): SpecAugment (2 frequency masks <= 8 mels, 2 time masks <= 10 % of the clip, <= 40
frames) and mixup (alpha 0.3, p=0.5). The slot model uses steps 2-6 only, plus DEMAND rooms in the noise pool.
"""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

FRAME, HOP, SR, MAX = 400, 160, 16_000, 80_000


def load_noise_dir(noise_dir: Path) -> list[Tensor]:
    files = sorted(Path(noise_dir).glob("*.wav"))
    if not files:
        raise FileNotFoundError(f"No background-noise WAV files in {noise_dir} (run data_prep/download_noise.py)")
    out = []
    for path in files:
        with wave.open(str(path), "rb") as h:
            if h.getframerate() != SR or h.getnchannels() != 1 or h.getsampwidth() != 2:
                raise ValueError(f"Noise must be mono 16-kHz PCM16: {path}")
            out.append(torch.from_numpy(np.frombuffer(h.readframes(h.getnframes()), dtype="<i2").copy()).float().div_(32768.0))
    return out


def speech_span(w: np.ndarray) -> tuple[int, int]:
    """Energy VAD: frames louder than max(10th percentile + 12 dB, peak - 35 dB)."""
    n = 1 + (w.size - FRAME) // HOP
    fr = np.lib.stride_tricks.as_strided(w, (n, FRAME), (w.strides[0] * HOP, w.strides[0]))
    db = 10 * np.log10((fr.astype(np.float64) ** 2).mean(1) + 1e-10)
    idx = np.flatnonzero(db > max(np.percentile(db, 10) + 12, db.max() - 35))
    return (0, w.size) if idx.size == 0 else (idx[0] * HOP, min(idx[-1] * HOP + FRAME, w.size))


def uniform(lo: float, hi: float) -> float:
    return lo + (hi - lo) * float(torch.rand(()))


class WaveformAugmentor:
    """Steps 2-6 (gain, speed, shift, noise, extra noise)."""

    def __init__(self, noises: list[Tensor]) -> None:
        self.noises = noises

    def __call__(self, waveform: Tensor) -> Tensor:
        if torch.rand(()) < 0.5:  # gain +/-6 dB
            waveform = waveform * 10.0 ** ((torch.rand(()) * 12 - 6) / 20)
        if torch.rand(()) < 0.5:  # speed 0.9-1.1
            factor = 0.9 + 0.2 * float(torch.rand(()))
            length = max(400, int(round(waveform.shape[-1] / factor)))
            waveform = nn.functional.interpolate(waveform[None], size=length, mode="linear", align_corners=False)[0][..., :MAX]
        waveform = waveform.clamp(-1.0, 1.0)
        if torch.rand(()) < 0.5:  # zero-filled shift, +/-100 ms
            shift = int(torch.randint(-1600, 1601, ()).item())
            if shift > 0:
                waveform = nn.functional.pad(waveform, (shift, 0))[..., : waveform.shape[-1]]
            elif shift < 0:
                waveform = nn.functional.pad(waveform[..., -shift:], (0, -shift))
        if torch.rand(()) < 0.5:  # background noise 10-25 dB SNR
            noise = self.noises[int(torch.randint(len(self.noises), ()).item())]
            if len(noise) < waveform.shape[-1]:
                noise = noise.repeat((waveform.shape[-1] + len(noise) - 1) // len(noise))
            start = int(torch.randint(len(noise) - waveform.shape[-1] + 1, ()).item())
            segment = noise[start:start + waveform.shape[-1]].unsqueeze(0)
            speech_rms = waveform.square().mean().sqrt().clamp_min(1e-5)
            noise_rms = segment.square().mean().sqrt().clamp_min(1e-5)
            snr_db = 10.0 + 15.0 * float(torch.rand(()))
            waveform = (waveform + segment * (speech_rms / (noise_rms * (10.0 ** (snr_db / 20.0))))).clamp(-1.0, 1.0)
        if torch.rand(()) < 0.3:  # extra low-SNR pass, 5-10 dB
            waveform = self._noise(waveform, 5.0 + 5.0 * float(torch.rand(())))
        return waveform

    def _noise(self, waveform: Tensor, snr_db: float) -> Tensor:
        noise = self.noises[int(torch.randint(len(self.noises), ()).item())]
        if len(noise) < waveform.shape[-1]:
            noise = noise.repeat((waveform.shape[-1] + len(noise) - 1) // len(noise))
        start = int(torch.randint(len(noise) - waveform.shape[-1] + 1, ()).item())
        segment = noise[start:start + waveform.shape[-1]].unsqueeze(0)
        scale = waveform.square().mean().sqrt().clamp_min(1e-5) / (
            segment.square().mean().sqrt().clamp_min(1e-5) * 10.0 ** (snr_db / 20.0))
        return (waveform + segment * scale).clamp(-1.0, 1.0)


class IntentAugmentor(WaveformAugmentor):
    """Step 1 (silence trim/pad) followed by steps 2-6."""

    def __init__(self, noises: list[Tensor], trimpad_prob: float = 0.5) -> None:
        super().__init__(noises); self.trimpad_prob = trimpad_prob

    def __call__(self, waveform: Tensor) -> Tensor:
        if waveform.shape[-1] >= FRAME and float(torch.rand(())) < self.trimpad_prob:
            waveform = self.trim_pad(waveform)
        return super().__call__(waveform)

    @staticmethod
    def trim_pad(waveform: Tensor) -> Tensor:
        a, b = speech_span(waveform[0].numpy())
        a = max(0, a - int(uniform(0.05, 0.25) * SR)); b = min(waveform.shape[-1], b + int(uniform(0.05, 0.25) * SR))
        speech = waveform[..., a:b][..., :MAX]
        total = int(uniform(speech.shape[-1], MAX)); slack = total - speech.shape[-1]
        lead = int(uniform(0, min(SR, slack))); trail = slack - lead
        floor = 10.0 ** (uniform(-90, -55) / 20)
        out = torch.cat((torch.randn(1, lead) * floor, speech, torch.randn(1, trail) * floor), dim=-1)
        if float(torch.rand(())) < 0.3:
            out = out * (uniform(0.03, 0.9) / out.abs().max().clamp_min(1e-5))
        return out.clamp(-1.0, 1.0)


def spec_augment(x: Tensor, lengths: Tensor, freq_masks=2, freq_width=8, time_masks=2, time_frac=0.1) -> Tensor:
    """Per-utterance frequency/time masks inside the valid length; masked cells -> 0 (the utterance mean)."""
    b, _, n_mels, t = x.shape; dev = x.device
    keep = torch.ones(b, n_mels, t, dtype=torch.bool, device=dev)
    mel = torch.arange(n_mels, device=dev)[None]; frames = torch.arange(t, device=dev)[None]
    for _ in range(freq_masks):
        w = torch.randint(0, freq_width + 1, (b, 1), device=dev)
        s = (torch.rand(b, 1, device=dev) * (n_mels - w + 1)).floor()
        keep &= ~((mel >= s) & (mel < s + w))[:, :, None]
    for _ in range(time_masks):
        w = (torch.rand(b, 1, device=dev) * (lengths[:, None] * time_frac).clamp(max=40)).floor()
        s = (torch.rand(b, 1, device=dev) * (lengths[:, None] - w + 1).clamp_min(1)).floor()
        keep &= ~((frames >= s) & (frames < s + w))[:, None, :]
    return x * keep[:, None].to(x.dtype)
