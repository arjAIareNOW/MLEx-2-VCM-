"""Front end shared by the intent and slot models (docs/02_preprocessing.md).

16 kHz mono PCM16 -> first 5 s (80,000 samples) -> [optional training augmentation] -> pad to >= 400 samples
-> power mel spectrogram (n_fft 512, window 400 = 25 ms Hann, hop 160 = 10 ms, 64 mels, 20-8000 Hz)
-> dB (top_db 80) -> per-utterance mean/std normalisation -> float32 [1, 64, T], T <= 501.
Batches are right-padded to the longest clip; the true frame counts travel with the batch so the
models can mask padding (batched and single-clip inference give the same logits).
deployment/vcm_engine/features.py is a NumPy copy of this front end used on the Raspberry Pi.
"""
from __future__ import annotations

import io, wave
from typing import Sequence

import numpy as np
import torch
import torchaudio
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence

SAMPLE_RATE, MAX_SAMPLES, WIN_LENGTH = 16_000, 80_000, 400


def decode_wav(audio: bytes) -> Tensor:
    """WAV bytes (16 kHz mono PCM16) -> float32 [1, N] in [-1, 1]."""
    with wave.open(io.BytesIO(audio), "rb") as handle:
        if handle.getframerate() != SAMPLE_RATE or handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise ValueError("Expected mono 16-kHz PCM16 WAV")
        pcm = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2").copy()
    return torch.from_numpy(pcm).float().div_(32768.0).unsqueeze(0)


class LogMel:
    def __init__(self) -> None:
        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=512, win_length=400, hop_length=160,
                                                        f_min=20.0, f_max=8000.0, n_mels=64, power=2.0)
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80.0)

    def __call__(self, waveform: Tensor) -> Tensor:
        if waveform.shape[-1] < WIN_LENGTH:
            waveform = nn.functional.pad(waveform, (0, WIN_LENGTH - waveform.shape[-1]))
        features = self.to_db(self.mel(waveform))
        return (features - features.mean()) / features.std().clamp_min(1e-5)


def collate(batch: Sequence[tuple[Tensor, int]]) -> tuple[Tensor, Tensor, Tensor]:
    """[(features [1,64,T_i], label)] -> features [B,1,64,T_max], frame_lengths [B], labels [B]."""
    features, labels = zip(*batch)
    lengths = torch.tensor([f.shape[-1] for f in features], dtype=torch.long)
    padded = pad_sequence([f.squeeze(0).transpose(0, 1) for f in features], batch_first=True).transpose(1, 2).unsqueeze(1)
    return padded, lengths, torch.tensor(labels, dtype=torch.long)
