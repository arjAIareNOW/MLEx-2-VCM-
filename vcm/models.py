"""Networks (docs/04_intent_model.md, docs/05_slot_model.md). Both take log-mel [B,1,64,T] + frame_lengths [B]
and are padding-invariant: padded frames never change the logits.

IntentConformer  the selected 19-intent model (~644 k parameters at d_model 96)
  stem    : 2x [Conv2D 3x3 stride 2 - BatchNorm - SiLU] (32 ch) -> 4x time / 4x frequency subsampling, Linear(512, d)
  encoder : torchaudio Conformer, 4 layers, 4 heads, FFN 2d, depthwise conv kernel 15, dropout 0.1;
            key-padding-masked attention, padded frames zeroed before every depthwise conv
  pooling : length-masked attentive statistics (weighted mean + std) -> dropout 0.3 -> Linear(2d, 19)

SlotTCDSCNN      the slot-value model (18 outputs; only the 3 values of the predicted intent compete)
  temporal depthwise-separable CNN: stem Conv2D(1,64,3x3,s2) + 2 frequency-strided DS blocks + 6 dilated
  temporal blocks (1..32) with squeeze-excitation, attentive statistics pooling, Linear(256, 18)
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torchaudio.models import Conformer


def temporal_mask(features: Tensor, lengths: Tensor) -> Tensor:
    steps = torch.arange(features.shape[-1], device=features.device).unsqueeze(0)
    mask = (steps < lengths.unsqueeze(1)).unsqueeze(1).unsqueeze(2)
    return features * mask.to(features.dtype)


def half_lengths(lengths: Tensor) -> Tensor:  # Conv(kernel 3, stride 2, padding 1)
    return torch.div(lengths - 1, 2, rounding_mode="floor") + 1


class AttentiveStatsPool(nn.Module):
    """Length-masked attentive statistics pooling over time on [B,C,T] -> [B,2C]."""

    def __init__(self, channels: int, hidden: int = 64) -> None:
        super().__init__()
        self.attention = nn.Sequential(nn.Conv1d(channels, hidden, 1), nn.Tanh(), nn.Conv1d(hidden, channels, 1))

    def forward(self, x: Tensor, lengths: Tensor) -> Tensor:
        valid = torch.arange(x.shape[-1], device=x.device)[None, None, :] < lengths[:, None, None]
        weights = self.attention(x).float().masked_fill(~valid, -torch.inf).softmax(-1)
        x = x.float()
        mean = (weights * x).sum(-1)
        std = ((weights * x.square()).sum(-1) - mean.square()).clamp_min(1e-5).sqrt()
        return torch.cat((mean, std), dim=1)


# ---------------------------------------------------------------- intent model
LAYERS, HEADS, KERNEL, ENC_DROPOUT = 4, 4, 15, 0.1


class IntentConformer(nn.Module):
    def __init__(self, num_classes: int = 19, dropout: float = 0.30, width: int = 96) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 32, 3, 2, 1, bias=False), nn.BatchNorm2d(32), nn.SiLU())
        self.conv2 = nn.Sequential(nn.Conv2d(32, 32, 3, 2, 1, bias=False), nn.BatchNorm2d(32), nn.SiLU())
        self.project = nn.Linear(32 * 16, width)
        self.encoder = Conformer(input_dim=width, num_heads=HEADS, ffn_dim=2 * width, num_layers=LAYERS,
                                 depthwise_conv_kernel_size=KERNEL, dropout=ENC_DROPOUT)
        self.pool = AttentiveStatsPool(width)
        self.dropout = nn.Dropout(dropout); self.classifier = nn.Linear(2 * width, num_classes)
        # Zero padded frames right before each depthwise conv so batched == single-clip inference.
        self._valid: Tensor | None = None
        for layer in self.encoder.conformer_layers:
            layer.conv_module.sequential[2].register_forward_pre_hook(self._mask_depthwise_input)

    def _mask_depthwise_input(self, module: nn.Module, args: tuple[Tensor]) -> tuple[Tensor]:
        return (args[0] * self._valid.to(args[0].dtype),)

    def forward(self, features: Tensor, frame_lengths: Tensor) -> Tensor:
        if features.ndim != 4 or features.shape[1:3] != (1, 64):
            raise ValueError(f"Expected [B,1,64,T], got {tuple(features.shape)}")
        x = temporal_mask(features, frame_lengths)
        l1 = half_lengths(frame_lengths); x = temporal_mask(self.conv1(x), l1)
        l2 = half_lengths(l1); x = temporal_mask(self.conv2(x), l2)
        b, c, f, t = x.shape
        x = self.project(x.permute(0, 3, 1, 2).reshape(b, t, c * f))
        lengths = l2.clamp(max=t)
        x = x[:, :int(lengths.max())]  # torchaudio builds its padding mask with T = lengths.max()
        self._valid = (torch.arange(x.shape[1], device=x.device)[None] < lengths[:, None])[:, None]  # [B,1,T]
        x, _ = self.encoder(x, lengths)
        return self.classifier(self.dropout(self.pool(x.transpose(1, 2), lengths)))


def conformer_config(width: int = 96) -> dict[str, object]:
    return {"family": "Conformer (torchaudio.models.Conformer)", "recurrent": False, "causal": False,
            "stem": "2x [Conv2D(3x3,stride 2,pad 1)-BatchNorm-SiLU], 32 ch; 4x time/freq subsampling; Linear(512,d)",
            "d_model": width, "layers": LAYERS, "heads": HEADS, "ffn_dim": 2 * width, "conv_kernel": KERNEL,
            "encoder_dropout": ENC_DROPOUT, "pooling": "length-masked attentive statistics (mean+std)",
            "classifier": f"Linear({2 * width},19) raw logits"}


def load_intent(checkpoint: dict) -> IntentConformer:
    cfg = checkpoint["model_config"]
    model = IntentConformer(cfg["num_classes"], cfg["dropout"], cfg["width"])
    model.load_state_dict(checkpoint["model_state"]); return model.eval()


# ---------------------------------------------------------------- slot model
def conv_output_lengths(lengths: Tensor, kernel: int, stride: int, padding: int) -> Tensor:
    return torch.div(lengths + 2 * padding - kernel, stride, rounding_mode="floor") + 1


def masked_mean(x: Tensor, lengths: Tensor) -> Tensor:
    """Mean over valid (frequency, time) cells of [B,C,F,T] -> [B,C]."""
    return temporal_mask(x, lengths).sum((2, 3)) / (x.shape[2] * lengths).unsqueeze(1).to(x.dtype)


class DSBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, frequency_stride: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, 3, stride=(frequency_stride, 1), padding=1,
                                   groups=in_channels, bias=False)
        self.dn = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.pn = nn.BatchNorm2d(out_channels); self.activation = nn.ReLU(inplace=True)

    def forward(self, x: Tensor, lengths: Tensor) -> Tensor:
        x = temporal_mask(self.activation(self.dn(self.depthwise(x))), lengths)
        return temporal_mask(self.activation(self.pn(self.pointwise(x))), lengths)


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        self.reduce = nn.Linear(channels, channels // reduction); self.expand = nn.Linear(channels // reduction, channels)

    def forward(self, x: Tensor, lengths: Tensor) -> Tensor:
        return x * torch.sigmoid(self.expand(torch.relu(self.reduce(masked_mean(x, lengths)))))[:, :, None, None]


class SETemporalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__(); self.dilation = dilation
        self.depthwise = nn.Conv2d(channels, channels, (1, 3), padding=(0, dilation), dilation=(1, dilation),
                                   groups=channels, bias=False)
        self.dn = nn.BatchNorm2d(channels)
        self.pointwise = nn.Conv2d(channels, channels, 1, bias=False)
        self.pn = nn.BatchNorm2d(channels)
        self.se = SqueezeExcitation(channels)
        self.drop = nn.Dropout2d(dropout); self.activation = nn.ReLU(inplace=True)

    def forward(self, x: Tensor, lengths: Tensor) -> Tensor:
        residual = temporal_mask(x, lengths)
        x = temporal_mask(self.activation(self.dn(self.depthwise(residual))), lengths)
        x = temporal_mask(self.pn(self.pointwise(x)), lengths)
        x = self.drop(self.se(x, lengths))
        return temporal_mask(self.activation(x + residual), lengths)


class SlotTCDSCNN(nn.Module):
    dilations = (1, 2, 4, 8, 16, 32)

    def __init__(self, num_classes: int = 18, dropout: float = 0.30, width: int = 128, block_dropout: float = 0.05) -> None:
        super().__init__()
        self.stem = nn.Conv2d(1, 64, 3, stride=(2, 2), padding=1, bias=False)
        self.stem_norm = nn.BatchNorm2d(64); self.activation = nn.ReLU(inplace=True)
        self.feature_blocks = nn.ModuleList((DSBlock(64, width, 2), DSBlock(width, width, 2)))
        self.temporal_blocks = nn.ModuleList(SETemporalBlock(width, d, block_dropout) for d in self.dilations)
        self.pool = AttentiveStatsPool(width)
        self.dropout = nn.Dropout(dropout); self.classifier = nn.Linear(2 * width, num_classes)

    @staticmethod
    def output_lengths(frame_lengths: Tensor) -> Tensor:
        return conv_output_lengths(frame_lengths, 3, 2, 1)

    def forward(self, features: Tensor, frame_lengths: Tensor) -> Tensor:
        if features.ndim != 4 or features.shape[1:3] != (1, 64):
            raise ValueError(f"Expected [B,1,64,T], got {tuple(features.shape)}")
        features = temporal_mask(features, frame_lengths); lengths = self.output_lengths(frame_lengths)
        x = temporal_mask(self.activation(self.stem_norm(self.stem(features))), lengths)
        for block in self.feature_blocks: x = block(x, lengths)
        for block in self.temporal_blocks: x = block(x, lengths)
        return self.classifier(self.dropout(self.pool(x.mean(2), lengths)))
