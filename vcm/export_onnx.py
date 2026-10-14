"""Export a trained checkpoint to ONNX (opset 17) and check ONNX Runtime against PyTorch.

Intent (Conformer): batch-1 graph with a dynamic number of frames, as used on the Raspberry Pi. Two Python-int length
steps of the training forward (the max-length slice and torchaudio's padding mask) would be frozen by tracing; with a
single unpadded clip both are no-ops, so the export uses an equivalent batch-1 forward. Attention is re-written with
shape-agnostic reshapes (same weights and maths). Checked on 6 lengths: max |logit diff| < 1e-3 and same argmax.
Slot (TC-DSCNN): dynamic batch and time.

Usage: python -m vcm.export_onnx intent RUN_DIR      -> RUN_DIR/intent.onnx + onnx_export.json
       python -m vcm.export_onnx slot   RUN_DIR      -> RUN_DIR/slot.onnx   + onnx_export.json
"""
from __future__ import annotations

import json, sys, types
from pathlib import Path

import numpy as np
import torch

from .common import sha256_file
from .models import SlotTCDSCNN, half_lengths, load_intent, temporal_mask


class TraceableSelfAttention(torch.nn.Module):
    """Eval-mode nn.MultiheadAttention self-attention without a mask, with a dynamic sequence length."""

    def __init__(self, mha):
        super().__init__()
        self.mha, self.heads = mha, mha.num_heads

    def forward(self, query, key, value, key_padding_mask=None, need_weights=False):
        t, b, d = query.shape
        q, k, v = torch.nn.functional.linear(query, self.mha.in_proj_weight, self.mha.in_proj_bias).chunk(3, dim=-1)
        split = lambda z: z.reshape(-1, b * self.heads, d // self.heads).transpose(0, 1)  # [B*H,T,Dh]
        out = torch.nn.functional.scaled_dot_product_attention(split(q), split(k), split(v))
        return self.mha.out_proj(out.transpose(0, 1).reshape(-1, b, d)), None


def conformer_batch1_forward(self, features, frame_lengths):
    x = temporal_mask(features, frame_lengths)
    l1 = half_lengths(frame_lengths); x = temporal_mask(self.conv1(x), l1)
    l2 = half_lengths(l1); x = temporal_mask(self.conv2(x), l2)
    b, c, f, t = x.shape
    x = self.project(x.permute(0, 3, 1, 2).reshape(b, t, c * f))
    self._valid = torch.ones_like(x[:, :1, :1]).expand(-1, 1, x.shape[1])  # [B,1,T], all frames valid
    x = x.transpose(0, 1)
    for layer in self.encoder.conformer_layers:
        x = layer(x, None)
    return self.classifier(self.dropout(self.pool(x.permute(1, 2, 0), l2)))


def main():
    kind, run = sys.argv[1], Path(sys.argv[2])
    assert kind in ("intent", "slot"), kind
    onnx_path, status_path = run / f"{kind}.onnx", run / "onnx_export.json"
    if onnx_path.exists():
        raise FileExistsError(f"Refusing to overwrite {onnx_path}")
    ck = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    cfg = ck["model_config"]
    if kind == "intent":
        reference, model = load_intent(ck), load_intent(ck)
        model.forward = types.MethodType(conformer_batch1_forward, model)
        for layer in model.encoder.conformer_layers:
            layer.self_attn = TraceableSelfAttention(layer.self_attn)
        x, lengths = torch.randn(1, 1, 64, 301), torch.tensor([301])
    else:
        model = SlotTCDSCNN(cfg["num_classes"], cfg["dropout"], cfg["width"]); model.load_state_dict(ck["model_state"])
        reference = model = model.eval()
        x, lengths = torch.randn(2, 1, 64, 301), torch.tensor([301, 211])
    torch.onnx.export(model, (x, lengths), onnx_path, input_names=["log_mel", "frame_lengths"], output_names=["logits"],
                      opset_version=17, dynamo=False,
                      dynamic_axes={"log_mel": {0: "batch", 3: "time"}, "frame_lengths": {0: "batch"}, "logits": {0: "batch"}})
    import onnxruntime as ort
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    cases = []
    for frames in (17, 41, 121, 173, 302, 501):
        x = torch.randn(1, 1, 64, frames); lens = torch.tensor([frames])
        with torch.inference_mode():
            expected = reference(x, lens).numpy()
        actual = session.run(["logits"], {"log_mel": x.numpy(), "frame_lengths": lens.numpy()})[0]
        cases.append({"frames": frames, "max_abs_logit_diff": float(np.abs(expected - actual).max()),
                      "same_argmax": bool((expected.argmax(1) == actual.argmax(1)).all())})
    status = {"kind": kind, "opset": 17, "onnx_file": onnx_path.name, "onnx_bytes": onnx_path.stat().st_size,
              "onnx_sha256": sha256_file(onnx_path), "checkpoint_sha256": sha256_file(run / "best.pt"),
              "batch": "exactly 1" if kind == "intent" else "dynamic",
              "inputs": {"log_mel": "float32 [B,1,64,T]", "frame_lengths": "int64 [B]"}, "output": f"logits [B,{cfg['num_classes']}]",
              "onnxruntime_version": ort.__version__, "pytorch_vs_onnxruntime": cases}
    assert all(c["same_argmax"] and c["max_abs_logit_diff"] < 1e-3 for c in cases), cases
    status_path.write_text(json.dumps(status, indent=2) + "\n"); print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
