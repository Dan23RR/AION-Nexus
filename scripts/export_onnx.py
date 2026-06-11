"""Export AION-NEXUS to ONNX for edge / cross-runtime deployment.

Run: python -m scripts.export_onnx --checkpoint checkpoints/aion_nexus_v1.pth \
                                    --out checkpoints/aion_nexus.onnx
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from aion_nexus import NUM_CHANNELS, SIGNAL_LENGTH, InferenceEngine


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="checkpoints/aion_nexus.onnx")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--dynamic-batch", action="store_true",
                        help="Allow dynamic batch dimension at inference time")
    args = parser.parse_args()

    engine = InferenceEngine.from_checkpoint(args.checkpoint)
    model = engine.model
    model.eval()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    dummy = torch.randn(1, NUM_CHANNELS, SIGNAL_LENGTH)
    dynamic_axes = {"input": {0: "batch"}, "logits": {0: "batch"}, "features": {0: "batch"}} \
        if args.dynamic_batch else None

    # Wrap to flatten the dict output: ONNX Runtime expects tensor(s)
    class _OnnxWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x):
            o = self.m(x)
            return o["logits"], o["features"]

    wrapped = _OnnxWrapper(model)

    torch.onnx.export(
        wrapped,
        dummy,
        out,
        input_names=["input"],
        output_names=["logits", "features"],
        opset_version=args.opset,
        dynamic_axes=dynamic_axes,
    )

    size_mb = out.stat().st_size / 1024 / 1024
    print(f"Exported ONNX: {out} ({size_mb:.2f} MB)")
    print(f"  opset:         {args.opset}")
    print(f"  dynamic_batch: {args.dynamic_batch}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
