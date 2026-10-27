"""RGB adaptation of upstream HSFormer; shared restoration loss, not paper reproduction."""
import importlib
import sys
from pathlib import Path
import torch
from torch import nn


class HSFormerAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        root = Path(__file__).resolve().parents[1] / "third_party"
        if not (root / "hsformer_runtime" / "HSFormer.py").exists():
            raise RuntimeError("Run python scripts/prepare_hsformer.py first.")
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        module = importlib.import_module("hsformer_runtime.HSFormer")
        self.net = module.Transformer(img_size=(256, 256), in_channels=3, out_cahnnels=3)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1:] != (3, 256, 256):
            raise ValueError("HSFormer pilot supports BCHW RGB 256x256 patches only")
        outputs, variances = self.net(x)
        raw = outputs[0]
        if raw.shape != x.shape:
            raise RuntimeError("Unexpected HSFormer output dimensions")
        # Variances are not exposed as calibrated confidence. The shared objective
        # uses the final restoration; upstream multiscale/NLL losses are not reproduced.
        return {"restored": raw.clamp(0, 1), "reconstruction": raw,
                "residual": raw-x}
