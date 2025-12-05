#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Polaris/TTSIM port of facebookresearch/detr/models/detr.py

PUBLIC API mapping, same order as baseline:

  - class DETR
  - class SetCriterion
  - class PostProcess
  - class MLP
  - def build(cfg)

Differences vs baseline:
  - No PyTorch
  - Backbone/transformer/criterion logic is skeletal
  - DETR.__call__ fabricates correctly-shaped outputs with minimal
    compute ops so Polaris/TTSIM can build graphs and gather stats.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# -----------------------------------------------------------------------------
# Make repo root importable so `import ttsim` works even when running
# `python detr.py` from workloads/DETR/models.
# -----------------------------------------------------------------------------

_THIS_FILE = Path(__file__).resolve()
# Layout: .../polaris/workloads/DETR/models/detr.py
# parents[0]=models, [1]=DETR, [2]=workloads, [3]=polaris
_REPO_ROOT = _THIS_FILE.parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import ttsim.front.functional.sim_nn as SimNN
import ttsim.front.functional.op as F

# -----------------------------------------------------------------------------
# Ensure workloads/DETR and workloads/DETR/models are importable
# -----------------------------------------------------------------------------

_THIS_DIR = _THIS_FILE.parent          # .../workloads/DETR/models
_WL_ROOT = _THIS_DIR.parent            # .../workloads/DETR

for p in (_WL_ROOT, _THIS_DIR):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

# Baseline-like imports (no relative dots so they work when loaded by file path)
from workloads.DETR.models.backbone import build_backbone
from workloads.DETR.models.transformer import build_transformer
from workloads.DETR.models.matcher import build_matcher
from workloads.DETR.models.segmentation import (
    DETRsegm,
    PostProcessSegm,
    PostProcessPanoptic,
)
from workloads.DETR.util.misc import NestedTensor
from workloads.DETR.util import box_ops

# -----------------------------------------------------------------------------
# 1. DETR main model (TTSIM workload entrypoint)
# -----------------------------------------------------------------------------

class DETR(SimNN.Module):
    """
    Polaris/TTSIM version of DETR model.

    Mirrors baseline DETR(nn.Module) public fields, but forward is a
    minimal shape-correct stub.

    For Polaris/TTSIM, this class is the only workload entrypoint and
    follows the same pattern as LeViT / EfficientViT_Cls:
      - __init__(name, cfg)
      - set_batch_size(...)
      - create_input_tensors(...)
      - __call__() -> forward
      - get_forward_graph()
      - analytical_param_count()
    """

    def __init__(self, name: str, cfg: Dict[str, Any]):
        super().__init__()
        self.name = name
        c = dict(cfg)

        # Image / batch config
        self.in_channels = int(c.get("img_channels", 3))
        self.in_height = int(c.get("img_height", 800))
        self.in_width = int(c.get("img_width", 800))
        self.bs = int(c.get("bs", 1))

        # DETR-specific config
        hidden_dim = int(c.get("hidden_dim", 256))
        num_queries = int(c.get("num_queries", 100))
        num_classes = int(c.get("num_classes", 91))
        nheads = int(c.get("nheads", 8))
        num_encoder_layers = int(c.get("num_encoder_layers", 6))
        num_decoder_layers = int(c.get("num_decoder_layers", 6))
        dim_feedforward = int(c.get("dim_feedforward", 2048))
        backbone_name = str(c.get("backbone", "resnet50"))

        # Store these for analytical_param_count
        self.hidden_dim = hidden_dim
        self.dim_feedforward = dim_feedforward
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.nheads = nheads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.backbone_name = backbone_name

        # Structurally instantiate backbone and transformer (stubs for now)
        self.backbone = build_backbone(c)
        self.transformer = build_transformer(c)
        if isinstance(self.backbone, SimNN.Module):
            self._submodules["backbone"] = self.backbone
        if isinstance(self.transformer, SimNN.Module):
            self._submodules["transformer"] = self.transformer

        # Simple compute ops so graph isn't "empty"
        self.mul_logits = F.Mul("detr.mul_logits"); self.mul_logits.set_module(self)
        self.mul_boxes  = F.Mul("detr.mul_boxes");  self.mul_boxes.set_module(self)

        self.training = False
        super().link_op2module()

    # ------------------- TTSIM helpers -------------------

    def set_batch_size(self, new_bs: int):
        self.bs = int(new_bs)

    def create_input_tensors(self):
        """
        Allocate input image tensor for Polaris/TTSIM.

        Shape: [bs, in_channels, in_height, in_width]
        """
        shp = [self.bs, self.in_channels, self.in_height, self.in_width]
        self.input_tensors = {
            "detr_input": F._from_shape("detr_input", shp)
        }

    # ------------------- forward / __call__ -------------------

    def __call__(self) -> Dict[str, Any]:
        """
        Minimal DETR forward for TTSIM.

        Baseline DETR forward(x) would:
          - wrap x in NestedTensor
          - run backbone + position encoding
          - run transformer encoder/decoder
          - apply class and bbox heads to final decoder layer
          - return:
              "pred_logits": [B, num_queries, num_classes+1]
              "pred_boxes":  [B, num_queries, 4]

        Here we:

          - Take x from self.input_tensors (LeViT-style).
          - Use cfg-based shapes to fabricate logits/boxes.
          - Run tiny Mul ops so graph has non-trivial compute.
        """
        x = self.input_tensors["detr_input"]
        B, C, H, W = x.shape

        Q = self.num_queries
        Ccls = self.num_classes + 1  # add "no-object" class

        # Constant logits and boxes with correct shapes
        logits_vals = np.zeros((B, Q, Ccls), dtype=np.float32)
        boxes_vals  = np.zeros((B, Q, 4),    dtype=np.float32)

        pred_logits = F._from_data("detr.pred_logits", logits_vals)
        self._tensors[pred_logits.name] = pred_logits   # register for graph
        pred_boxes  = F._from_data("detr.pred_boxes",  boxes_vals)
        self._tensors[pred_boxes.name] = pred_boxes     # register for graph

        # Attach simple compute ops so the graph isn't "empty"
        one_logits = F._from_data("detr.one_logits", np.array(1.0, dtype=np.float32))
        self._tensors[one_logits.name] = one_logits
        one_boxes = F._from_data("detr.one_boxes", np.array(1.0, dtype=np.float32))
        self._tensors[one_boxes.name] = one_boxes

        pred_logits = self.mul_logits(pred_logits, one_logits)
        pred_boxes  = self.mul_boxes(pred_boxes,  one_boxes)

        return {
            "pred_logits": pred_logits,
            "pred_boxes":  pred_boxes,
        }

    def get_forward_graph(self):
        """
        LeViT-style: delegate to SimNN base helper.
        """
        return super()._get_forward_graph(self.input_tensors)

    def analytical_param_count(self, lvl: int = 0) -> int:
        """
        Approximate analytical parameter count for DETR.

        We include:
          - Transformer encoder and decoder (attention + FFN + norms)
          - Query embeddings
          - Class and box heads
          - Approximate backbone param count (ResNet-50 / ResNet-101 constants)

        This is a static math estimate, no SimTensor / TT-Sim ops involved.
        """

        # Core transformer config
        d = int(getattr(self, "hidden_dim", 256))
        d_ff = int(getattr(self, "dim_feedforward", 2048))
        Lenc = int(getattr(self, "num_encoder_layers", 6))
        Ldec = int(getattr(self, "num_decoder_layers", 6))
        Nq = int(getattr(self, "num_queries", 100))
        num_classes = int(getattr(self, "num_classes", 91))

        # --- Transformer encoder layer params ---
        # Each encoder layer (very roughly):
        #   - Self-attention: 4 * d * d  (Q, K, V, out)
        #   - FFN: 2 * d * d_ff
        #   - Norms: 2 LayerNorms ~ 4 * d params (gamma + beta for 2 norms)
        enc_layer = 4 * d * d + 2 * d * d_ff + 4 * d
        enc_total = Lenc * enc_layer

        # --- Transformer decoder layer params ---
        # Each decoder layer (rough):
        #   - Self-attn:       4 * d * d
        #   - Cross-attn:      4 * d * d
        #   - FFN:             2 * d * d_ff
        #   - Norms: 3 LayerNorms ~ 6 * d
        dec_layer = 8 * d * d + 2 * d * d_ff + 6 * d
        dec_total = Ldec * dec_layer

        # --- Query embeddings ---
        # Learned object queries: [num_queries, d]
        query_emb = Nq * d

        # --- Prediction heads ---
        # Classification: Linear(d, num_classes+1)
        cls_out = num_classes + 1
        head_cls = d * cls_out + cls_out  # weights + bias

        # Box regression: Linear(d, 4)
        head_box = d * 4 + 4  # weights + bias

        transformer_and_heads = enc_total + dec_total + query_emb + head_cls + head_box

        # --- Backbone approximate param count ---
        # Use constants close to torchvision ResNet-50 / ResNet-101.
        backbone_lut = {
            "resnet50": 23508032,   # ~23.5M params
            "resnet101": 42500000,  # ~42.5M params
        }
        backbone_name = getattr(self, "backbone_name", None)
        backbone_name = getattr(self.backbone, "backbone_name", None)
        backbone_params = backbone_lut.get(backbone_name or "", 0)
        
        total = backbone_params + transformer_and_heads
        return int(total)


# -----------------------------------------------------------------------------
# 2. SetCriterion (training-only, stubbed)
# -----------------------------------------------------------------------------

class SetCriterion:
    """
    Baseline DETR SetCriterion(nn.Module) encapsulates loss computation.

    Here we keep the constructor arguments and call signature as a stub.
    Training/loss is out of scope for Polaris DETR workloads.
    """

    def __init__(
        self,
        num_classes: int,
        matcher: Any,
        weight_dict: Dict[str, float],
        eos_coef: float,
        losses: List[str],
    ):
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = dict(weight_dict)
        self.eos_coef = float(eos_coef)
        self.losses = list(losses)

    def __call__(self, outputs: Dict[str, Any], targets: List[Dict[str, Any]]) -> Dict[str, Any]:
        raise NotImplementedError(
            "SetCriterion is not implemented in the TTSIM DETR port (training-only component)."
        )


# -----------------------------------------------------------------------------
# 3. PostProcess (inference formatting, stubbed)
# -----------------------------------------------------------------------------

class PostProcess:
    """
    Baseline DETR PostProcess(nn.Module) converts raw model outputs into
    COCO-style results (rescaled boxes, scores, labels).

    Here we keep the call signature but do not implement the logic.
    """

    def __call__(self, outputs: Dict[str, Any], target_sizes: Any) -> List[Dict[str, Any]]:
        raise NotImplementedError(
            "PostProcess.__call__ is not implemented in the minimal TTSIM DETR port."
        )


# -----------------------------------------------------------------------------
# 4. MLP helper (unused stub)
# -----------------------------------------------------------------------------

class MLP:
    """
    Baseline DETR uses a small MLP for bbox regression.

    We keep the class and constructor for structural parity, but the
    minimal TTSIM port does not currently use it.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

    def __call__(self, x: Any) -> Any:
        raise NotImplementedError("MLP is not used in the minimal TTSIM DETR port.")


# -----------------------------------------------------------------------------
# 5. build(cfg) factory
# -----------------------------------------------------------------------------

def build(cfg: Dict[str, Any]) -> Tuple[DETR, Optional[SetCriterion], Dict[str, Any]]:
    """
    TTSIM version of DETR build(args).

    Baseline:

        def build(args):
            backbone = build_backbone(args)
            transformer = build_transformer(args)
            model = DETR(...)
            matcher = build_matcher(args)
            criterion = SetCriterion(...)
            postprocessors = {"bbox": PostProcess()}
            ...

            return model, criterion, postprocessors

    Here we:

      - Build the DETR model (TTSIM).
      - Construct stub matcher/criterion/postprocessors for structural parity.
    """
    model = DETR("detr", cfg)

    matcher = build_matcher(cfg)
    weight_dict: Dict[str, float] = {}
    eos_coef = float(cfg.get("eos_coef", 0.1))
    losses = ["labels", "boxes", "cardinality"]

    criterion = SetCriterion(
        num_classes=int(cfg.get("num_classes", 91)),
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=eos_coef,
        losses=losses,
    )
    postprocessors: Dict[str, Any] = {
        "bbox": PostProcess(),
    }

    # Panoptic / segm postprocessors (PostProcessSegm, PostProcessPanoptic)
    # can be attached later if needed.
    return model, criterion, postprocessors


# -----------------------------------------------------------------------------
# Smoke test (__main__)
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Simple DETR config similar to detr_r50 but with smaller image for quick test
    cfg = {
        "img_channels": 3,
        "img_height": 224,
        "img_width": 224,
        "bs": 1,
        "backbone": "resnet50",
        "dilation": False,
        "num_queries": 100,
        "hidden_dim": 256,
        "num_classes": 91,
        "nheads": 8,
        "num_encoder_layers": 6,
        "num_decoder_layers": 6,
        "dim_feedforward": 2048,
    }

    model = DETR("detr_smoke_test", cfg)
    model.set_batch_size(1)
    model.create_input_tensors()
    out = model()

    print("\n=== DETR Smoke Test Output Shapes ===")
    print("pred_logits:", out["pred_logits"].shape)
    print("pred_boxes :", out["pred_boxes"].shape)
    print("Analytical parameter count:", model.analytical_param_count())

    gg = model.get_forward_graph()
    gg.graph2onnx("detr_smoke_test.onnx", do_model_check=False)
    print("Exported ONNX to detr_smoke_test.onnx\n")

