#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
DETR segmentation heads & losses — TT-Sim skeleton.

This file mirrors the structure of the upstream DETR `models/segmentation.py`
module, but replaces all PyTorch usage with TT-Sim-friendly stubs.

Baseline symbols preserved (same names / rough order):

  - class DETRsegm
  - def _expand
  - class MaskHeadSmallConv
  - class MHAttentionMap
  - def dice_loss
  - def sigmoid_focal_loss
  - class PostProcessSegm
  - class PostProcessPanoptic

Implementation notes:
  * No PyTorch tensors or nn.Module.
  * No real convolution / attention math yet — these will be added later
    using ttsim.front.functional (F) blocks, once the DETR core is fully
    validated.
  * For now, everything is wired as minimal SimNN.Module shells so reviewers
    can check strict 1:1 symbol mapping vs. the baseline.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np

import ttsim.front.functional.sim_nn as SimNN
import ttsim.front.functional.op as F  # TT-Sim functional ops

if TYPE_CHECKING:
    from ..util.misc import NestedTensor
else:
    from workloads.DETR.util.misc import NestedTensor



# ---------------------------------------------------------------------------
#  DETRsegm
# ---------------------------------------------------------------------------

class DETRsegm(SimNN.Module):
    """
    TT-Sim skeleton for the baseline `DETRsegm` wrapper.

    In the original DETR code this:
      * wraps a DETR detector,
      * adds MHAttentionMap + MaskHeadSmallConv,
      * returns dict with {pred_logits, pred_boxes, pred_masks, aux_outputs}.

    Here we only keep:
      * constructor signature,
      * attribute names,
      * forward() shape-level contract (returns a dict).
    """

    def __init__(self, detr: Any, freeze_detr: bool = False):
        super().__init__()
        self.detr = detr
        self.freeze_detr = bool(freeze_detr)

        # Baseline uses detr.transformer.d_model, detr.transformer.nhead.
        # We only keep them as metadata for now.
        self.hidden_dim = int(getattr(getattr(detr, "transformer", detr), "d_model", getattr(detr, "hidden_dim", 256)))
        self.nheads = int(getattr(getattr(detr, "transformer", detr), "nheads", 8))

        # Placeholders for segmentation heads (no math yet)
        self.bbox_attention = MHAttentionMap(
            name="detr.segm.bbox_attention",
            query_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            num_heads=self.nheads,
            dropout=0.0,
        )
        self._submodules[self.bbox_attention.name] = self.bbox_attention

        self.mask_head = MaskHeadSmallConv(
            name="detr.segm.mask_head",
            dim=self.hidden_dim + self.nheads,
            fpn_dims=[1024, 512, 256],
            context_dim=self.hidden_dim,
        )
        self._submodules[self.mask_head.name] = self.mask_head

        self.training = False
        super().link_op2module()

    def set_batch_size(self, new_bs: int):
        # pass-through to underlying DETR for consistency
        if hasattr(self.detr, "set_batch_size"):
            self.detr.set_batch_size(new_bs)

    def create_input_tensors(self):
        # Reuse DETR input creation (NestedTensor is handled upstream)
        if hasattr(self.detr, "create_input_tensors"):
            self.detr.create_input_tensors()
        self.input_tensors = getattr(self.detr, "input_tensors", {})

    def __call__(self, samples: Optional[NestedTensor] = None) -> Dict[str, Any]:
        """
        Baseline forward(samples: NestedTensor) → dict with logits, boxes, masks.

        For now we simply:
          * delegate to DETR.__call__ to get logits/boxes,
          * stub pred_masks as a TT-Sim tensor of zeros with compatible shape.
        """
        # Run baseline DETR path
        outputs = self.detr()

        # Expect baseline DETR dict-like output
        pred_logits = outputs.get("pred_logits")
        pred_boxes = outputs.get("pred_boxes")

        out: Dict[str, Any] = {
            "pred_logits": pred_logits,
            "pred_boxes": pred_boxes,
        }

        # If DETR provides aux_outputs, preserve them
        if "aux_outputs" in outputs:
            out["aux_outputs"] = outputs["aux_outputs"]

        # Stub mask logits: [B, num_queries, Hm, Wm]
        # Here we only ensure the *shape* exists, not real features.
        if pred_logits is not None:
            B = pred_logits.shape[0]
            Q = pred_logits.shape[1]
        else:
            # Extremely defensive default
            B = getattr(self.detr, "bs", 1)
            Q = getattr(self.detr, "num_queries", 100)

        # Choose a small, fixed mask resolution for now
        Hm, Wm = 64, 64
        mask_vals = np.zeros((B, Q, Hm, Wm), dtype=np.float32)
        mask_tensor = F._from_data("detr.segm.stub_masks", mask_vals)
        self._tensors[mask_tensor.name] = mask_tensor

        out["pred_masks"] = mask_tensor
        return out

    def analytical_param_count(self, lvl: int = 0) -> int:
        """
        Approximate param count = DETR + bbox_attention + mask_head (if implemented).
        For now we only delegate to DETR to keep things simple.
        """
        cnt = 0
        if hasattr(self.detr, "analytical_param_count"):
            cnt += self.detr.analytical_param_count(lvl + 1)
        return cnt


# ---------------------------------------------------------------------------
#  Helper: _expand  (same helper name as in baseline)
# ---------------------------------------------------------------------------

def _expand(tensor, length: int):
    """
    Baseline helper for repeating [B,...] → [B*length,...].

    In TT-Sim we will later re-express this via proper reshape + tile ops.
    For now this is not used in the execution path.
    """
    raise NotImplementedError(
        "_expand is a structural helper only. "
        "It will be implemented with TT-Sim ops when segmentation math is added."
    )


# ---------------------------------------------------------------------------
#  MaskHeadSmallConv
# ---------------------------------------------------------------------------

class MaskHeadSmallConv(SimNN.Module):
    """
    Skeleton of baseline MaskHeadSmallConv.

    Baseline behavior:
      * small conv tower with GroupNorm,
      * FPN-style upsampling with adapters,
      * produces 1-channel mask logits per query.

    Here it's a stub that keeps the constructor & call signature only.
    """

    def __init__(self, name: str, dim: int, fpn_dims: List[int], context_dim: int):
        super().__init__()
        self.name = name
        self.dim = int(dim)
        self.fpn_dims = [int(x) for x in fpn_dims]
        self.context_dim = int(context_dim)

        # TODO(ttsim): when we wire real conv/FPN blocks, they will be added here
        # as SimNN submodules (Conv2d, GroupNorm, etc.).

        self.training = False
        super().link_op2module()

    def __call__(self, x, bbox_mask, fpns: List[Any]):
        """
        Baseline forward:
          x:         [B, C, H, W] backbone feature map
          bbox_mask: attention weights
          fpns:      list of 3 FPN feature maps

        For TT-Sim skeleton we simply return a passthrough of `x` so shape
        plumbing remains trivial. Higher-level DETR code reshapes this output.
        """
        return x

    def analytical_param_count(self, lvl: int = 0) -> int:
        # No internal ops yet → 0 parameters from this block
        return 0


# ---------------------------------------------------------------------------
#  MHAttentionMap
# ---------------------------------------------------------------------------

class MHAttentionMap(SimNN.Module):
    """
    Skeleton of baseline MHAttentionMap.

    Baseline behavior:
      * 2D attention that returns only attention weights (no V multiplication).
      * q_linear / k_linear projections, softmax over HW.

    For now we only keep metadata and return the `k` tensor as a placeholder.
    """

    def __init__(
        self,
        name: str,
        query_dim: int,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.name = name
        self.query_dim = int(query_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.dropout = float(dropout)
        self.bias = bool(bias)

        # Real linear projections will be added later as TT-Sim Linear blocks.

        self.training = False
        super().link_op2module()

    def __call__(self, q, k, mask=None):
        """
        Baseline forward(q, k, mask) → attention weights [B,Q,H,W].

        We currently return `k` unchanged as a stub.
        """
        return k

    def analytical_param_count(self, lvl: int = 0) -> int:
        # No internal Linear blocks yet → 0 for now.
        return 0


# ---------------------------------------------------------------------------
#  Loss stubs: dice_loss, sigmoid_focal_loss
# ---------------------------------------------------------------------------

def dice_loss(inputs, targets, num_boxes: float):
    """
    Skeleton of baseline `dice_loss`.

    Original:
      * computes DICE loss for binary masks.

    TT-Sim DETR port does not use mask losses yet, so this returns 0.0 as
    a TT-Sim constant tensor but preserves the callable interface.
    """
    val = F._from_data("detr.segm.dice_loss_zero", np.array(0.0, dtype=np.float32))
    return val


def sigmoid_focal_loss(inputs, targets, num_boxes: float, alpha: float = 0.25, gamma: float = 2.0):
    """
    Skeleton of baseline `sigmoid_focal_loss`.

    Original:
      * focal loss on mask logits.

    Here we return 0.0 as a TT-Sim constant while keeping the signature.
    """
    val = F._from_data(
        "detr.segm.focal_loss_zero",
        np.array(0.0, dtype=np.float32),
    )
    return val


# ---------------------------------------------------------------------------
#  Post-processing stubs: PostProcessSegm, PostProcessPanoptic
# ---------------------------------------------------------------------------

class PostProcessSegm(SimNN.Module):
    """
    Skeleton for baseline PostProcessSegm.

    In DETR this:
      * interpolates masks to original image size,
      * converts to expected output format.

    For TT-Sim we keep it as a no-op wrapper over the DETR-style dict.
    """

    def __init__(self, name: str = "detr.segm.postprocess"):
        super().__init__()
        self.name = name
        self.training = False
        super().link_op2module()

    def __call__(self, results: List[Dict[str, Any]], outputs: Dict[str, Any], orig_sizes):
        # No-op: just forward results unchanged
        return results

    def analytical_param_count(self, lvl: int = 0) -> int:
        return 0


class PostProcessPanoptic(SimNN.Module):
    """
    Skeleton for baseline PostProcessPanoptic.

    Baseline:
      * converts DETR outputs into COCO-style panoptic maps,
      * uses panopticapi.id2rgb/rgb2id utilities.

    Here: structural placeholder only, no real panoptic conversion.
    """

    def __init__(self, name: str = "detr.segm.postprocess_panoptic"):
        super().__init__()
        self.name = name
        self.training = False
        super().link_op2module()

    def __call__(self, outputs: Dict[str, Any], target_sizes: Optional[Any] = None):
        # No-op placeholder; upstream will ignore this until panoptic is wired.
        return outputs

    def analytical_param_count(self, lvl: int = 0) -> int:
        return 0
