#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Polaris/TTSIM skeleton for facebookresearch/detr/models/matcher.py

PUBLIC API mapping:

  - class HungarianMatcher
  - def build_matcher(cfg)

Differences vs baseline:
  - No PyTorch / torch.distributed
  - No real Hungarian matching (training-only)
  - __call__ is a stub; not used in minimal DETR TTSIM forward
"""

from typing import Any, Dict, List, Tuple, TYPE_CHECKING
import importlib
import sys
from pathlib import Path

# -----------------------------------------------------------------------------
# Ensure workloads/DETR and workloads/DETR/models are importable
# -----------------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent          # .../workloads/DETR/models
_WL_ROOT = _THIS_DIR.parent                          # .../workloads/DETR

for p in (_WL_ROOT, _THIS_DIR):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

# Non-relative import so it works when loaded via file path
if TYPE_CHECKING:
    from ..util import box_ops as _box_ops_mod
else:
    try:
        from workloads.DETR.util import box_ops as _box_ops_mod
    except ModuleNotFoundError:  # pragma: no cover - fallback for dynamic execution
        _box_ops_mod = importlib.import_module("workloads.DETR.util.box_ops")

box_ops = _box_ops_mod


# -----------------------------------------------------------------------------
# HungarianMatcher (stub)
# -----------------------------------------------------------------------------

class HungarianMatcher:
    """
    Baseline DETR uses HungarianMatcher(nn.Module) to solve the assignment
    between predicted boxes and ground-truth boxes using the Hungarian algo.

    Constructor signature in baseline:

        def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):

    TTSIM skeleton:

      - Same constructor arguments (for 1:1 mapping).
      - __call__ is a stub and raises NotImplementedError, since
        training / loss computation are out of scope for Polaris DETR.
    """

    def __init__(self, cost_class: float = 1.0, cost_bbox: float = 1.0, cost_giou: float = 1.0):
        self.cost_class = float(cost_class)
        self.cost_bbox = float(cost_bbox)
        self.cost_giou = float(cost_giou)

    def __call__(self, outputs: Dict[str, Any], targets: List[Dict[str, Any]]) -> List[Tuple[Any, Any]]:
        """
        Baseline returns a list of size batch_size, each element being a
        tuple (indices_pred, indices_target) with matched indices.

        TTSIM skeleton:

          - Not implemented; only present for structural parity with baseline.
        """
        raise NotImplementedError(
            "HungarianMatcher.__call__ is not implemented in the TTSIM DETR port (training-only component)."
        )


# -----------------------------------------------------------------------------
# build_matcher factory
# -----------------------------------------------------------------------------

def build_matcher(cfg: Dict[str, Any]) -> HungarianMatcher:
    """
    Polaris/TTSIM port of baseline build_matcher(args).

    Baseline logic (simplified):

        return HungarianMatcher(
            cost_class=args.set_cost_class,
            cost_bbox=args.set_cost_bbox,
            cost_giou=args.set_cost_giou,
        )

    Here we pull the same fields from cfg with sensible defaults.
    """
    cost_class = float(cfg.get("set_cost_class", 1.0))
    cost_bbox = float(cfg.get("set_cost_bbox", 1.0))
    cost_giou = float(cfg.get("set_cost_giou", 1.0))
    return HungarianMatcher(cost_class=cost_class, cost_bbox=cost_bbox, cost_giou=cost_giou)
