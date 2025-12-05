#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Polaris/TTSIM port of facebookresearch/detr/util/box_ops.py

Strict PUBLIC API mapping:

    - box_cxcywh_to_xyxy
    - box_xyxy_to_cxcywh
    - generalized_box_iou
    - box_iou

Differences from baseline:
    - No PyTorch
    - All numeric operations are stubbed / unimplemented
    - Pure structural mapping for compatibility with DETR model files
"""

from typing import Any, Tuple
import numpy as np


# ---------------------------------------------------------------------------
# box_cxcywh_to_xyxy
# ---------------------------------------------------------------------------

def box_cxcywh_to_xyxy(x: Any) -> Any:
    """
    Baseline behavior:

        cx, cy, w, h = x.unbind(-1)
        b = [
            cx - 0.5 * w,
            cy - 0.5 * h,
            cx + 0.5 * w,
            cy + 0.5 * h,
        ]

    Returns [x_min, y_min, x_max, y_max].

    Polaris/TTSIM skeleton:
        Not implemented.
    """
    raise NotImplementedError(
        "box_cxcywh_to_xyxy is not implemented in the TTSIM DETR skeleton."
    )


# ---------------------------------------------------------------------------
# box_xyxy_to_cxcywh
# ---------------------------------------------------------------------------

def box_xyxy_to_cxcywh(x: Any) -> Any:
    """
    Baseline behavior:

        x0, y0, x1, y1 = x.unbind(-1)
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        w  = (x1 - x0)
        h  = (y1 - y0)

    Returns [cx, cy, w, h].

    Polaris/TTSIM skeleton:
        Not implemented.
    """
    raise NotImplementedError(
        "box_xyxy_to_cxcywh is not implemented in the TTSIM DETR skeleton."
    )


# ---------------------------------------------------------------------------
# box_iou
# ---------------------------------------------------------------------------

def box_iou(boxes1: Any, boxes2: Any) -> Tuple[Any, Any]:
    """
    Baseline behavior:

        # Computes IoU and union area for each pair of boxes1 x boxes2

    Returns:
        (iou, union)

    Polaris/TTSIM skeleton:
        Not implemented.
    """
    raise NotImplementedError(
        "box_iou is not implemented in the TTSIM DETR skeleton."
    )


# ---------------------------------------------------------------------------
# generalized_box_iou
# ---------------------------------------------------------------------------

def generalized_box_iou(boxes1: Any, boxes2: Any) -> Any:
    """
    Baseline behavior:

        # Computes GIoU between boxes1 and boxes2
        # using enclosing box area

    Polaris/TTSIM skeleton:
        Not implemented.
    """
    raise NotImplementedError(
        "generalized_box_iou is not implemented in the TTSIM DETR skeleton."
    )

