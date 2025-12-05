#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Polaris/TTSIM port of facebookresearch/detr/models/__init__.py

The baseline file simply imports and exposes model components.
We maintain strict 1:1 structure with no added APIs.
"""

from .detr import DETR, SetCriterion, PostProcess, MLP, build
from .backbone import build_backbone
from .transformer import build_transformer
from .matcher import HungarianMatcher, build_matcher
from .position_encoding import PositionEmbeddingSine, PositionEmbeddingLearned, build_position_encoding
from .segmentation import (
    DETRsegm,
    MHAttentionMap,
    MaskHeadSmallConv,
    dice_loss,
    sigmoid_focal_loss,
    PostProcessSegm,
    PostProcessPanoptic,
)

