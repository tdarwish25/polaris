#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Polaris/TTSIM skeleton for facebookresearch/detr/models/backbone.py

PUBLIC API mapping (same symbols as baseline):

  - class FrozenBatchNorm2d
  - class BackboneBase
  - class Backbone
  - class Joiner
  - def build_backbone(cfg)

Differences vs baseline:
  - No PyTorch / torchvision / ResNet
  - No real feature extraction logic
  - Used for structural parity and TTSIM integration only
"""

from typing import Any, Dict, List, Tuple, TYPE_CHECKING

import ttsim.front.functional.sim_nn as SimNN

if TYPE_CHECKING:
    from .position_encoding import build_position_encoding
    from ..util.misc import NestedTensor
else:
    from workloads.DETR.models.position_encoding import build_position_encoding
    from workloads.DETR.util.misc import NestedTensor


# -----------------------------------------------------------------------------
# FrozenBatchNorm2d
# -----------------------------------------------------------------------------

class FrozenBatchNorm2d(SimNN.Module):
    """
    Baseline:

      - A BatchNorm2d with fixed parameters (no running stats update).

    TTSIM skeleton:

      - Kept only for API / name parity.
      - __call__ returns input unchanged (no-op normalization).
    """

    def __init__(self, num_features: int):
        super().__init__()
        self.num_features = int(num_features)
        super().link_op2module()

    def __call__(self, x: Any) -> Any:
        # No-op normalization: pass tensor through unchanged
        return x


# -----------------------------------------------------------------------------
# BackboneBase
# -----------------------------------------------------------------------------

class BackboneBase(SimNN.Module):
    """
    Baseline BackboneBase(nn.Module) wraps a CNN and exposes:

      - self.body          (the CNN)
      - self.num_channels  (output feature channels)
      - forward(NestedTensor) -> Dict[str, NestedTensor]

    TTSIM skeleton:

      - We keep num_channels and the call signature.
      - No internal CNN is implemented yet.
    """

    def __init__(self, num_channels: int):
        super().__init__()
        self.num_channels = int(num_channels)
        super().link_op2module()

    def __call__(self, tensor_list: NestedTensor) -> List[NestedTensor]:
        """
        Baseline returns a list/Dict of NestedTensor feature maps.

        Minimal skeleton:

          - Not used by the current DETR minimal forward.
          - Left unimplemented for now.
        """
        raise NotImplementedError(
            "BackboneBase.__call__ is not implemented in the minimal TTSIM DETR port."
        )


# -----------------------------------------------------------------------------
# Backbone
# -----------------------------------------------------------------------------

class Backbone(BackboneBase):
    """
    Polaris/TTSIM version of DETR Backbone.

    Baseline signature conceptually depends on:
      - backbone name (e.g., 'resnet50', 'resnet101')
      - train_backbone (bool)
      - return_interm_layers (bool)
      - dilation (bool)

    In this minimal TTSIM port:

      - We derive num_channels from cfg['hidden_dim'] for shape sanity.
      - We do not construct an actual ResNet.
      - We keep this class primarily to mirror baseline structure.
    """

    def __init__(self, cfg: Dict[str, Any]):
        hidden_dim = int(cfg.get("hidden_dim", 256))
        super().__init__(num_channels=hidden_dim)

        # Keep some flags for potential future extension
        self.backbone_name = cfg.get("backbone", "resnet50")
        self.dilation = bool(cfg.get("dilation", False))
        self.return_interm_layers = bool(cfg.get("return_interm_layers", False))

        # No actual CNN body is created here; this is a structural stub.


# -----------------------------------------------------------------------------
# Joiner: backbone + position encoding
# -----------------------------------------------------------------------------

class Joiner(SimNN.Module):
    """
    TTSIM version of DETR's Joiner, which in baseline composes:

      - a Backbone instance
      - a PositionEmbedding module

    Baseline Joiner(backbone, position_embedding) returns:

      - list of feature maps as NestedTensor
      - list of positional encodings (same spatial shapes)

    Minimal skeleton:

      - Keeps constructor and attributes.
      - __call__ is not implemented yet, as DETR minimal forward path
        currently fabricates outputs without consuming backbone features.
    """

    def __init__(self, backbone: Backbone, position_embedding: SimNN.Module):
        super().__init__()
        self.backbone = backbone
        self.position_embedding = position_embedding

        # Mirror baseline: expose num_channels from backbone
        self.num_channels = backbone.num_channels

        if isinstance(self.backbone, SimNN.Module):
            self._submodules["backbone"] = self.backbone
        if isinstance(self.position_embedding, SimNN.Module):
            self._submodules["position_embedding"] = self.position_embedding

        super().link_op2module()

    def __call__(self, tensor_list: NestedTensor) -> Tuple[List[NestedTensor], List[Any]]:
        """
        Baseline behavior (for reference):

            xs = self.backbone(tensor_list)
            out = []
            pos = []
            for name, x in xs.items():
                out.append(x)
                pos.append(self.position_embedding(x).to(x.tensors.dtype))
            return out, pos

        Minimal skeleton:

          - Not used in the current minimal DETR forward.
          - Left as unimplemented stub for now.
        """
        raise NotImplementedError(
            "Joiner.__call__ is not implemented in the minimal TTSIM DETR port."
        )


# -----------------------------------------------------------------------------
# build_backbone
# -----------------------------------------------------------------------------

def build_backbone(cfg: Dict[str, Any]) -> Joiner:
    """
    TTSIM port of baseline build_backbone(args).

    Baseline behavior:

        - Build a ResNet backbone (Backbone)
        - Build position encoding
        - Wrap them in a Joiner
        - Set model.num_channels = backbone.num_channels

    Minimal TTSIM version:

        - Creates a stub Backbone with num_channels tied to cfg['hidden_dim'].
        - Creates a real PositionEmbedding module via build_position_encoding.
        - Wraps both in a Joiner and returns it.
    """
    backbone = Backbone(cfg)
    position_embedding = build_position_encoding(cfg)
    model = Joiner(backbone, position_embedding)
    return model
