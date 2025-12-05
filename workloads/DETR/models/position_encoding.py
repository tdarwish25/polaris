#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTSIM/SimNN port of facebookresearch/detr/models/position_encoding.py

Strict PUBLIC API mapping:

  - class PositionEmbeddingSine
  - class PositionEmbeddingLearned
  - def build_position_encoding(cfg)

Differences:
  - No PyTorch
  - PositionEmbeddingSine.__call__ implemented using NumPy + TTSIM constant
    tensors (no autograd).
  - PositionEmbeddingLearned remains a stub (not needed for DETR baseline configs).
"""

from typing import Any, Dict, Optional, TYPE_CHECKING
import sys
from pathlib import Path

import numpy as np
import ttsim.front.functional.sim_nn as SimNN
import ttsim.front.functional.op as F

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
    from ..util.misc import NestedTensor
else:
    from workloads.DETR.util.misc import NestedTensor


# -----------------------------------------------------------------------------
# PositionEmbeddingSine
# -----------------------------------------------------------------------------

class PositionEmbeddingSine(SimNN.Module):
    """
    Polaris/TTSIM version of DETR's PositionEmbeddingSine.

    Baseline signature:

        def __init__(self, num_pos_feats=64, temperature=10000,
                     normalize=False, scale=None)

    num_pos_feats: number of features per axis (x and y each)

    Final embedding has dimension 2 * num_pos_feats (y + x), matching
    DETR behavior: hidden_dim = 2 * num_pos_feats.
    """

    def __init__(
        self,
        num_pos_feats: int = 64,
        temperature: float = 10000.0,
        normalize: bool = False,
        scale: Optional[float] = None,
    ):
        super().__init__()
        self.num_pos_feats = int(num_pos_feats)
        self.temperature = float(temperature)
        self.normalize = bool(normalize)
        self.scale = scale

        super().link_op2module()

    def __call__(self, tensor_list: NestedTensor) -> Any:
        """
        Baseline behavior (PyTorch, summarized):

          - src = tensor_list.tensors    # [B, C, H, W]
          - mask = tensor_list.mask      # [B, H, W] (optional)
          - build coordinate grids x,y
          - optionally normalize to [0, scale]
          - compute sinusoidal embeddings using temperature and
            sin/cos interleaving
          - return pos of shape [B, 2*num_pos_feats, H, W]

        TTSIM implementation:

          - Uses NumPy to build the embedding (no torch).
          - Ignores mask for now (assumes all positions are valid).
          - Returns a TTSIM constant tensor via F._from_data(...) with
            name "pos_embed".
        """
        src = tensor_list.tensors
        if not isinstance(src, np.ndarray):
            raise TypeError(
                f"PositionEmbeddingSine expects NestedTensor.tensors as np.ndarray, "
                f"got {type(src)}"
            )

        B, C, H, W = src.shape

        # Coordinate grids [H, W]
        y_coords = np.arange(H, dtype=np.float32).reshape(H, 1)
        x_coords = np.arange(W, dtype=np.float32).reshape(1, W)
        y = np.broadcast_to(y_coords, (H, W))  # [H,W]
        x = np.broadcast_to(x_coords, (H, W))  # [H,W]

        if self.normalize:
            # Normalize to [0,1] and optionally scale by 2π (baseline)
            eps = 1e-6
            y = y / (max(H - 1, 1) + eps)
            x = x / (max(W - 1, 1) + eps)
            if self.scale is None:
                scale = 2.0 * np.pi
            else:
                scale = float(self.scale)
            y = y * scale
            x = x * scale

        # dim_t: [num_pos_feats] frequency denominators
        dim_t = self.temperature ** (
            2 * (np.arange(self.num_pos_feats, dtype=np.float32) // 2) / self.num_pos_feats
        )  # [F]

        # Compute embeddings for y and x: [H, W, num_pos_feats]
        pos_y = np.zeros((H, W, self.num_pos_feats), dtype=np.float32)
        pos_x = np.zeros((H, W, self.num_pos_feats), dtype=np.float32)

        for i in range(self.num_pos_feats):
            if i % 2 == 0:
                # even index → sin
                pos_y[..., i] = np.sin(y / dim_t[i])
                pos_x[..., i] = np.sin(x / dim_t[i])
            else:
                # odd index → cos
                pos_y[..., i] = np.cos(y / dim_t[i])
                pos_x[..., i] = np.cos(x / dim_t[i])

        # Concatenate y and x encodings: [H, W, 2 * num_pos_feats]
        pos_hw = np.concatenate([pos_y, pos_x], axis=-1)

        # Broadcast to batch and permute to [B, 2*num_pos_feats, H, W]
        pos_bhwc = np.broadcast_to(pos_hw[None, ...], (B, H, W, 2 * self.num_pos_feats))
        pos = np.transpose(pos_bhwc, (0, 3, 1, 2))

        # Wrap as a TTSIM constant tensor
        pos_tensor = F._from_data("pos_embed", pos)
        return pos_tensor


# -----------------------------------------------------------------------------
# PositionEmbeddingLearned
# -----------------------------------------------------------------------------

class PositionEmbeddingLearned(SimNN.Module):
    """
    Polaris/TTSIM version of DETR's PositionEmbeddingLearned.

    Baseline behavior:
      - A learned table of size [H_max * W_max, 2*num_pos_feats]
      - Index into this table for each spatial location

    Not implemented yet in TTSIM; kept only for API parity.
    """

    def __init__(self, num_pos_feats: int = 256):
        super().__init__()
        self.num_pos_feats = int(num_pos_feats)

        # Baseline: nn.Embedding(50*50, 2*num_pos_feats)
        # TTSIM: stub only

        super().link_op2module()

    def __call__(self, tensor_list: NestedTensor) -> Any:
        raise NotImplementedError(
            "PositionEmbeddingLearned.__call__ is not implemented in the TTSIM DETR skeleton."
        )


# -----------------------------------------------------------------------------
# Factory
# -----------------------------------------------------------------------------

def build_position_encoding(cfg: Dict[str, Any]) -> SimNN.Module:
    """
    Polaris/TTSIM version of DETR's build_position_encoding(args).

    Baseline logic:

        hidden_dim = args.hidden_dim
        N_steps = hidden_dim // 2

        if args.position_embedding in ('v2','sine'):
            return PositionEmbeddingSine(...)

        if args.position_embedding in ('v3','learned'):
            return PositionEmbeddingLearned(...)

    Code below maps cfg fields 1:1.
    """

    hidden_dim = int(cfg.get("hidden_dim", 256))
    n_steps = hidden_dim // 2

    pos_type = cfg.get("position_embedding", "sine")  # 'sine','v2','learned','v3'

    if pos_type in ("sine", "v2"):
        normalize = bool(cfg.get("pos_normalize", True))
        scale = cfg.get("pos_scale", None)
        return PositionEmbeddingSine(
            num_pos_feats=n_steps,
            temperature=float(cfg.get("pos_temperature", 10000.0)),
            normalize=normalize,
            scale=scale,
        )

    if pos_type in ("learned", "v3"):
        return PositionEmbeddingLearned(num_pos_feats=n_steps)

    # Fallback to sine (baseline does this)
    return PositionEmbeddingSine(num_pos_feats=n_steps)
