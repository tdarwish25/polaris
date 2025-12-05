#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTSIM/SimNN port of facebookresearch/detr/models/transformer.py

Strict PUBLIC API mapping (same names as baseline):

  - class Transformer
  - class TransformerEncoder
  - class TransformerDecoder
  - class TransformerEncoderLayer
  - class TransformerDecoderLayer
  - def _get_clones(module, N)
  - def build_transformer(cfg)
  - def _get_activation_fn(activation)

Differences:
  - No torch / torch.nn / torch.nn.functional.
  - Encoder/decoder layers are structural stubs.
  - Transformer.forward is implemented in a minimal way:
      * Produces correctly-shaped hs and memory tensors using TTSIM
        constant data, without full attention math.
"""

from typing import Any, Dict, List, Optional, Tuple, cast
import copy
import numpy as np

import ttsim.front.functional.sim_nn as SimNN
import ttsim.front.functional.op as F


# -----------------------------------------------------------------------------
# Transformer
# -----------------------------------------------------------------------------

class Transformer(SimNN.Module):
    """
    TTSIM port of baseline Transformer(nn.Module).

    Baseline signature:

        def __init__(
            self,
            d_model=512,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=2048,
            dropout=0.1,
            activation="relu",
            normalize_before=False,
            return_intermediate_dec=False,
        )

    We keep the same args and attribute names.

    Minimal implementation:

      - Encoder/decoder objects are constructed but NOT used.
      - forward() returns:
          hs:     [num_decoder_layers, B, num_queries, d_model]
          memory: [B, C, H, W]  (just src)
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        normalize_before: bool = False,
        return_intermediate_dec: bool = False,
    ):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            normalize_before,
        )
        encoder_norm = None  # nn.LayerNorm in baseline if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            normalize_before,
        )
        decoder_norm = None  # nn.LayerNorm in baseline
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
        )

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers

        if isinstance(self.encoder, SimNN.Module):
            self._submodules["encoder"] = self.encoder
        if isinstance(self.decoder, SimNN.Module):
            self._submodules["decoder"] = self.decoder

        super().link_op2module()

    def _reset_parameters(self):
        """
        Baseline:

            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

        TTSIM:
          - Parameter initialization is handled at tensor creation sites.
          - No-op here.
        """
        return

    def forward(
        self,
        src: Any,
        mask: Any,
        query_embed: Any,
        pos_embed: Any,
    ) -> Tuple[Any, Any]:
        """
        Minimal TTSIM implementation of baseline forward.

        Baseline (for reference):

            # src: [B, C, H, W]
            bs, c, h, w = src.shape
            src = src.flatten(2).permute(2, 0, 1)       # [HW, B, C]
            pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
            query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)  # [Q, B, C]
            mask = mask.flatten(1)                                   # [B, HW]

            tgt = torch.zeros_like(query_embed)                      # [Q, B, C]
            memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
            hs = self.decoder(
                tgt,
                memory,
                memory_key_padding_mask=mask,
                pos=pos_embed,
                query_pos=query_embed,
            )
            return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)

        Minimal TTSIM version:

          - Does NOT call encoder/decoder.
          - Uses only shapes to fabricate a constant hs tensor.
          - Returns:
              hs:     [num_decoder_layers, B, num_queries, d_model]
              memory: src  (assumed [B, C, H, W] with C == d_model)
        """
        # src expected from DETR.input_proj: [B, C, H, W]
        B, C, H, W = src.shape

        # query_embed was created in DETR as [num_queries, d_model]
        num_queries = query_embed.shape[0]

        # Create a constant hs tensor of zeros with correct DETR shape:
        #   [num_decoder_layers, B, num_queries, d_model]
        hs_vals = np.zeros(
            (self.num_decoder_layers, B, num_queries, self.d_model),
            dtype=np.float32,
        )
        hs = F._from_data("transformer.hs", hs_vals)

        # For memory, we simply return src as-is (baseline memory is
        # encoder output reshaped back to [B, C, H, W]).
        memory = src

        return hs, memory


# -----------------------------------------------------------------------------
# TransformerEncoder
# -----------------------------------------------------------------------------

class TransformerEncoder(SimNN.Module):
    """
    TTSIM port of baseline TransformerEncoder(nn.Module).

    Encoder layers are constructed but not used by the minimal
    Transformer.forward above. We still keep them for structural parity.
    """

    def __init__(self, encoder_layer: "TransformerEncoderLayer", num_layers: int, norm: Optional[Any] = None):
        super().__init__()
        self.layers: List[TransformerEncoderLayer] = cast(
            List[TransformerEncoderLayer],
            _get_clones(encoder_layer, num_layers),
        )
        self.num_layers = num_layers
        self.norm = norm  # placeholder for future LayerNorm TTSIM module

        for i, layer in enumerate(self.layers):
            if isinstance(layer, SimNN.Module):
                self._submodules[f"encoder_layer_{i}"] = layer

        super().link_op2module()

    def forward(
        self,
        src: Any,
        mask: Optional[Any] = None,
        src_key_padding_mask: Optional[Any] = None,
        pos: Optional[Any] = None,
    ) -> Any:
        """
        Baseline loops through layers; left unimplemented here as
        Transformer.forward does not call encoder in the minimal port.
        """
        raise NotImplementedError(
            "TransformerEncoder.forward is not used in the minimal TTSIM DETR port."
        )


# -----------------------------------------------------------------------------
# TransformerDecoder
# -----------------------------------------------------------------------------

class TransformerDecoder(SimNN.Module):
    """
    TTSIM port of baseline TransformerDecoder(nn.Module).

    Decoder layers are constructed but not used by the minimal
    Transformer.forward above. We keep the structure only.
    """

    def __init__(
        self,
        decoder_layer: "TransformerDecoderLayer",
        num_layers: int,
        norm: Optional[Any] = None,
        return_intermediate: bool = False,
    ):
        super().__init__()
        self.layers: List[TransformerDecoderLayer] = cast(
          List[TransformerDecoderLayer],
            _get_clones(decoder_layer, num_layers),
        )
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

        for i, layer in enumerate(self.layers):
            if isinstance(layer, SimNN.Module):
                self._submodules[f"decoder_layer_{i}"] = layer

        super().link_op2module()

    def forward(
        self,
        tgt: Any,
        memory: Any,
        tgt_mask: Optional[Any] = None,
        memory_mask: Optional[Any] = None,
        tgt_key_padding_mask: Optional[Any] = None,
        memory_key_padding_mask: Optional[Any] = None,
        pos: Optional[Any] = None,
        query_pos: Optional[Any] = None,
    ) -> Any:
        """
        Not used in minimal Transformer.forward; stubbed.
        """
        raise NotImplementedError(
            "TransformerDecoder.forward is not used in the minimal TTSIM DETR port."
        )


# -----------------------------------------------------------------------------
# TransformerEncoderLayer
# -----------------------------------------------------------------------------

class TransformerEncoderLayer(SimNN.Module):
    """
    TTSIM port of baseline TransformerEncoderLayer(nn.Module).

    Kept for structural parity; math not implemented.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        normalize_before: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout_p = dropout
        self.activation_name = activation
        self.normalize_before = normalize_before

        self.self_attn = None
        self.linear1 = None
        self.dropout = None
        self.linear2 = None
        self.norm1 = None
        self.norm2 = None
        self.dropout1 = None
        self.dropout2 = None
        self.activation = _get_activation_fn(activation)

        super().link_op2module()

    def with_pos_embed(self, tensor: Any, pos: Optional[Any]) -> Any:
        if pos is None:
            return tensor
        raise NotImplementedError(
            "TransformerEncoderLayer.with_pos_embed requires a TTSIM Add implementation."
        )

    def forward(
        self,
        src: Any,
        src_mask: Optional[Any] = None,
        src_key_padding_mask: Optional[Any] = None,
        pos: Optional[Any] = None,
    ) -> Any:
        raise NotImplementedError(
            "TransformerEncoderLayer.forward is not implemented in the TTSIM DETR skeleton."
        )


# -----------------------------------------------------------------------------
# TransformerDecoderLayer
# -----------------------------------------------------------------------------

class TransformerDecoderLayer(SimNN.Module):
    """
    TTSIM port of baseline TransformerDecoderLayer(nn.Module).

    Kept for structural parity; math not implemented.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        normalize_before: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout_p = dropout
        self.activation_name = activation
        self.normalize_before = normalize_before

        self.self_attn = None
        self.multihead_attn = None
        self.linear1 = None
        self.dropout = None
        self.linear2 = None
        self.norm1 = None
        self.norm2 = None
        self.norm3 = None
        self.dropout1 = None
        self.dropout2 = None
        self.dropout3 = None
        self.activation = _get_activation_fn(activation)

        super().link_op2module()

    def with_pos_embed(self, tensor: Any, pos: Optional[Any]) -> Any:
        if pos is None:
            return tensor
        raise NotImplementedError(
            "TransformerDecoderLayer.with_pos_embed requires a TTSIM Add implementation."
        )

    def forward(
        self,
        tgt: Any,
        memory: Any,
        tgt_mask: Optional[Any] = None,
        memory_mask: Optional[Any] = None,
        tgt_key_padding_mask: Optional[Any] = None,
        memory_key_padding_mask: Optional[Any] = None,
        pos: Optional[Any] = None,
        query_pos: Optional[Any] = None,
    ) -> Any:
        raise NotImplementedError(
            "TransformerDecoderLayer.forward is not implemented in the TTSIM DETR skeleton."
        )


# -----------------------------------------------------------------------------
# Helper: clone layers
# -----------------------------------------------------------------------------

def _get_clones(module: SimNN.Module, N: int) -> List[SimNN.Module]:
    """
    Baseline:

        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

    TTSIM:
      - A simple Python list of deep copies is sufficient.
    """
    return [copy.deepcopy(module) for _ in range(N)]


# -----------------------------------------------------------------------------
# build_transformer(cfg)
# -----------------------------------------------------------------------------

def build_transformer(cfg: Dict[str, Any]) -> Transformer:
    """
    TTSIM port of baseline build_transformer(args).

    Baseline:

        return Transformer(
            d_model=args.hidden_dim,
            dropout=args.dropout,
            nhead=args.nheads,
            dim_feedforward=args.dim_feedforward,
            num_encoder_layers=args.enc_layers,
            num_decoder_layers=args.dec_layers,
            normalize_before=args.pre_norm,
            return_intermediate_dec=True,
        )

    Here cfg is a dict (Polaris/LeViT style).
    """
    d_model = int(cfg.get("hidden_dim", 256))
    dropout = float(cfg.get("dropout", 0.1))
    nhead = int(cfg.get("nheads", 8))
    dim_feedforward = int(cfg.get("dim_feedforward", 2048))
    num_encoder_layers = int(cfg.get("enc_layers", cfg.get("num_encoder_layers", 6)))
    num_decoder_layers = int(cfg.get("dec_layers", cfg.get("num_decoder_layers", 6)))
    normalize_before = bool(cfg.get("pre_norm", False))

    transformer = Transformer(
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        activation=cfg.get("activation", "relu"),
        normalize_before=normalize_before,
        return_intermediate_dec=True,
    )
    return transformer


# -----------------------------------------------------------------------------
# Activation helper
# -----------------------------------------------------------------------------

def _get_activation_fn(activation: str):
    """
    Baseline:

        if activation == "relu": return F.relu
        if activation == "gelu": return F.gelu
        if activation == "glu":  return F.glu

    TTSIM skeleton:
      - We don't hook into TTSIM activations here.
      - Return identity to keep structure.
    """
    def identity(x):
        return x

    return identity

