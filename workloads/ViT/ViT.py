#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent
# SPDX-License-Identifier: Apache-2.0

import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN
from ttsim.front.functional.tensor_op import *

class Dropout(SimNN.Module):
    def __init__(self, name: str, rate: float):
        super().__init__()
        self.name = name
        self.rate = float(rate)
        self.training = False
        super().link_op2module()
    def __call__(self, x): return x
    def analytical_param_count(self, lvl: int = 0) -> int: return 0

class StochasticDepth(Dropout): pass

class MultiHeadDotProductAttention(SimNN.Module):
    def __init__(self, name: str, dim: int, num_heads: int, attention_dropout: float, qkv_bias: bool):
        super().__init__()
        self.name = name
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        if self.dim % self.num_heads != 0:
            raise ValueError(f"{self.name}: dim {self.dim} not divisible by num_heads {self.num_heads}")
        self.head_dim = self.dim // self.num_heads
        self.scale_val = np.array(self.head_dim ** -0.5, dtype=np.float32)

        self.qkv = F.Linear(self.name + ".qkv.op", self.dim, self.dim * 3, bias=bool(qkv_bias))
        self.proj = F.Linear(self.name + ".proj.op", self.dim, self.dim, bias=True)

        self.transpose_q = F.Transpose(self.name + ".transpose_q", perm=[0, 2, 1, 3])
        self.transpose_k = F.Transpose(self.name + ".transpose_k", perm=[0, 2, 3, 1])
        self.transpose_v = F.Transpose(self.name + ".transpose_v", perm=[0, 2, 1, 3])
        self.transpose_out = F.Transpose(self.name + ".transpose_out", perm=[0, 2, 1, 3])

        self.matmul_qk = F.MatMul(self.name + ".matmul_qk")
        self.matmul_av = F.MatMul(self.name + ".matmul_av")
        self.softmax = F.Softmax(self.name + ".softmax", axis=-1)
        self.scale = F.Mul(self.name + ".scale")

        self.attn_dropout = Dropout(self.name + ".attn_dropout", float(attention_dropout))
        self.proj_dropout = Dropout(self.name + ".proj_dropout", float(attention_dropout))

        for m in (self.attn_dropout, self.proj_dropout):
            self._submodules[m.name] = m
        super().link_op2module()

    def __call__(self, x):
        B, N, _ = x.shape
        qkv = self.qkv(x)
        C = self.dim
        q = qkv[..., 0:C].reshape(B, N, self.num_heads, self.head_dim)
        k = qkv[..., C:2*C].reshape(B, N, self.num_heads, self.head_dim)
        v = qkv[..., 2*C:3*C].reshape(B, N, self.num_heads, self.head_dim)

        q = self.transpose_q(q)
        k = self.transpose_k(k)
        v = self.transpose_v(v)

        scl = F._from_data(self.name + ".scale_val", self.scale_val); self._tensors[scl.name] = scl

        attn = self.matmul_qk(q, k)
        attn = self.scale(attn, scl)
        attn = self.softmax(attn)
        attn = self.attn_dropout(attn)

        y = self.matmul_av(attn, v)
        y = self.transpose_out(y).reshape(B, N, self.dim)
        y = self.proj(y)
        y = self.proj_dropout(y)
        return y

    def analytical_param_count(self, lvl: int = 0) -> int:
        qkv_w = self.dim * (self.dim * 3)
        qkv_b = (self.dim * 3)
        proj_w = self.dim * self.dim
        proj_b = self.dim
        return qkv_w + qkv_b + proj_w + proj_b

class MlpBlock(SimNN.Module):
    def __init__(self, name: str, dim: int, mlp_dim: int, dropout_rate: float):
        super().__init__()
        self.name = name
        self._d0_in = int(dim)
        self._d0_out = int(mlp_dim)
        self._d1_in = int(mlp_dim)
        self._d1_out = int(dim)
        self.Dense_0 = F.Linear(self.name + ".Dense_0.op", self._d0_in, self._d0_out, bias=True)
        self.Dense_1 = F.Linear(self.name + ".Dense_1.op", self._d1_in, self._d1_out, bias=True)
        self.gelu = F.Gelu(self.name + ".gelu"); self.gelu.set_module(self)
        self.Dropout_0 = Dropout(self.name + ".Dropout_0", float(dropout_rate))
        self.Dropout_1 = Dropout(self.name + ".Dropout_1", float(dropout_rate))
        for m in (self.Dropout_0, self.Dropout_1):
            self._submodules[m.name] = m
        super().link_op2module()
    def __call__(self, x):
        x = self.Dense_0(x); x = self.gelu(x); x = self.Dropout_0(x)
        x = self.Dense_1(x); x = self.Dropout_1(x); return x
    def analytical_param_count(self, lvl: int = 0) -> int:
        d0_w = self._d0_in * self._d0_out; d0_b = self._d0_out
        d1_w = self._d1_in * self._d1_out; d1_b = self._d1_out
        return d0_w + d0_b + d1_w + d1_b

class Encoder1DBlock(SimNN.Module):
    def __init__(self, name: str, dim: int, num_heads: int, mlp_dim: int, dropout_rate: float, attention_dropout_rate: float, stochastic_depth: float, layer_norm_eps: float, qkv_bias: bool):
        super().__init__()
        self.name = name
        self._dim = int(dim)
        self.LayerNorm_0 = F.LayerNorm(self.name + ".LayerNorm_0.op", self._dim, normalized_shape=[self._dim], eps=float(layer_norm_eps))
        self.MultiHeadDotProductAttention_0 = MultiHeadDotProductAttention(self.name + ".MultiHeadDotProductAttention_0", self._dim, int(num_heads), float(attention_dropout_rate), bool(qkv_bias))
        self.Dropout_0 = Dropout(self.name + ".Dropout_0", float(dropout_rate))
        self.DropPath_0 = StochasticDepth(self.name + ".DropPath_0", float(stochastic_depth))
        self.Add_0 = F.Add(self.name + ".Add_0")
        self.LayerNorm_1 = F.LayerNorm(self.name + ".LayerNorm_1.op", self._dim, normalized_shape=[self._dim], eps=float(layer_norm_eps))
        self.MlpBlock_0 = MlpBlock(self.name + ".MlpBlock_0", self._dim, int(mlp_dim), float(dropout_rate))
        self.DropPath_1 = StochasticDepth(self.name + ".DropPath_1", float(stochastic_depth))
        self.Add_1 = F.Add(self.name + ".Add_1")
        for m in (self.MultiHeadDotProductAttention_0, self.Dropout_0, self.DropPath_0, self.MlpBlock_0, self.DropPath_1):
            self._submodules[m.name] = m
        super().link_op2module()
    def __call__(self, inputs):
        x = inputs
        y = self.LayerNorm_0(x); y = self.MultiHeadDotProductAttention_0(y); y = self.Dropout_0(y); y = self.DropPath_0(y); x = self.Add_0(x, y)
        y = self.LayerNorm_1(x); y = self.MlpBlock_0(y); y = self.DropPath_1(y); x = self.Add_1(x, y)
        return x
    def analytical_param_count(self, lvl: int = 0) -> int:
        dim = self._dim; cnt = 2 * dim
        cnt += self.MultiHeadDotProductAttention_0.analytical_param_count(lvl + 1)
        cnt += 2 * dim; cnt += self.MlpBlock_0.analytical_param_count(lvl + 1); return cnt

class Encoder(SimNN.Module):
    def __init__(self, name: str, dim: int, mlp_dim: int, num_heads: int, num_layers: int, dropout_rate: float, attention_dropout_rate: float, stochastic_depth_rate: float, layer_norm_eps: float, qkv_bias: bool):
        super().__init__()
        self.name = name
        self._dim = int(dim)
        self.Dropout = Dropout(self.name + ".Dropout", float(dropout_rate))
        self.layers: List[Encoder1DBlock] = []
        drop_rates = np.linspace(0.0, float(stochastic_depth_rate), int(num_layers), dtype=np.float32)
        for i in range(int(num_layers)):
            blk = Encoder1DBlock(f"{self.name}.Encoder1DBlock_{i}", self._dim, int(num_heads), int(mlp_dim), float(dropout_rate), float(attention_dropout_rate), float(drop_rates[i]), float(layer_norm_eps), bool(qkv_bias))
            self.layers.append(blk); self._submodules[blk.name] = blk
        self.LayerNorm = F.LayerNorm(self.name + ".LayerNorm.op", self._dim, normalized_shape=[self._dim], eps=float(layer_norm_eps))
        self._submodules[self.Dropout.name] = self.Dropout
        super().link_op2module()
    def __call__(self, inputs):
        x = self.Dropout(inputs)
        for blk in self.layers: x = blk(x)
        return self.LayerNorm(x)
    def analytical_param_count(self, lvl: int = 0) -> int:
        dim = self._dim; cnt = 2 * dim
        for blk in self.layers: cnt += blk.analytical_param_count(lvl + 1)
        return cnt

@dataclass
class config:
    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    num_classes: int = 1000
    hidden_size: int = 768
    mlp_dim: int = 3072
    num_heads: int = 12
    num_layers: int = 12
    dropout_rate: float = 0.0
    attention_dropout_rate: float = 0.0
    stochastic_depth_rate: float = 0.0
    qkv_bias: bool = True
    layer_norm_eps: float = 1.0e-6
    classifier: str = "token"
    representation_size: Optional[int] = None

class VisionTransformer(SimNN.Module):
    def __init__(self, name: str, cfg: Dict[str, Any]):
        super().__init__()
        self.name = name
        self.config = self._build_config(cfg)
        c = self.config
        if c.image_size % c.patch_size != 0:
            raise ValueError(f"{self.name}: image size {c.image_size} must be divisible by patch size {c.patch_size}")
        self.batch_size = int(cfg.get("bs", 1))
        self.num_patches = (c.image_size // c.patch_size) ** 2
        self.sequence_length = self.num_patches + 1

        self.embedding = type("embedding", (), {})()
        setattr(self.embedding, "op", F.Conv2d(f"{self.name}.embedding.op", int(c.in_channels), int(c.hidden_size), kernel_size=int(c.patch_size), stride=int(c.patch_size), padding=0, bias=True))
        k = int(c.patch_size); in_c = int(c.in_channels); out_c = int(c.hidden_size)
        emb_w = F._from_data(f"{self.name}.embedding.op.param", np.zeros([out_c, in_c, k, k], dtype=np.float32))
        emb_b = F._from_data(f"{self.name}.embedding.op.bias", np.zeros([out_c], dtype=np.float32))
        self._tensors[emb_w.name] = emb_w; self._tensors[emb_b.name] = emb_b

        setattr(self.embedding, "transpose_nch", F.Transpose(f"{self.name}.embedding.transpose_nch", perm=[0, 2, 1]))
        self.embedding.op.set_module(self); self.embedding.transpose_nch.set_module(self)
        self._op_hndls[self.embedding.op.name] = self.embedding.op
        self._op_hndls[self.embedding.transpose_nch.name] = self.embedding.transpose_nch

        self.cls = F._from_data(f"{self.name}.cls", np.zeros([1, 1, c.hidden_size], dtype=np.float32))
        self.posembed_input = F._from_data(f"{self.name}.posembed_input", np.zeros([1, self.sequence_length, c.hidden_size], dtype=np.float32))
        self._tensors[self.cls.name] = self.cls; self._tensors[self.posembed_input.name] = self.posembed_input

        self._Concat = F.ConcatX(self.name + ".Concat", axis=1); self._Concat.set_module(self); self._op_hndls[self._Concat.name] = self._Concat
        self._AddPosemb = F.Add(self.name + ".AddPosemb")

        mlp_dim = int(cfg.get("mlp_dim", c.mlp_dim))
        self.encoder = Encoder(f"{self.name}.encoder", c.hidden_size, mlp_dim, c.num_heads, c.num_layers, c.dropout_rate, c.attention_dropout_rate, c.stochastic_depth_rate, c.layer_norm_eps, c.qkv_bias)
        self._submodules[self.encoder.name] = self.encoder

        if c.representation_size and c.representation_size > 0:
            self.pre_logits = F.Linear(f"{self.name}.pre_logits.op", c.hidden_size, int(c.representation_size), bias=True)
            self._Tanh = F.Tanh(self.name + ".tanh"); self._Tanh.set_module(self)
        else:
            self.pre_logits = None; self._Tanh = None

        head_in = int(c.representation_size) if (self.pre_logits is not None) else c.hidden_size
        self.head = F.Linear(f"{self.name}.head.op", head_in, int(c.num_classes), bias=True) if c.num_classes > 0 else None

        self._GatherCLS = F.Gather(self.name + ".GatherCLS", axis=1)
        self._GlobalAvg = F.ReduceMean(self.name + ".GlobalAvg", axes=[1]) if c.classifier == "gap" else None

        self._cls_broadcast_cache: Dict[int, Any] = {}
        self._posemb_broadcast_cache: Dict[int, Any] = {}
        self._cls_index = F._from_data(f"{self.name}.cls_index", np.array([0], dtype=np.int64))
        self._tensors[self._cls_index.name] = self._cls_index

        self.training = False
        super().link_op2module()

    @staticmethod
    def _build_config(cfg: Dict[str, Any]) -> config:
        cfg = dict(cfg)
        mlp_dim = cfg.get("mlp_dim", int(cfg.get("embed_dim", cfg.get("hidden_size", 768)) * cfg.get("mlp_ratio", 4.0)))
        return config(
            image_size=int(cfg.get("img_height", cfg.get("image_size", 224))),
            patch_size=int(cfg.get("patch_size", 16)),
            in_channels=int(cfg.get("img_channels", cfg.get("in_channels", 3))),
            num_classes=int(cfg.get("num_classes", 1000)),
            hidden_size=int(cfg.get("embed_dim", cfg.get("hidden_size", 768))),
            mlp_dim=int(mlp_dim),
            num_heads=int(cfg.get("num_heads", 12)),
            num_layers=int(cfg.get("depth", cfg.get("num_layers", 12))),
            dropout_rate=float(cfg.get("dropout", cfg.get("dropout_rate", 0.0))),
            attention_dropout_rate=float(cfg.get("attention_dropout", cfg.get("attention_dropout_rate", 0.0))),
            stochastic_depth_rate=float(cfg.get("drop_path", cfg.get("stochastic_depth_rate", 0.0))),
            qkv_bias=bool(cfg.get("qkv_bias", True)),
            layer_norm_eps=float(cfg.get("layer_norm_eps", 1.0e-6)),
            classifier=str(cfg.get("classifier", "token")).lower(),
            representation_size=cfg.get("representation_size", None),
        )

    def _get_cls_broadcast(self, B: int, C: int):
        t = self._cls_broadcast_cache.get(B)
        if t is None:
            arr = np.zeros((B, 1, C), dtype=np.float32)
            t = F._from_data(f"{self.name}.cls.broadcast.{B}", arr)
            self._tensors[t.name] = t
            self._cls_broadcast_cache[B] = t
        return t

    def _get_posemb_broadcast(self, B: int, N: int, C: int):
        t = self._posemb_broadcast_cache.get(B)
        if t is None:
            arr = np.zeros((B, N, C), dtype=np.float32)
            t = F._from_data(f"{self.name}.posembed_input.broadcast.{B}", arr)
            self._tensors[t.name] = t
            self._posemb_broadcast_cache[B] = t
        return t

    def create_input_tensors(self):
        c = self.config
        self.input_tensors = {
            f"{self.name}_input": F._from_shape(f"{self.name}_input", [int(self.batch_size), int(c.in_channels), int(c.image_size), int(c.image_size)])
        }

    def __call__(self):
        c = self.config
        x = self.input_tensors[f"{self.name}_input"]

        y = self.embedding.op(x)
        B, C, H, W = y.shape
        y = y.reshape(B, C, H * W)
        x = self.embedding.transpose_nch(y)  # [B, N, C]
        B = x.shape[0]

        # Use pure data tensors for broadcasts (no Add)
        cls_broadcast = self._get_cls_broadcast(int(B), c.hidden_size)      # [B,1,C] zeros
        cls_token = F._from_data(f"{self.name}.cls_token.{B}", np.zeros((B,1,c.hidden_size), dtype=np.float32))
        self._tensors[cls_token.name] = cls_token
        # Instead of adding, just use cls_broadcast as the per-batch cls token; keep base cls as parameter leaf
        # If you need base cls value included, you can model it as parameterized data during weight load.

        tokens = self._Concat(cls_token, x)  # [B, N+1, C]

        posemb = self._get_posemb_broadcast(int(B), self.sequence_length, c.hidden_size)
        x = self._AddPosemb(tokens, posemb)

        x = self.encoder(x)

        if c.classifier == "gap":
            representation = self._GlobalAvg(x)
        else:
            cls_only = self._GatherCLS(x, F._from_data(f"{self.name}.cls_index", np.array([0], dtype=np.int64)))
            representation = cls_only.reshape(x.shape[0], c.hidden_size)

        if self.pre_logits is not None:
            representation = self.pre_logits(representation); representation = self._Tanh(representation)
        if self.head is not None:
            logits = self.head(representation); return logits
        return representation

    def get_forward_graph(self):
        return super()._get_forward_graph(self.input_tensors)

    def analytical_param_count(self, lvl: int = 0) -> int:
        c = self.config
        k = int(c.patch_size)
        embed_w = int(c.in_channels) * int(c.hidden_size) * k * k
        embed_b = int(c.hidden_size)
        cls_p = 1 * 1 * int(c.hidden_size)
        pos_p = 1 * (self.sequence_length) * int(c.hidden_size)
        cnt = embed_w + embed_b + cls_p + pos_p
        cnt += self.encoder.analytical_param_count(lvl + 1)
        if self.pre_logits is not None:
            pre_in = c.hidden_size; pre_out = int(c.representation_size)
            cnt += pre_in * pre_out + pre_out
        if self.head is not None:
            head_in = int(c.representation_size) if (self.pre_logits is not None) else c.hidden_size
            head_out = int(c.num_classes)
            cnt += head_in * head_out + head_out
        return cnt

VIT = VisionTransformer

if __name__ == "__main__":
    cfg = {
        "img_height": 224, "img_width": 224, "img_channels": 3, "bs": 1,
        "patch_size": 16, "embed_dim": 768, "depth": 12, "num_heads": 12,
        "mlp_ratio": 4.0, "num_classes": 1000, "dropout": 0.0,
        "attention_dropout": 0.0, "drop_path": 0.0, "qkv_bias": True,
        "layer_norm_eps": 1.0e-6, "classifier": "token",
    }
    model = VisionTransformer("vit_test", cfg)
    model.create_input_tensors()
    out = model()
    try: shape = out.shape
    except Exception: shape = getattr(out, "out", out).shape
    print("Output shape:", shape)
    print("Analytical parameter count:", f"{model.analytical_param_count():,d}")
    g = model.get_forward_graph()
    # Optional debug dumps (if supported)
    try:
        g.debug_dump_edges("vit_edges.txt")
    except Exception:
        pass
    g.graph2onnx("vit.onnx", do_model_check=False)
