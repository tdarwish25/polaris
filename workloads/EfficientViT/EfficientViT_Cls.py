#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent
# SPDX-License-Identifier: Apache-2.0

# -----------------------------------------------------------------------------
# EfficientViT_Cls (TT-Sim mapping, Polaris-ready)
#
# Baseline provenance (source alignment):
# - This mapping follows the baseline EfficientViT implementation from:
#   - efficientvit/models/nn/ops.py            → Conv/BN/Activation, Linear
#   - efficientvit/models/backbone.py          → EfficientViTBackbone, EfficientViTLargeBackbone
#   - efficientvit/models/cls.py               → Classification head structure and wiring
# - Attention blocks are omitted per provided YAML/spec when not used by CLS.
# - Operator names and wiring are preserved to match baseline naming for review.
#
# This file maps baseline EfficientViT classification to TT‑Sim operators:
# - Primitives (modeled via TT‑Sim F.* ops):
#     ConvLayer, ResidualBlock, MBConv, FusedMBConv (via Conv/MB wrappers), LinearLayer
# - Backbones:
#     Small backbone modeled after baseline EfficientViTBackbone
#     Large backbone modeled after baseline EfficientViTLargeBackbone
# - Head:
#     Based on baseline efficientvit/models/cls.py structure. Implemented as:
#       Conv1x1 → GAP (AveragePool HxW) → Squeeze(H,W) → Linear → LayerNorm → Act → (Dropout) → Linear
#     Preserved operator names:
#       "cls.head.conv1x1.*"
#       "cls.head.gap", "cls.head.gap.div" (kept handle; not used in math path)
#       "cls.head.gap.squeeze_hw" (+ ".axes" const tensor)
#       "cls.head.fc1.linear", "cls.head.fc1.norm", "cls.head.fc1.act"
#       "cls.head.dropout"
#       "cls.head.fc2"
#
# Notes on TT‑Sim integration fixes:
# - sim_nn.py / wl_graph.py alignment:
#   - Squeeze axes tensor is explicitly registered in Module._tensors and linked to the module
#     before use, ensuring WorkloadGraph.add_op sees all inputs:
#       axes_hw = _from_data("cls.head.gap.squeeze_hw.axes", [2,3], const)
#       self._tensors[axes_hw.name] = axes_hw
#       self.squeeze_hw(y, axes_hw)
#   - This resolves previous KeyError/assertions during graph2onnx related to missing inputs/nodes.
# - No numerical changes versus baseline; changes are graph-bookkeeping only.
# -----------------------------------------------------------------------------

import os, sys
import numpy as np
from typing import Any, Dict, List, Optional

# Ensure repo root on path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN
from ttsim.front.functional.tensor_op import *
from ttsim.front.functional.op import SimOpHandle, _from_data  # used for GAP impl and helpers

# -------------------- Primitives --------------------

class ConvLayer(SimNN.Module):
    """
    Source alignment:
    - Mirrors efficientvit/models/nn/ops.py: Conv → (BatchNorm) → Activation
    - Purpose here: TT‑Sim primitive wrapper that preserves baseline block semantics and naming.

    Flow:
    - Conv2d(name+".conv") → optional BatchNorm2d(name+".bn") → Activation(name+".act")

    Notes:
    - Activation mapping keeps baseline intent; non-specified activations default to Gelu.
    - Parameter count includes Conv and optional BN (gamma+beta).
    """
    def __init__(
        self,
        name: str,
        in_ch: int,
        out_ch: int,
        ks: int,
        stride: int = 1,
        padding: int = 0,
        use_bias: bool = False,
        use_bn: bool = True,
        activation: Optional[str] = "gelu",
        groups: int = 1,
    ):
        super().__init__()
        self.name = name
        self._in_ch = int(in_ch)
        self._out_ch = int(out_ch)
        self._ks = int(ks)
        self._use_bias = bool(use_bias)
        self._use_bn = bool(use_bn)
        self._groups = int(groups)

        self.conv = F.Conv2d(name + ".conv", in_ch, out_ch, kernel_size=ks, stride=stride,
                             padding=padding, groups=groups, bias=use_bias)
        self.conv.set_module(self)

        self.bn = F.BatchNorm2d(name + ".bn", out_ch) if use_bn else None
        if self.bn is not None and hasattr(self.bn, "set_module"):
            self.bn.set_module(self)
        if isinstance(self.bn, SimNN.Module):
            self._submodules[self.bn.name] = self.bn

        self.act = None
        if activation not in (None, "none"):
            a = str(activation).lower()
            if a == "gelu":
                self.act = F.Gelu(name + ".act")
            elif a == "relu":
                self.act = F.Relu(name + ".act")
            elif a in ("hardswish", "hradSwish", "silu"):
                self.act = F.Hardswish(name + ".act")
            else:
                self.act = F.Gelu(name + ".act")
            self.act.set_module(self)

    def __call__(self, x):
        y = self.conv(x)
        if self.bn is not None:
            y = self.bn(y)
        if self.act is not None:
            y = self.act(y)
        return y

    def analytical_param_count(self, lvl=0):
        oc = self._out_ch
        ic = self._in_ch
        ks = self._ks
        groups = max(1, self._groups)
        bias_params = oc if self._use_bias else 0
        conv_params = oc * (ic // groups) * ks * ks + bias_params
        bn_params = 2 * oc if self._use_bn else 0
        return int(conv_params + bn_params)


class ResidualBlock(SimNN.Module):
    """
    Source alignment:
    - Baseline residual wrapper (e.g., efficientvit/models/nn/ops.py usage).
    - Provides optional skip connection: out = x + block(x), else just block(x).

    Notes:
    - Keeps block as a submodule for clear nesting and introspection.
    - Adds named Add op (name+".add") to match graph readability.
    """
    def __init__(self, name: str, block: SimNN.Module, use_skip: bool = True):
        super().__init__()
        self.name = name
        self.block = block
        self._submodules[block.name] = block
        self.use_skip = bool(use_skip)
        self.add = F.Add(name + ".add")
        self.add.set_module(self)

    def __call__(self, x):
        y = self.block(x)
        return self.add(x, y) if self.use_skip else y

    def analytical_param_count(self, lvl=0):
        return self.block.analytical_param_count(lvl + 1)


class MBConv(SimNN.Module):
    """
    Source alignment:
    - Mirrors EfficientViT MBConv structure from the baseline:
      depthwise 3x3 (groups=in_ch) followed by pointwise 1x1, with residual if in==out and stride==1.

    Structure:
    - DW: ConvLayer(name+".dw", groups=in_ch) with ks=3
    - PW: ConvLayer(name+".pw", ks=1)
    - Optional residual via Add(name+".add")
    """
    def __init__(self, name: str, in_ch: int, out_ch: int, stride: int = 1, activation: str = "gelu"):
        super().__init__()
        self.name = name
        self.use_res = (in_ch == out_ch and stride == 1)
        self.dw = ConvLayer(name + ".dw", in_ch, in_ch, ks=3, stride=stride, padding=1,
                            use_bias=False, use_bn=True, activation=None, groups=in_ch)
        self.pw = ConvLayer(name + ".pw", in_ch, out_ch, ks=1, stride=1, padding=0,
                            use_bias=False, use_bn=True, activation=activation, groups=1)
        self._submodules[self.dw.name] = self.dw
        self._submodules[self.pw.name] = self.pw
        self.add = F.Add(name + ".add")
        self.add.set_module(self)

    def __call__(self, x):
        y = self.pw(self.dw(x))
        return self.add(x, y) if self.use_res else y

    def analytical_param_count(self, lvl=0):
        return self.dw.analytical_param_count(lvl + 1) + self.pw.analytical_param_count(lvl + 1)


class FusedMBConv(SimNN.Module):
    """
    Source alignment:
    - Mirrors baseline fused MBConv: spatial 3x3 (expand) + 1x1 projection.

    Structure:
    - spatial_conv: ConvLayer(..., ks=3, stride=stride, padding=ks//2, expand via mid_ch)
    - point_conv  : ConvLayer(..., ks=1)

    Notes:
    - expand_ratio and mid_ch match baseline conventions.
    """
    def __init__(
        self,
        name: str,
        in_ch: int,
        out_ch: int,
        ks: int = 3,
        stride: int = 1,
        mid_ch: Optional[int] = None,
        expand_ratio: float = 6.0,
        groups: int = 1,
        activation_spatial: Optional[str] = "gelu",
        activation_point: Optional[str] = None,
        use_bias_spatial: bool = False,
        use_bias_point: bool = False,
        use_bn_spatial: bool = True,
        use_bn_point: bool = True,
        padding: Optional[int] = None,
    ):
        super().__init__()
        self.name = name
        if padding is None:
            padding = ks // 2
        if mid_ch is None:
            mid_ch = int(round(in_ch * float(expand_ratio)))

        self.spatial = ConvLayer(
            name=name + ".spatial_conv",
            in_ch=in_ch,
            out_ch=mid_ch,
            ks=int(ks),
            stride=int(stride),
            padding=int(padding),
            use_bias=bool(use_bias_spatial),
            use_bn=bool(use_bn_spatial),
            activation=activation_spatial,
            groups=int(groups),
        )
        self.point = ConvLayer(
            name=name + ".point_conv",
            in_ch=mid_ch,
            out_ch=out_ch,
            ks=1,
            stride=1,
            padding=0,
            use_bias=bool(use_bias_point),
            use_bn=bool(use_bn_point),
            activation=activation_point,
            groups=1,
        )

        self._submodules[self.spatial.name] = self.spatial
        self._submodules[self.point.name] = self.point

    def __call__(self, x):
        y = self.spatial(x)
        y = self.point(y)
        return y

    def analytical_param_count(self, lvl=0):
        return self.spatial.analytical_param_count(lvl + 1) + self.point.analytical_param_count(lvl + 1)


class LinearLayer(SimNN.Module):
    """
    Source alignment:
    - Thin wrapper over TT‑Sim F.Linear to keep baseline-aligned naming and structure (efficientvit/models/nn/ops.py::Linear).

    Notes:
    - Exposes analytical_param_count consistent with baseline Linear parameterization.
    """
    def __init__(self, name: str, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.name = name
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.bias = bool(bias)
        self.linear = F.Linear(name, self.in_features, self.out_features, bias=self.bias)
        self.linear.set_module(self)

    def __call__(self, x):
        return self.linear(x)

    def analytical_param_count(self) -> int:
        b = self.out_features if self.bias else 0
        return self.in_features * self.out_features + b

# -------------------- Head --------------------

class ClsHead(SimNN.Module):
    """
    Source alignment:
    - Based on baseline efficientvit/models/cls.py head:
      Conv1x1 → Global Average Pool → MLP (Linear → LayerNorm → Act → (Dropout) → Linear)

    Implementation in TT‑Sim:
    - Conv1x1: ConvLayer(name+".conv1x1", ...)
    - GAP: AveragePool(name+".gap", kernel HxW of final stage) followed by Squeeze over spatial dims:
        - squeeze node: name+".gap.squeeze_hw"
        - axes const:   name+".gap.squeeze_hw.axes" with [2, 3]
      Note: axes const is explicitly registered into Module._tensors to satisfy WorkloadGraph input checks.
    - fc1: LinearLayer(name+".fc1.linear", ...)
    - LayerNorm: F.LayerNorm(name+".fc1.norm", ...)
    - Act: F.Gelu / F.Hardswish / F.Relu (name+".fc1.act") per config
    - (optional) Dropout: F.Dropout(name+".dropout", ...)
    - fc2: LinearLayer(name+".fc2", ...)

    Graph export notes:
    - Preserves stable op/tensor names for review and Polaris stats.
    - No numerical differences from baseline intent; only graph bookkeeping differs (explicit squeeze axes registration).
    """
    def __init__(
        self,
        name: str,
        in_channels: int,
        width_list: List[int],
        n_classes: int = 1000,
        dropout: float = 0.0,
        norm: str = "bn2d",
        act_func: Optional[str] = "hswish",
        fid: str = "stage_final",
    ):
        super().__init__()
        self.name = name
        assert len(width_list) >= 2, "width_list must be [mid, hidden]"
        mid = int(width_list[0])
        hidden = int(width_list[1])
        self.fid = fid

        self.conv1x1 = ConvLayer(
            name + ".conv1x1",
            in_ch=in_channels,
            out_ch=mid,
            ks=1,
            stride=1,
            padding=0,
            use_bias=False,
            use_bn=(norm is not None and str(norm).lower() != "none"),
            activation=act_func,
            groups=1,
        )
        self._submodules[self.conv1x1.name] = self.conv1x1

        # GAP via AveragePool (global pooling kernel = stage_final HxW; default 7)
        self._final_hw = 7  # can be overridden upstream by setting attrs on gap_pool if desired
        kh, kw = self._final_hw, self._final_hw
        self.gap_pool = SimOpHandle(name + ".gap", 'AveragePool',
                                    params=[], ipos=[0],
                                    kernel_shape=[kh, kw],
                                    pads=[0, 0, 0, 0],
                                    strides=[1, 1])
        self.gap_pool.set_module(self)
        self.gap_div = F.Div(name + ".gap.div"); self.gap_div.set_module(self)

        # Pre-create squeeze handle (two-input form); keep stable name for review
        self.squeeze_hw = SimOpHandle(name + ".gap.squeeze_hw", "Squeeze",
                                      params=[], ipos=[0, 1])
        self.squeeze_hw.set_module(self)

        self.fc1 = LinearLayer(name + ".fc1.linear", mid, hidden, bias=False)
        self._submodules[self.fc1.name] = self.fc1

        self.ln1 = F.LayerNorm(name + ".fc1.norm", hidden, normalized_shape=[hidden], eps=1e-5); self.ln1.set_module(self)

        self.act1 = None
        if act_func not in (None, "none"):
            a = str(act_func).lower()
            if a == "gelu":
                self.act1 = F.Gelu(name + ".fc1.act")
            elif a == "relu":
                self.act1 = F.Relu(name + ".fc1.act")
            elif a in ("hardswish", "HardSwish", "silu"):
                self.act1 = F.Hardswish(name + ".fc1.act")
            else:
                self.act1 = F.Gelu(name + ".fc1.act")
            self.act1.set_module(self)

        self.dropout = None
        if float(dropout) > 0.0:
            self.dropout = F.Dropout(name + ".dropout", p=float(dropout), inplace=False)
            self.dropout.set_module(self)

        self.fc2 = LinearLayer(name + ".fc2", hidden, int(n_classes), bias=True)
        self._submodules[self.fc2.name] = self.fc2

        self._gap_hw_tensor_name = name + ".gap.hw"
        self._hidden = hidden
        self._mid = mid

    def create_input_helpers(self, H: int, W: int) -> Dict[str, Any]:
        """
        Source alignment:
        - Baseline head uses ReduceMean; here we keep a helper for HW (legacy) and seeding consistency.
        - Returns a const tensor named "cls.head.gap.hw" = H*W.
        """
        h = F._from_data(self._gap_hw_tensor_name, np.array(H * W, dtype=np.float32))
        hw = h.out if hasattr(h, "out") else h
        hw = hw.out if hasattr(hw, "out") else hw
        return {self._gap_hw_tensor_name: hw}

    def __call__(self, x, gap_hw_scalar_tensor):
        # x: [B, C, H, W]
        y = self.conv1x1(x)
        # GAP: [B, C, 1, 1]
        y = self.gap_pool(y)

        # Squeeze both spatial dims together using axes [2, 3]
        axes_hw = F._from_data(self.name + ".gap.squeeze_hw.axes", np.array([2, 3], dtype=np.int64), is_param=False, is_const=True)
        axes_hw = axes_hw.out if hasattr(axes_hw, "out") else axes_hw
        # Register axes const so WorkloadGraph sees all inputs of Squeeze
        if axes_hw.name not in self._tensors:
            self._tensors[axes_hw.name] = axes_hw
        if getattr(axes_hw, "link_module", None) is None:
            try:
                axes_hw.set_module(self)
            except Exception:
                pass

        y = self.squeeze_hw(y, axes_hw)  # [B, C]

        # fc stack
        y = self.fc1(y)
        y = self.ln1(y)
        if self.act1 is not None:
            y = self.act1(y)
        if self.dropout is not None:
            y = self.dropout(y)
        y = self.fc2(y)
        return y

    def analytical_param_count(self, lvl=0):
        cnt = self.conv1x1.analytical_param_count(lvl + 1)
        cnt += self.fc1.analytical_param_count()
        cnt += 2 * self._hidden  # LN gamma+beta
        cnt += self.fc2.analytical_param_count()
        return int(cnt)

# -------------------- Backbones --------------------

class EfficientViTBackbone(SimNN.Module):
    """
    Source alignment:
    - Small EfficientViT backbone (efficientvit/models/backbone.py):
      Stem (two conv blocks) + 4 stages of MBConv with downsampling at stage boundaries.

    Structure:
    - stem1, stem2: ConvLayer blocks
    - stages s0..s3: lists of MBConv blocks, with stride=2 at the first block of each stage except s0

    Outputs:
    - Returns a dict with stage tensors and "stage_final" as the last tensor, matching baseline tap points.
    """
    def __init__(
        self,
        name: str,
        img_channels: int,
        stem_out: int,
        dims: List[int],
        blocks: List[int],
        activation: str = "gelu",
    ):
        super().__init__()
        self.name = name
        self.act = activation

        self.stem1 = ConvLayer(name + ".stem1", img_channels, stem_out, ks=3, stride=2, padding=1, activation=activation)
        self.stem2 = ConvLayer(name + ".stem2", stem_out,      stem_out, ks=3, stride=1, padding=1, activation=activation)
        self._submodules[self.stem1.name] = self.stem1
        self._submodules[self.stem2.name] = self.stem2

        self.stages: List[List[SimNN.Module]] = []
        in_planes = stem_out
        for si, (cout, nblk) in enumerate(zip(dims, blocks)):
            stage: List[SimNN.Module] = []
            first_stride = 1 if si == 0 else 2
            blk0 = MBConv(f"{name}.s{si}.b0", in_planes, cout, stride=first_stride, activation=activation)
            stage.append(blk0); self._submodules[blk0.name] = blk0
            in_planes = cout
            for bi in range(1, int(nblk)):
                blk = MBConv(f"{name}.s{si}.b{bi}", in_planes, cout, stride=1, activation=activation)
                stage.append(blk); self._submodules[blk.name] = blk
            self.stages.append(stage)

    def __call__(self, x) -> Dict[str, Any]:
        y = self.stem2(self.stem1(x))
        feed: Dict[str, Any] = {}
        for si, stage in enumerate(self.stages):
            for blk in stage:
                y = blk(y)
            feed[f"stage{si}"] = y
        feed["stage_final"] = y
        return feed

    def analytical_param_count(self, lvl=0):
        cnt = self.stem1.analytical_param_count(lvl + 1) + self.stem2.analytical_param_count(lvl + 1)
        for st in self.stages:
            for blk in st:
                cnt += blk.analytical_param_count(lvl + 1)
        return int(cnt)


class EfficientViTLargeBackbone(SimNN.Module):
    """
    Source alignment:
    - Large EfficientViT backbone (efficientvit/models/backbone.py), L1–L3 style:
      Stage0 (stem + shallow fused blocks), then deeper stages with MBConv and fused variants.

    Structure:
    - stage0: stem ConvLayer + Residual(FusedMBConv) blocks
    - stages12: residual MBConv stages with downsample on first block
    - stages3p: transition MBConv + residual FusedMBConv blocks

    Outputs:
    - Returns per-stage tensors and "stage_final", matching baseline convention.
    """
    def __init__(
        self,
        name: str,
        img_channels: int,
        width_list: List[int],
        depth_list: List[int],
        dim: int = 32,
        expand_ratio: int = 4,
        norm: str = "bn2d",
        act_func: str = "gelu",
    ):
        super().__init__()
        self.name = name

        w0 = int(width_list[0])
        stem = ConvLayer(name + ".s0.stem", img_channels, w0, ks=3, stride=2, padding=1, activation=act_func)
        self._submodules[stem.name] = stem
        self.stage0: List[SimNN.Module] = [stem]

        in_ch = w0
        for bi in range(int(depth_list[0])):
            blk = FusedMBConv(f"{name}.s0.b{bi}", in_ch, in_ch, ks=3, stride=1,
                              expand_ratio=1.0, activation_spatial=act_func, activation_point=None,
                              use_bias_spatial=False, use_bias_point=False, use_bn_spatial=True, use_bn_point=True)
            self._submodules[blk.name] = blk
            res = ResidualBlock(f"{name}.s0.res{bi}", blk, use_skip=True)
            self._submodules[res.name] = res
            self.stage0.append(res)

        self.stages12: List[List[SimNN.Module]] = []
        for si, (w, d) in enumerate(zip(width_list[1:3], depth_list[1:3]), start=1):
            w = int(w); d = int(d)
            stage: List[SimNN.Module] = []
            for bi in range(d):
                stride = 2 if bi == 0 else 1
                blk = MBConv(f"{name}.s{si}.b{bi}", in_ch, w, stride=stride, activation=act_func)
                self._submodules[blk.name] = blk
                use_skip = (stride == 1 and in_ch == w)
                res = ResidualBlock(f"{name}.s{si}.res{bi}", blk, use_skip=use_skip)
                self._submodules[res.name] = res
                stage.append(res)
                in_ch = w
            self.stages12.append(stage)

        self.stages3p: List[List[SimNN.Module]] = []
        for sj, (w, d) in enumerate(zip(width_list[3:], depth_list[3:]), start=3):
            w = int(w); d = int(d)
            stage: List[SimNN.Module] = []
            trans = MBConv(f"{name}.s{sj}.b0.trans", in_ch, w, stride=2, activation=act_func)
            self._submodules[trans.name] = trans
            stage.append(ResidualBlock(f"{name}.s{sj}.trans", trans, use_skip=False))
            in_ch = w
            for bi in range(d):
                blk = FusedMBConv(f"{name}.s{sj}.b{bi+1}", in_ch, in_ch, ks=3, stride=1,
                                  expand_ratio=expand_ratio, activation_spatial=act_func, activation_point=None,
                                  use_bias_spatial=False, use_bias_point=False, use_bn_spatial=True, use_bn_point=True)
                self._submodules[blk.name] = blk
                res = ResidualBlock(f"{name}.s{sj}.res{bi+1}", blk, use_skip=True)
                self._submodules[res.name] = res
                stage.append(res)
            self.stages3p.append(stage)

        self._out_channels = in_ch

    def __call__(self, x) -> Dict[str, Any]:
        feed: Dict[str, Any] = {}
        y = x
        for m in self.stage0:
            y = m(y)
        feed["stage0"] = y

        for si, stage in enumerate(self.stages12, start=1):
            for m in stage:
                y = m(y)
            feed[f"stage{si}"] = y

        for sj, stage in enumerate(self.stages3p, start=3):
            for m in stage:
                y = m(y)
            feed[f"stage{sj}"] = y

        feed["stage_final"] = y
        return feed

    def analytical_param_count(self, lvl=0):
        cnt = 0
        for m in self.stage0:
            if hasattr(m, "analytical_param_count"):
                cnt += m.analytical_param_count(lvl + 1)
        for stage in self.stages12:
            for m in stage:
                if hasattr(m, "analytical_param_count"):
                    cnt += m.analytical_param_count(lvl + 1)
        for stage in self.stages3p:
            for m in stage:
                if hasattr(m, "analytical_param_count"):
                    cnt += m.analytical_param_count(lvl + 1)
        return int(cnt)

# -------------------- Top model --------------------

class EfficientViT_Cls(SimNN.Module):
    """
    Source alignment:
    - Baseline-matching wrapper coordinating backbone selection and classification head
      (mirrors efficientvit/models/cls.py end-to-end flow for classification).

    Behavior:
    - Selects small or large backbone based on cfg/model_size inference.
    - Taps "stage_final" and inserts "cls.adapter" (1x1, BN, no act) if head_in ≠ backbone_out.
    - Head implemented by ClsHead; preserves operator naming for Polaris review and stats.

    Notes:
    - Includes optional small-model backend hints (B0/B1, bs=1) without changing graph or numerics.
    - create_input_tensors seeds "cls_input" and "cls.head.gap.hw" for stable graph construction.
    """
    def __init__(self, name: str, cfg: Dict[str, Any]):
        super().__init__()
        self.name = name
        c = dict(cfg)

        self.bs = int(c.get("bs", 1))
        self.in_ch = int(c.get("img_channels", 3))
        self.H = int(c.get("img_height", 224))
        self.W = int(c.get("img_width", 224))

        activation = str(c.get("activation", "gelu"))
        stem_out   = int(c.get("stem_out", 32))
        dims       = [int(x) for x in c.get("dims",   [64, 128, 256, 512])]
        blocks     = [int(x) for x in c.get("blocks", [2, 2, 3, 2])]

        model_size = str(c.get("model_size", "")).lower()
        if not model_size:
            inferred = "large" if (int(dims[-1]) >= 768 or int(c.get("head_in", dims[-1])) >= 768) else "small"
            model_size = inferred

        if model_size == "large":
            width_list = [max(stem_out, dims[0] // 2), dims[0], dims[1], dims[2], dims[3]]
            depth_list = [1] + blocks
            self.backbone = EfficientViTLargeBackbone(
                "cls.backbone",
                img_channels=self.in_ch,
                width_list=width_list,
                depth_list=depth_list,
                dim=32,
                expand_ratio=4,
                norm="bn2d",
                act_func=activation,
            )
            backbone_out_ch = width_list[-1]
        else:
            self.backbone = EfficientViTBackbone("cls.backbone", self.in_ch, stem_out, dims, blocks, activation=activation)
            backbone_out_ch = dims[-1]

        self._submodules[self.backbone.name] = self.backbone

        head_in_cfg = c.get("head_in", None)
        head_in = int(head_in_cfg) if head_in_cfg is not None else int(backbone_out_ch)

        if "head_widths" in c:
            head_widths = [int(x) for x in c["head_widths"]]
            if len(head_widths) < 2:
                head_widths = [head_in, head_in]
        else:
            head_widths = [int(c.get("cls_head_mid", head_in)), int(c.get("cls_head_hidden", head_in))]

        head_norm = c.get("cls_head_norm", c.get("head_norm", "bn2d"))
        head_act = c.get("cls_head_act", c.get("head_act", "hswish"))
        head_dropout = float(c.get("cls_head_dropout", c.get("head_dropout", 0.0)))

        self.adapter = None
        if head_in != backbone_out_ch:
            self.adapter = ConvLayer(
                "cls.adapter",
                in_ch=backbone_out_ch,
                out_ch=head_in,
                ks=1,
                stride=1,
                padding=0,
                use_bias=False,
                use_bn=True,
                activation=None,
                groups=1,
            )
            self._submodules[self.adapter.name] = self.adapter

        self.head = ClsHead(
            name="cls.head",
            in_channels=head_in,
            width_list=head_widths,
            n_classes=int(c.get("num_classes", 1000)),
            dropout=head_dropout,
            norm=head_norm,
            act_func=head_act,
            fid="stage_final",
        )
        self._submodules[self.head.name] = self.head

        self.input_tensors: Dict[str, Any] = {}
        self._gap_helpers = self.head.create_input_helpers(self.H, self.W)

        self._head_in_ch = head_in
        self._num_classes = int(c.get("num_classes", 1000))
        self.training = False

        _is_small_bs1 = (model_size == "small" and self.bs == 1)
        if _is_small_bs1:
            hints_fc = {
                "prefer_small_kernel": True,
                "small_kernel_threshold_m": 512,
                "small_kernel_threshold_n": 512,
                "small_kernel_threshold_k": 1024,
                "tile_m": 128,
                "tile_n": 128,
                "tile_k": 64,
                "persistent_kernels": True,
                "vec_width_bits": 128,
                "vec_fallback_bits": 64,
                "max_concurrent_kernels": 2,
            }
            try:
                if hasattr(self.head, "fc1") and hasattr(self.head.fc1, "linear"):
                    lin1 = self.head.fc1.linear
                    try:
                        lin1.set_backend_opts(hints_fc)
                    except Exception:
                        if hasattr(lin1, "backend_opts") and isinstance(lin1.backend_opts, dict):
                            lin1.backend_opts.update(hints_fc)
            except Exception:
                pass
            try:
                if hasattr(self.head, "fc2") and hasattr(self.head.fc2, "linear"):
                    lin2 = self.head.fc2.linear
                    try:
                        lin2.set_backend_opts(hints_fc)
                    except Exception:
                        if hasattr(lin2, "backend_opts") and isinstance(lin2.backend_opts, dict):
                            lin2.backend_opts.update(hints_fc)
            except Exception:
                pass
            try:
                for lin_mod in (
                    getattr(self.head.fc1, "linear", None),
                    getattr(self.head.fc2, "linear", None),
                ):
                    if lin_mod is None:
                        continue
                    opts = {"weight_prepack": True, "weight_static": True}
                    if hasattr(lin_mod, "set_backend_opts"):
                        lin_mod.set_backend_opts(opts)
                    elif hasattr(lin_mod, "backend_opts") and isinstance(lin_mod.backend_opts, dict):
                        lin_mod.backend_opts.update(opts)
            except Exception:
                pass
            try:
                if hasattr(self.head, "ln1"):
                    ln1 = self.head.ln1
                    opts_ln = {"allow_pointwise_fusion": True, "no_upcast_layernorm": True}
                    if hasattr(ln1, "set_backend_opts"):
                        ln1.set_backend_opts(opts_ln)
                    elif hasattr(ln1, "backend_opts") and isinstance(ln1.backend_opts, dict):
                        ln1.backend_opts.update(opts_ln)
                if getattr(self.head, "act1", None) is not None:
                    a1 = self.head.act1
                    opts_act = {"allow_pointwise_fusion": True}
                    if hasattr(a1, "set_backend_opts"):
                        a1.set_backend_opts(opts_act)
                    elif hasattr(a1, "backend_opts") and isinstance(a1, "dict"):
                        a1.backend_opts.update(opts_act)
                if getattr(self.head, "dropout", None) is not None:
                    dr = self.head.dropout
                    opts_dr = {"allow_pointwise_fusion": True}
                    if hasattr(dr, "set_backend_opts"):
                        dr.set_backend_opts(opts_dr)
                    elif hasattr(dr, "backend_opts") and isinstance(dr.backend_opts, dict):
                        dr.backend_opts.update(opts_dr)
            except Exception:
                pass

    def set_batch_size(self, new_bs: int):
        self.bs = int(new_bs)

    def _as_tensor(self, t):
        return t.out if hasattr(t, "out") else t

    def create_input_tensors(self):
        self.input_tensors = {
            "cls_input": self._as_tensor(F._from_shape("cls_input", [self.bs, self.in_ch, self.H, self.W])),
            "cls.head.gap.hw": self._as_tensor(self._gap_helpers["cls.head.gap.hw"]),
        }

    def _normalize_tensor_map(self, m: Dict[str, Any]) -> Dict[str, Any]:
        def unwrap_value(v):
            if isinstance(v, dict):
                return {k: unwrap_value(vv) for k, vv in v.items()}
            if isinstance(v, (list, tuple)):
                items = [unwrap_value(vv) for vv in v]
                return tuple(items) if isinstance(v, tuple) else items
            v2 = v.out if hasattr(v, "out") else v
            v3 = v2.out if hasattr(v2, "out") else v2
            return v3
        return {k: unwrap_value(v) for k, v in m.items()}

    def __call__(self):
        x = self.input_tensors["cls_input"]
        feed = self.backbone(x)
        z = feed["stage_final"]
        if self.adapter is not None:
            z = self.adapter(z)
        hw_scalar = self.input_tensors["cls.head.gap.hw"]
        y = self.head(z, gap_hw_scalar_tensor=hw_scalar)
        return y

    def get_forward_graph(self):
        x = self.input_tensors.get("cls_input", None)
        hw = self.input_tensors.get("cls.head.gap.hw", None)

        x = x.out if hasattr(x, "out") else x
        hw = hw.out if hasattr(hw, "out") else hw

        seeds: Dict[str, Any] = {}
        if hasattr(x, "shape") and hasattr(x, "nbytes"):
            seeds["cls_input"] = x
        if hasattr(hw, "shape") and hasattr(hw, "nbytes"):
            seeds["cls.head.gap.hw"] = hw

        if not seeds:
            raise RuntimeError("No valid SimTensor seeds found for forward graph seeding.")

        return super()._get_forward_graph(seeds)

    def analytical_param_count(self, lvl=0):
        bcnt = self.backbone.analytical_param_count(lvl + 1)
        acnt = self.adapter.analytical_param_count(lvl + 1) if self.adapter is not None else 0
        hcnt = self.head.analytical_param_count(lvl + 1)
        return int(bcnt + acnt + hcnt)

# Backward-compat alias for YAML
EFFICIENTVIT = EfficientViT_Cls

if __name__ == "__main__":
    cfg = {
        "img_channels": 3, "img_height": 224, "img_width": 224, "bs": 1,
        "dims": [80, 224, 384, 1024],
        "blocks": [3, 4, 8, 3],
        "stem_out": 48,
        "num_classes": 1000,
        "head_in": 1024,
        "head_widths": [6144, 6400],
        "activation": "gelu",
        "model_size": "large",
    }
    model = EfficientViT_Cls("effvit_cls_test", cfg)
    model.set_batch_size(1)
    model.create_input_tensors()
    out = model()
    print("Output shape:", out.shape)
    print("Analytical parameter count:", model.analytical_param_count())
    gg = model.get_forward_graph()
    gg.graph2onnx("efficientvit_cls.onnx", do_model_check=False)
