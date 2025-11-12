#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# -----------------------------------------------------------------------------
# EfficientViT_Seg (TT-Sim mapping, Polaris-ready)
#
# This file maps baseline EfficientViT segmentation to TT‑Sim operators:
# - Primitives (aligned with baseline efficientvit/models/nn/ops.py):
#     ConvLayer, AddLayer, UpSampleLayer, ResidualBlock, MBConv
# - Backbone:
#     EfficientViTBackbone modeled after baseline efficientvit/models/backbone.py
#     stages with MBConv blocks; exposes stage2/3/4/5 taps (strides ~4/8/16/32)
# - Head:
#     SegHead aligns to baseline segmentation head behavior (adapter 1x1 per tap,
#     resample to head_stride, additive fuse chain, middle residual MBConv stack,
#     optional expand, final 1x1 logits).
#
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

# -------------------- Small op wrappers (baseline-aligned) --------------------
# Source reference: baseline efficientvit/models/nn/ops.py semantics, mapped to TT‑Sim.


class AddLayer(SimNN.Module):
    """
    Baseline: elementwise add merge. Review mapping:
      - Baseline DAG merge='add' → TT‑Sim F.Add
      - Name pattern preserved: "<parent>.add"
    """
    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.add = F.Add(name + ".op")
        super().link_op2module()

    def __call__(self, x, y):
        return self.add(x, y)


class UpSampleLayer(SimNN.Module):
    """
    Baseline: integer-factor upsample (tile mode) with strict early no-op parity:
      - If (self.size is not None and current size matches) or factor == 1 → return x
      - Mode retained for compatibility; using integer "tile"
    """
    def __init__(self, name: str, factor: int, mode: str = "tile", size: Optional[List[int]] = None):
        super().__init__()
        self.name = name
        self.factor = int(factor) if factor is not None else 1
        self.mode = str(mode)
        # Optional target size (H, W), to mirror baseline resize signature
        self.size = list(size) if size is not None else None
        if self.factor > 1:
            ones = np.ones((1, 1, 1, self.factor, 1, self.factor), dtype=np.float32)
            self.k = F._from_data(name + ".ones", ones)
            self.mul = F.Mul(name + ".mul")
        else:
            self.k = None
            self.mul = None
        super().link_op2module()

    def __call__(self, x):
        # Baseline no-op parity: if size matches OR factor == 1, return x
        if self.size is not None:
            Ht, Wt = int(self.size[0]), int(self.size[1])
            if tuple(x.shape[-2:]) == (Ht, Wt):
                return x
        if self.factor is None or self.factor <= 1:
            return x
        B, C, H, W = x.shape
        x6 = x.reshape([B, C, H, 1, W, 1])
        y6 = self.mul(x6, self.k)
        y  = y6.reshape([B, C, H * self.factor, W * self.factor])
        return y

# -------------------- Core layers and blocks --------------------
# Source reference: efficientvit/models/nn/ops.py → TT‑Sim equivalents here.


class ConvLayer(SimNN.Module):
    """
    Baseline: Conv → (BN) → Act. Review mapping:
      - F.Conv2d, F.BatchNorm2d, activation via F.Gelu/F.Relu/F.Hardswish
      - Activation selection aligned to CLS; maps {"hardswish","hradSwish","silu"} → Hardswish
      - Name pattern preserved: "<parent>.*.conv|.bn|.act"
    """
    def __init__(self, name, in_ch, out_ch, ks=1, stride=1, pad=0, groups=1, activation="hswish", bias=False):
        super().__init__()
        self.name = name
        self.ic = int(in_ch); self.oc = int(out_ch)
        self.ks = int(ks); self.stride = int(stride); self.pad = int(pad)
        self.groups = int(groups); self.use_bias = bool(bias)

        self.conv = F.Conv2d(name + ".conv", self.ic, self.oc,
                             kernel_size=self.ks, stride=self.stride, padding=self.pad,
                             dilation=1, groups=self.groups, bias=self.use_bias)
        self.bn = F.BatchNorm2d(name + ".bn", self.oc)

        # Activation selection block (aligned with CLS)
        self.act = None
        if activation not in (None, "none"):
            a = str(activation).lower()
            if a == "gelu":
                self.act = F.Gelu(name + ".act")
            elif a == "relu":
                self.act = F.Relu(name + ".act")
            elif a in ("hardswish", "hradswish", "silu"):
                self.act = F.Hardswish(name + ".act")
            else:
                self.act = F.Gelu(name + ".act")
            self.act.set_module(self)

        super().link_op2module()

    def __call__(self, x):
        y = self.conv(x)
        y = self.bn(y)
        if self.act is not None:
            y = self.act(y)
        return y

    def analytical_param_count(self, lvl=0):
        conv_w = self.oc * (self.ic // max(1, self.groups)) * self.ks * self.ks
        conv_b = self.oc if self.use_bias else 0
        bn = 2 * self.oc
        return int(conv_w + conv_b + bn)


class MBConv(SimNN.Module):
    """
    Baseline: MBConv (DW 3x3 + PW 1x1). Review mapping:
      - Depthwise: ConvLayer(groups=in_ch)
      - Pointwise: ConvLayer(1x1)
      - Residual if in==out and stride==1
      - Name pattern preserved: "<parent>.dw", "<parent>.pw", "<parent>.add"
    """
    def __init__(self, name, in_ch, out_ch, kernel=3, stride=1, activation="hswish"):
        super().__init__()
        self.name = name
        pad = int(kernel) // 2
        self.use_res = (int(in_ch) == int(out_ch) and int(stride) == 1)
        self.dw = ConvLayer(name + ".dw", in_ch, in_ch, ks=kernel, stride=stride, pad=pad,
                            groups=in_ch, activation=activation)
        self.pw = ConvLayer(name + ".pw", in_ch, out_ch, ks=1, stride=1, pad=0,
                            groups=1, activation=activation)
        self._submodules[self.dw.name] = self.dw
        self._submodules[self.pw.name] = self.pw
        if self.use_res:
            self.add = AddLayer(name + ".add")
            self._submodules[self.add.name] = self.add
        else:
            self.add = None
        super().link_op2module()

    def __call__(self, x):
        y = self.pw(self.dw(x))
        return self.add(x, y) if self.use_res else y

    def analytical_param_count(self, lvl=0):
        return self.dw.analytical_param_count(lvl+1) + self.pw.analytical_param_count(lvl+1)

# -------------------- Backbone (baseline-aligned name) --------------------
# Source: efficientvit/models/backbone.py :: EfficientViTBackbone (seg variant)
# Exposed taps for head: stage2 (~stride 4), stage3 (~stride 8), stage4 (~stride 16), stage5 (~stride 32)


class EfficientViTBackbone(SimNN.Module):
    """
    Baseline: EfficientViTBackbone (segmentation). Review mapping:
      - Stem: two ConvLayers (stride 2 then stride 1)
      - Stages: MBConv blocks; first block in every stage uses stride 2 (including stage 0)
      - Exposes taps: "stage2","stage3","stage4","stage5"
      - Name pattern preserved: "seg.backbone.*", "seg.backbone.s{si}.b{bi}.*"
    """
    def __init__(self, name, in_ch=3, stem_out=16, dims=(32,64,128,128), blocks=(1,2,2,2), activation="hswish"):
        super().__init__()
        self.name = name
        self.stem1 = ConvLayer(name + ".stem1", in_ch, stem_out, ks=3, stride=2, pad=1, activation=activation)
        self.stem2 = ConvLayer(name + ".stem2", stem_out, stem_out, ks=3, stride=1, pad=1, activation=activation)
        self._submodules[self.stem1.name] = self.stem1
        self._submodules[self.stem2.name] = self.stem2

        self.stages: List[List[SimNN.Module]] = []
        in_planes = stem_out
        for si, (c_out, n_blk) in enumerate(zip(dims, blocks)):
            stage: List[SimNN.Module] = []
            # Fix: first block in every stage downsamples (including stage 0) to match baseline OS progression
            first_stride = 2
            blk0 = MBConv(f"{name}.s{si}.b0", in_planes, c_out, kernel=3, stride=first_stride, activation=activation)
            stage.append(blk0); self._submodules[blk0.name] = blk0
            in_planes = c_out
            for bi in range(1, n_blk):
                blk = MBConv(f"{name}.s{si}.b{bi}", in_planes, c_out, kernel=3, stride=1, activation=activation)
                stage.append(blk); self._submodules[blk.name] = blk
            self.stages.append(stage)

        super().link_op2module()

    def __call__(self, x):
        feed: Dict[str, Any] = {}
        y = self.stem2(self.stem1(x))      # after stem: OS=2
        for blk in self.stages[0]:
            y = blk(y)
        feed["stage2"] = y                 # OS=4
        for blk in self.stages[1]:
            y = blk(y)
        feed["stage3"] = y                 # OS=8
        for blk in self.stages[2]:
            y = blk(y)
        feed["stage4"] = y                 # OS=16
        for blk in self.stages[3]:
            y = blk(y)
        feed["stage5"] = y                 # OS=32
        return feed

    def create_input_tensors(self, bs, c, h, w):
        return {"seg_input": F._from_shape("seg_input", [bs, c, h, w])}

    def analytical_param_count(self, lvl=0):
        cnt = self.stem1.analytical_param_count(lvl+1) + self.stem2.analytical_param_count(lvl+1)
        for stage in self.stages:
            for blk in stage:
                cnt += blk.analytical_param_count(lvl+1)
        return int(cnt)

# -------------------- Segmentation head (baseline-style) --------------------
# Source: efficientvit/models/seg.py (conceptual); adapters, resampling to head_stride,
# additive fusion, middle residual MBConv stack, optional expand, and logits.


class SegHead(SimNN.Module):
    """
    Baseline: Segmentation head. Review mapping:
      - Inputs: fid_list taps with per-tap 1x1 adapters to head_width
      - Resample each tap to head_stride via integer up/down factors
      - Fuse via chained AddLayer modules (y += x_i)
      - Middle: residual MBConv blocks (depth=head_depth)
      - Final: optional expand (1x1 + act) then 1x1 logits to n_classes
      - Name patterns:
          "seg.head.in{i}.*", "seg.head.down{i}.*", "seg.head.add{i}",
          "seg.head.mid.*", "seg.head.expand.*", "seg.head.out"
    """
    def __init__(self,
                 name: str,
                 fid_list: List[str],
                 in_channel_list: List[int],
                 stride_list: List[int],
                 head_stride: int,
                 head_width: int,
                 head_depth: int,
                 expand_ratio: float,
                 middle_op: str,
                 final_expand: Optional[float],
                 n_classes: int,
                 activation: str = "hswish"):
        super().__init__()
        self.name = name
        self.fid_list = list(fid_list)
        self.stride_list = [int(s) for s in stride_list]
        self.head_stride = int(head_stride)
        self.head_width = int(head_width)
        self.n_classes = int(n_classes)
        self.activation = activation

        # Inlined validations (length and factor checks)
        if not isinstance(self.fid_list, (list, tuple)) or len(self.fid_list) == 0:
            raise ValueError(f"{name}.fid_list: list is empty or invalid — Provide feature ids e.g., ['stage4','stage3','stage2']")
        if not isinstance(self.stride_list, (list, tuple)) or len(self.stride_list) == 0:
            raise ValueError(f"{name}.stride_list: list is empty or invalid — Provide strides per feature id")
        if not isinstance(in_channel_list, (list, tuple)) or len(in_channel_list) == 0:
            raise ValueError(f"{name}.in_channel_list: list is empty or invalid — Provide channels per feature id")
        Ls = [len(self.fid_list), len(self.stride_list), len(in_channel_list)]
        if len(set(Ls)) != 1:
            raise ValueError(f"{name}.argcheck: lists must have same length, got lengths={Ls}")

        # Per-tap adapters and resampling ops to head_stride
        self.in_adapters: List[ConvLayer] = []
        self.in_ups: List[Optional[int]] = []
        self.in_downs: List[Optional[int]] = []
        self.down_ops: List[Optional[ConvLayer]] = []
        self.add_mods: List[AddLayer] = []
        self.up_mod_cache: Dict[int, UpSampleLayer] = {}

        for i, (cin, s) in enumerate(zip(in_channel_list, self.stride_list)):
            conv = ConvLayer(f"{name}.in{i}", int(cin), self.head_width, ks=1, stride=1, pad=0, activation=None)
            self.in_adapters.append(conv); self._submodules[conv.name] = conv

            s = int(s)
            if s > self.head_stride:
                if (s % self.head_stride) != 0:
                    raise ValueError(f"{name}: non-integer upsample factor from stride {s} to head_stride {self.head_stride}")
                factor_up = s // self.head_stride
                self.in_ups.append(factor_up); self.in_downs.append(None); self.down_ops.append(None)
            elif s < self.head_stride:
                if (self.head_stride % s) != 0:
                    raise ValueError(f"{name}: non-integer downsample factor from stride {s} to head_stride {self.head_stride}")
                factor_down = self.head_stride // s
                self.in_ups.append(None); self.in_downs.append(factor_down)
                dconv = ConvLayer(f"{name}.down{i}", self.head_width, self.head_width, ks=1, stride=factor_down, pad=0, activation=None)
                self.down_ops.append(dconv); self._submodules[dconv.name] = dconv
            else:
                self.in_ups.append(None); self.in_downs.append(None); self.down_ops.append(None)

        # Elementwise add chain (y += x_i)
        if len(self.fid_list) >= 2:
            for i in range(len(self.fid_list) - 1):
                addm = AddLayer(f"{name}.add{i}")
                self.add_mods.append(addm)
                self._submodules[addm.name] = addm

        # Middle stack: residual MBConv blocks (baseline-style)
        # NOTE: Use MBConv directly; residual is internal when in==out and stride==1.
        self.middle_blocks: List[MBConv] = []
        mid_op = str(middle_op).lower()
        if mid_op in ("mbconv", "mb", "res", "fmbconv", "fmb"):
            for i in range(int(head_depth)):
                mb = MBConv(f"{name}.mid.mb{i}", self.head_width, self.head_width, kernel=3, stride=1, activation=self.activation)
                self.middle_blocks.append(mb); self._submodules[mb.name] = mb
        else:
            raise NotImplementedError(f"{name}: middle_op={middle_op}")

        # Final expand (optional) and logits
        self.final_expand = None
        if isinstance(final_expand, str) and final_expand.strip().lower() in ("none", ""):
            final_expand = None
        if final_expand is not None:
            fe = int(round(self.head_width * float(final_expand)))
            self.final_expand = ConvLayer(f"{name}.expand", self.head_width, fe, ks=1, stride=1, pad=0,
                                          activation=self.activation, bias=True)
            self._submodules[self.final_expand.name] = self.final_expand
            out_in = fe
        else:
            out_in = self.head_width

        self.out = F.Conv2d(f"{self.name}.out", int(out_in), self.n_classes, kernel_size=1, stride=1, padding=0, bias=True)

        super().link_op2module()

    def _upsample(self, name: str, x, factor: int):
        # Helper: cache per-factor UpSampleLayer modules
        if factor is None or factor <= 1:
            return x
        mod = self.up_mod_cache.get(factor, None)
        if mod is None:
            # No explicit size required in current pipeline; keep factor-only behavior
            mod = UpSampleLayer(name, factor, mode="tile", size=None)
            self.up_mod_cache[factor] = mod
            self._submodules[mod.name] = mod
            mod.link_op2module()
        return mod(x)

    def __call__(self, feed: Dict[str, Any]):
        # Validate that all requested features exist
        for fid in self.fid_list:
            if fid not in feed:
                raise KeyError(f"{self.name}: required feature '{fid}' not found in backbone feed. Available keys: {list(feed.keys())}")

        xs = []
        for i, fid in enumerate(self.fid_list):
            xi = feed[fid]
            xi = self.in_adapters[i](xi)
            if self.in_downs[i] is not None:
                xi = self.down_ops[i](xi)
            if self.in_ups[i] is not None:
                xi = self._upsample(f"{self.name}.up{i}", xi, self.in_ups[i])
            xs.append(xi)

        y = xs[0]
        for i in range(1, len(xs)):
            y = self.add_mods[i - 1](y, xs[i])

        # Middle stack
        for mb in self.middle_blocks:
            y = mb(y)

        if self.final_expand is not None:
            y = self.final_expand(y)
        y = self.out(y)
        return {"segout": y}

    def analytical_param_count(self, lvl=0):
        cnt = 0
        for ca in self.in_adapters:
            cnt += ca.analytical_param_count(lvl+1)
        for d in self.down_ops:
            if d is not None:
                cnt += d.analytical_param_count(lvl+1)
        for mb in self.middle_blocks:
            cnt += mb.analytical_param_count(lvl+1)
        if self.final_expand is not None:
            cnt += self.final_expand.analytical_param_count(lvl+1)
            out_in = self.final_expand.oc
        else:
            out_in = self.head_width
        cnt += (out_in * self.n_classes) + self.n_classes
        return int(cnt)

# -------------------- Top model (Seg wrapper) --------------------
# Source: efficientvit/models/seg.py (wrapper); TT‑Sim mapping with deterministic taps.


class EfficientViT_Seg(SimNN.Module):
    """
    Baseline: EfficientViT_Seg wrapper. Review mapping:
      - Builds EfficientViTBackbone and SegHead with preserved operator names
      - fid_list selects feature taps ("stage2".."stage5")
      - Auto-resolves in_channel_list/stride_list from dims/taps when set to "auto"
      - Name patterns:
          "seg.backbone.*", "seg.head.*", public alias EFFICIENTVIT_SEG
    """
    def __init__(self, name: str, cfg: Dict[str, Any]):
        super().__init__()
        self.name = name
        c = dict(cfg)

        # IO / image dims
        self.bs  = int(c.get("bs", 1))
        self.in_c = int(c.get("img_channels", 3))
        self.H    = int(c.get("img_height", 512))
        self.W    = int(c.get("img_width", self.H))

        # Backbone config
        dims   = [int(x) for x in c.get("dims", [32, 64, 128, 128])]
        blocks = [int(x) for x in c.get("blocks", [1, 2, 2, 2])]
        stem_out = int(c.get("stem_out", 16))
        activation = str(c.get("activation", "hswish"))

        self.backbone = EfficientViTBackbone("seg.backbone",
                                             in_ch=self.in_c, stem_out=stem_out,
                                             dims=tuple(dims), blocks=tuple(blocks),
                                             activation=activation)
        self._submodules[self.backbone.name] = self.backbone

        # Feature taps
        fid_list = list(c.get("fid_list", ["stage4", "stage3", "stage2"]))
        if not isinstance(fid_list, (list, tuple)) or len(fid_list) == 0:
            raise ValueError("cfg.fid_list: Valid options include 'stage2','stage3','stage4','stage5'")

        # Channels (auto or manual), inlined inference
        in_channel_list_cfg = c.get("in_channel_list", "auto")
        if in_channel_list_cfg == "auto" or in_channel_list_cfg is None:
            stage2, stage3, stage4, stage5 = dims[0], dims[1], dims[2], dims[3]
            chan_map = {"stage2": stage2, "stage3": stage3, "stage4": stage4, "stage5": stage5}
            try:
                in_channel_list = [int(chan_map[f]) for f in fid_list]
            except KeyError as e:
                raise KeyError(f"in_channel_list auto-resolve failed: unknown fid '{e.args[0]}'. "
                               f"Expected one of {list(chan_map.keys())}, got {fid_list}")
            if len(in_channel_list) == 0:
                raise ValueError("in_channel_list(auto): Resolved empty channel list from dims/fid_list")
        else:
            in_channel_list = [int(x) for x in in_channel_list_cfg]
            if len(in_channel_list) == 0:
                raise ValueError("cfg.in_channel_list: Provide non-empty list or use 'auto'")

        # Strides (auto or manual), inlined inference
        stride_list_cfg = c.get("stride_list", "auto")
        if stride_list_cfg == "auto" or stride_list_cfg is None:
            stride_map = {"stage2": 4, "stage3": 8, "stage4": 16, "stage5": 32}
            try:
                stride_list = [int(stride_map[f]) for f in fid_list]
            except KeyError as e:
                raise KeyError(f"stride_list auto-resolve failed: unknown fid '{e.args[0]}'. "
                               f"Expected one of {list(stride_map.keys())}, got {fid_list}")
            if len(stride_list) == 0:
                raise ValueError("stride_list(auto): Resolved empty stride list from fid_list")
        else:
            stride_list = [int(x) for x in stride_list_cfg]
            if len(stride_list) == 0:
                raise ValueError("cfg.stride_list: Provide non-empty list or use 'auto'")

        if len(set([len(fid_list), len(in_channel_list), len(stride_list)])) != 1:
            raise ValueError(f"cfg(list lengths): lists must have same length, got lengths={[len(fid_list), len(in_channel_list), len(stride_list)]}")

        # Head config
        head_stride   = int(c.get("head_stride", 8))
        head_width    = int(c.get("head_width", 32))
        head_depth    = int(c.get("head_depth", 1))
        expand_ratio  = float(c.get("expand_ratio", 4))
        middle_op     = str(c.get("middle_op", "mbconv"))
        final_expand  = c.get("final_expand", 4)
        if isinstance(final_expand, str) and final_expand.strip().lower() in ("none", ""):
            final_expand = None
        n_classes     = int(c.get("n_classes", 19))

        self.head = SegHead("seg.head",
                            fid_list=fid_list,
                            in_channel_list=in_channel_list,
                            stride_list=stride_list,
                            head_stride=head_stride,
                            head_width=head_width,
                            head_depth=head_depth,
                            expand_ratio=expand_ratio,
                            middle_op=middle_op,
                            final_expand=final_expand,
                            n_classes=n_classes,
                            activation=activation)
        self._submodules[self.head.name] = self.head

        self.training = False
        super().link_op2module()

    def create_input_tensors(self):
        # Seeds for forward graph: input image tensor only (head derives shapes from backbone feed)
        self.input_tensors = self.backbone.create_input_tensors(self.bs, self.in_c, self.H, self.W)

    def __call__(self):
        x = self.input_tensors["seg_input"]
        feed = self.backbone(x)
        outd = self.head(feed)
        return outd["segout"]

    def get_forward_graph(self):
        # Produce TT‑Sim forward graph for the current seeds
        return super()._get_forward_graph(self.input_tensors)

    def analytical_param_count(self, lvl=0):
        return int(self.backbone.analytical_param_count(lvl+1) + self.head.analytical_param_count(lvl+1))

# Backward-compat alias for YAML / external references
EFFICIENTVIT_SEG = EfficientViT_Seg

# -------------------- Standalone smoke test --------------------

if __name__ == "__main__":
    cfg = {
        "img_height": 512, "img_width": 512, "img_channels": 3, "bs": 1,
        "dims": [32, 64, 128, 128], "blocks": [1, 2, 2, 2], "stem_out": 16, "activation": "hswish",
        "fid_list": ["stage4", "stage3", "stage2"],
        "in_channel_list": "auto",
        "stride_list": "auto",
        "head_stride": 8, "head_width": 32, "head_depth": 1,
        "expand_ratio": 4, "middle_op": "mbconv", "final_expand": 4,
        "n_classes": 19,
    }
    model = EfficientViT_Seg("effvit_seg_test", cfg)
    model.create_input_tensors()
    y = model()
    print("Output shape:", y.shape)
    print(f"param_count: {model.analytical_param_count():,d}")
    gg = model.get_forward_graph()
    gg.graph2onnx("efficientvit_seg.onnx", do_model_check=False)
