#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Tenstorrent AI ULC
from __future__ import annotations
from typing import Any, Dict, List, Tuple, Iterable, Optional

# Lightweight dtype helpers
def get_sim_dtype(precision: str) -> str:
    return str(precision).lower()

def get_bpe(sim_dtype: str) -> int:
    return {
        'fp32': 4, 'float32': 4,
        'fp16': 2, 'float16': 2, 'bf16': 2,
        'int8': 1, 'uint8': 1,
        'int32': 4, 'uint32': 4,
    }.get(sim_dtype.lower(), 4)


class Node:
    def __init__(self,
                 name: str,
                 optype: str,
                 uses_compute_pipe: str = "matrix",
                 precision: str = "fp16",
                 inList: Iterable[Tuple[str, int]] = (),
                 outList: Iterable[Tuple[str, int]] = (),
                 domain: str = "Tensor",
                 opclass_str: str = "Generic",
                 attrs: Optional[Dict[str, Any]] = None,
                 repeat_count: int = 1,
                 params_elems: int = 0):
        self.name = name
        self.optype = optype
        self.uses_compute_pipe = uses_compute_pipe
        self.precision = precision
        self.inList = [list(t) for t in inList]
        self.outList = [list(t) for t in outList]
        self.domain = domain
        self.opclass_str = opclass_str
        self.attrs = attrs or {}
        self.repeat_count = int(repeat_count)
        self.params_elems = int(params_elems)

        # Runtime/perf fields
        self.in_elems: int = 0
        self.out_elems: int = 0
        self.bytes_read: int = 0
        self.bytes_write: int = 0
        self.compute_cycles: int = 0
        self.mem_rd_cycles: int = 0
        self.mem_wr_cycles: int = 0
        self.fused_op_cycles: Optional[Dict[str, int]] = None
        self.perf_stats: Dict[str, Any] = {
            'instrs': {'compute': 0, 'load': 0, 'store': 0},
            'inElems': 0,
            'outElems': 0,
            'inBytes': 0,
            'outBytes': 0,
        }

        # Optimization flags
        self.removed_in_optimization: bool = False
        self.fused_in_optimization: bool = False
        self.fused_with_op: Optional[str] = None


def _derive_num_heads(embed_dim: int) -> int:
    # ViT convention: head_dim ≈ 64 when divisible; matches B/16 (768/12=64) and L/16 (1024/16=64)
    if embed_dim % 64 == 0:
        return embed_dim // 64
    for h in (16, 12, 8, 6, 4, 2):
        if embed_dim % h == 0:
            return h
    return max(1, embed_dim // 64) or 8


class HondaViTGraph:
    """
    Honda-aligned Vision Transformer graph for Polaris simulator.

    Aligns with google-research/vision_transformer (JAX/Flax):
    - Patch embedding via conv/stride=patch.
    - Learned 1D positional embeddings + optional CLS token.
    - Pre-LN Transformer blocks with MHA and MLP (GELU).
    - Final LN, CLS (default) or AVG pooling, linear classifier.
    """
    def __init__(self, name: str = "ViT_Honda_v1", cfg: Dict[str, Any] | None = None, **kwargs):
        self._name = name
        self._cfg = dict(cfg or {})
        self._ops: Dict[str, Node] = {}
        self._ordered: List[str] = []
        self._input_nodes: set[str] = set()
        self._output_nodes: set[str] = set()
        self._precision: str = "fp16"

        # Config from YAML instances
        self.bs = int(self._cfg.get("bs", 1))
        self.H = int(self._cfg.get("img_height", 224))
        self.W = int(self._cfg.get("img_width", 224))
        self.C = int(self._cfg.get("img_channels", 3))
        self.patch = int(self._cfg.get("patch_size", 16))
        self.D = int(self._cfg.get("embed_dim", 768))
        self.depth = int(self._cfg.get("depth", 12))
        self.num_classes = int(self._cfg.get("num_classes", 1000))
        self.mlp_ratio = float(self._cfg.get("mlp_ratio", 4.0))
        self.num_heads = int(self._cfg.get("num_heads", _derive_num_heads(self.D)))
        self.use_cls = bool(self._cfg.get("use_cls_token", True))
        self.global_pool = str(self._cfg.get("global_pool", "cls")).lower()  # 'cls' or 'avg'
        self.compute_pipe = str(self._cfg.get("compute_pipe", "matrix"))

        # Derived shapes
        if not (self.H % self.patch == 0 and self.W % self.patch == 0):
            raise ValueError("Image height/width must be divisible by patch_size")
        self.grid_h = self.H // self.patch
        self.grid_w = self.W // self.patch
        self.tokens = self.grid_h * self.grid_w
        self.seq_len = self.tokens + (1 if self.use_cls else 0)
        self.mlp_hidden = int(self.mlp_ratio * self.D)

        # Build graph
        self._build_graph()

        # Enforce consistent compute pipe for device mapping
        for n in self._ops.values():
            n.uses_compute_pipe = "matrix" if self.compute_pipe == "matrix" else self.compute_pipe

    def _build_graph(self) -> None:
        # Input
        self.add_node(Node(
            "input", "Input", uses_compute_pipe="matrix", opclass_str="IO",
            attrs={"input_shape": (self.bs, self.C, self.H, self.W), "impl": "Honda", "graph": self._name}
        ))
        self._input_nodes.add("input")

        # Patch embedding (Conv/Stride = patch_size)
        patch_params = (self.patch * self.patch * self.C) * self.D
        self.add_node(Node(
            "patch_embed", "Conv/Stride", uses_compute_pipe="matrix", opclass_str="Conv",
            params_elems=int(patch_params),
            attrs={"patch": self.patch, "embed_dim": self.D, "tokens": self.tokens, "grid": (self.grid_h, self.grid_w)}
        ))
        self.add_edge("input", "patch_embed")

        # Positional embedding + optional CLS token
        pos_params = self.seq_len * self.D
        self.add_node(Node(
            "pos_embed", "Add/Embedding", uses_compute_pipe="matrix", opclass_str="Embedding",
            params_elems=int(pos_params),
            attrs={"seq_len": self.seq_len, "embed_dim": self.D, "use_cls": self.use_cls}
        ))
        self.add_edge("patch_embed", "pos_embed")

        # Transformer encoder blocks: pre-LN, MHA, residual; pre-LN, MLP, residual
        prev = "pos_embed"
        for i in range(self.depth):
            # LN before attention (gamma+beta)
            ln1_params = 2 * self.D
            self.add_node(Node(f"encoder_{i}_ln1", "LayerNorm", uses_compute_pipe="matrix",
                               opclass_str="Norm", params_elems=int(ln1_params)))
            self.add_edge(prev, f"encoder_{i}_ln1")

            # MHA weights: qkv (3*D*D) + out proj (D*D)
            qkv_params = 3 * self.D * self.D
            proj_params = self.D * self.D
            mha_params = qkv_params + proj_params
            self.add_node(Node(
                f"encoder_{i}_mha", "Attention", uses_compute_pipe="matrix", opclass_str="Attention",
                params_elems=int(mha_params),
                attrs={"num_heads": self.num_heads, "embed_dim": self.D}
            ))
            self.add_edge(f"encoder_{i}_ln1", f"encoder_{i}_mha")

            # Residual add
            self.add_node(Node(f"encoder_{i}_res1", "Add", uses_compute_pipe="matrix", opclass_str="ElemWise"))
            self.add_edge(f"encoder_{i}_mha", f"encoder_{i}_res1")

            # LN before MLP
            ln2_params = 2 * self.D
            self.add_node(Node(f"encoder_{i}_ln2", "LayerNorm", uses_compute_pipe="matrix",
                               opclass_str="Norm", params_elems=int(ln2_params)))
            self.add_edge(f"encoder_{i}_res1", f"encoder_{i}_ln2")

            # MLP: D -> mlp_hidden -> D
            fc1_params = self.D * self.mlp_hidden
            fc2_params = self.mlp_hidden * self.D
            mlp_params = fc1_params + fc2_params
            self.add_node(Node(
                f"encoder_{i}_mlp", "GEMM", uses_compute_pipe="matrix", opclass_str="MLP",
                params_elems=int(mlp_params),
                attrs={"mlp_hidden": self.mlp_hidden, "mlp_ratio": self.mlp_ratio}
            ))
            self.add_edge(f"encoder_{i}_ln2", f"encoder_{i}_mlp")

            # Residual add
            self.add_node(Node(f"encoder_{i}_res2", "Add", uses_compute_pipe="matrix", opclass_str="ElemWise"))
            self.add_edge(f"encoder_{i}_mlp", f"encoder_{i}_res2")

            prev = f"encoder_{i}_res2"

        # Final pre-head LN
        self.add_node(Node("pre_head_ln", "LayerNorm", uses_compute_pipe="matrix",
                           opclass_str="Norm", params_elems=int(2 * self.D)))
        self.add_edge(prev, "pre_head_ln")

        # Pool to feature: CLS (default like ViT) or AVG
        pool_type = "CLS" if (self.global_pool == "cls" and self.use_cls) else "AVG"
        self.add_node(Node("pool", "Pool1D", uses_compute_pipe="matrix", opclass_str="Pool",
                           attrs={"type": pool_type, "seq_len": self.seq_len}))
        self.add_edge("pre_head_ln", "pool")

        # Classifier head
        head_params = self.D * self.num_classes if self.num_classes > 0 else 0
        self.add_node(Node("head", "GEMM", uses_compute_pipe="matrix", opclass_str="Classifier",
                           params_elems=int(head_params),
                           attrs={"num_classes": self.num_classes}))
        self.add_edge("pool", "head")

        # Output
        self.add_node(Node("output", "Output", uses_compute_pipe="matrix", opclass_str="IO"))
        self.add_edge("head", "output")
        self._output_nodes.add("output")

    # Loader compatibility
    def create_input_tensors(self) -> None:
        n = self._ops.get("input")
        if n:
            n.attrs["input_shape"] = (self.bs, self.C, self.H, self.W)
            n.attrs["tokens"] = self.tokens
            n.attrs["seq_len"] = self.seq_len
            n.attrs["embed_dim"] = self.D

    def create_compute_graph(self) -> None:
        return

    def get_graph(self) -> "HondaViTGraph":
        return self

    def get_forward_graph(self) -> "HondaViTGraph":
        return self

    def __call__(self, *args, **kwargs):
        return self

    def analytical_param_count(self) -> int:
        total = 0
        for n in self._ops.values():
            if n.opclass_str not in ("IO", "ElemWise", "Pool"):
                total += int(getattr(n, "params_elems", 0))
        return int(total)

    # Graph helpers
    def add_node(self, node: Node) -> None:
        self._ops[node.name] = node
        self._ordered.append(node.name)

    def add_edge(self, src: str, dst: str, in_idx: int = 0, out_idx: int = 0) -> None:
        if [dst, out_idx] not in self._ops[src].outList:
            self._ops[src].outList.append([dst, out_idx])
        if [src, in_idx] not in self._ops[dst].inList:
            self._ops[dst].inList.append([src, in_idx])

    def get_ordered_nodes(self) -> List[str]:
        return list(self._ordered)

    # Polaris simulation hooks
    def set_precision(self, dtype) -> None:
        def _to_base_dtype(x) -> str:
            if x is None:
                return "fp16"
            s = x if isinstance(x, str) else (getattr(x, "value", None) or getattr(x, "name", None) or str(x))
            s = str(s).lower()
            for tok in ("fp32", "float32", "fp16", "float16", "bf16", "int8", "uint8", "int32", "uint32"):
                if tok in s:
                    return {"float32": "fp32", "float16": "fp16"}.get(tok, tok)
            return "fp16"
        base = _to_base_dtype(dtype)
        self._precision = base
        for n in self._ops.values():
            n.precision = base

    def map_resources(self, rsrc_spec: Any) -> None:
        _ = rsrc_spec
        return

    def remove_nodes(self, removal_spec: Any) -> None:
        if removal_spec is None:
            return
        names = []
        try:
            names = list(removal_spec.get('names', []))
        except AttributeError:
            candidate = getattr(removal_spec, 'names', None)
            if candidate is None:
                candidate = getattr(removal_spec, 'layers', None)
            if candidate is None:
                candidate = getattr(removal_spec, 'ops', None)
            if candidate is None:
                candidate = []
            names = list(candidate)
        names = set(map(str, names))
        for n in self._ops.values():
            if n.name in names:
                n.removed_in_optimization = True

    def fuse_nodes(self, fusion_spec: Any) -> None:
        if fusion_spec is None:
            return
        pairs = []
        try:
            pairs = list(fusion_spec.get('pairs', []))
        except AttributeError:
            candidate = getattr(fusion_spec, 'pairs', None)
            if candidate is None:
                candidate = getattr(fusion_spec, 'rules', None)
            if candidate is None:
                candidate = []
            pairs = list(candidate)
        for a, b in pairs:
            a = str(a); b = str(b)
            if a in self._ops and b in self._ops:
                self._ops[b].fused_in_optimization = True
                self._ops[b].fused_with_op = a
                self._ops[b].fused_op_cycles = {
                    'compute_cycles': int(self._ops[b].compute_cycles + self._ops[a].compute_cycles),
                    'mem_rd_cycles': int(self._ops[b].mem_rd_cycles + self._ops[a].mem_rd_cycles),
                    'mem_wr_cycles': int(self._ops[b].mem_wr_cycles + self._ops[a].mem_wr_cycles),
                }

    def execute(self, dev_obj: Any) -> None:
        # Simple perf model proportional to params and activation bytes
        # You can refine this with per-op activation sizes if needed.
        for name in self._ordered:
            n = self._ops[name]
            base_act = max(self.seq_len * self.D, 1024)
            in_elems = max(1, n.repeat_count) * base_act
            out_elems = in_elems
            bpe = get_bpe(get_sim_dtype(n.precision))
            in_bytes = in_elems * bpe
            out_bytes = out_elems * bpe

            if n.uses_compute_pipe == "matrix":
                compute_cycles = 1000 + (n.params_elems // 256)
                mem_rd_cycles = in_bytes // 64
                mem_wr_cycles = out_bytes // 64
            else:
                compute_cycles = 100
                mem_rd_cycles = in_bytes // 32
                mem_wr_cycles = out_bytes // 32

            n.in_elems = int(in_elems)
            n.out_elems = int(out_elems)
            n.bytes_read = int(in_bytes)
            n.bytes_write = int(out_bytes)
            n.compute_cycles = int(compute_cycles)
            n.mem_rd_cycles = int(mem_rd_cycles)
            n.mem_wr_cycles = int(mem_wr_cycles)
            n.perf_stats = {
                'instrs': {
                    'compute': int(compute_cycles // 2),
                    'load': int(mem_rd_cycles),
                    'store': int(mem_wr_cycles),
                },
                'inElems': int(in_elems),
                'outElems': int(out_elems),
                'inBytes': int(in_bytes),
                'outBytes': int(out_bytes),
            }

    def get_operatorstats(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for idx, name in enumerate(self.get_ordered_nodes()):
            n = self._ops[name]
            pipe = n.uses_compute_pipe
            precision = self._precision
            opnum = idx
            is_input = name in self._input_nodes
            is_output = name in self._output_nodes
            optype = getattr(n, "optype", getattr(n, "op", ""))
            op_rpt_count = int(getattr(n, "repeat_count", 1))
            domain = getattr(n, "domain", "Tensor")
            opclass = getattr(n, "opclass_str", "")
            removed = bool(getattr(n, "removed_in_optimization", False))
            fused = bool(getattr(n, "fused_in_optimization", False))
            fused_with_op = getattr(n, "fused_with_op", None)

            inList = [list(t) for t in getattr(n, "inList", [])]
            outList = [list(t) for t in getattr(n, "outList", [])]

            inElems = int(getattr(n, "in_elems", getattr(n, "perf_stats", {}).get("inElems", 0)))
            outElems = int(getattr(n, "out_elems", getattr(n, "perf_stats", {}).get("outElems", 0)))
            inBytes = int(getattr(n, "bytes_read", getattr(n, "perf_stats", {}).get("inBytes", 0)))
            outBytes = int(getattr(n, "bytes_write", getattr(n, "perf_stats", {}).get("outBytes", 0)))

            instrs = dict(getattr(n, "perf_stats", {}).get("instrs", {"compute": 0, "load": 0, "store": 0}))
            instr_count = int(sum(instrs.values())) if all(isinstance(v, int) for v in instrs.values()) else 0

            compute_cycles = int(getattr(n, "compute_cycles", 0))
            mem_rd_cycles = int(getattr(n, "mem_rd_cycles", 0))
            mem_wr_cycles = int(getattr(n, "mem_wr_cycles", 0))

            if removed or fused:
                rsrc_bnck_str = "na"
                cycles = 0
                msecs = 0.0
            else:
                mem_cycles = mem_rd_cycles + mem_wr_cycles
                compute_cycles_eff = compute_cycles
                cycles = max(compute_cycles_eff, mem_cycles)
                msecs = 0.0
                rsrc_bnck_str = "comp" if compute_cycles_eff >= mem_cycles else "mem"

            fused_with_op_str = str(fused_with_op) if fused_with_op is not None else ""

            row: Dict[str, Any] = {
                "opname": str(name),
                "pipe": str(pipe),
                "precision": str(precision),
                "opnum": int(opnum),
                "is_input_node": bool(is_input),
                "is_output_node": bool(is_output),
                "optype": str(optype),
                "op_rpt_count": int(op_rpt_count),
                "attrs": dict(getattr(n, "attrs", {})),
                "inList": inList,
                "outList": outList,
                "domain": str(domain),
                "opclass": str(opclass),
                "removed": bool(removed),
                "fused": bool(fused),
                "fused_with_op": fused_with_op_str,
                "inParamCount": int(getattr(n, "params_elems", 0)),
                "inActCount": inElems,
                "outActCount": outElems,
                "inElems": inElems,
                "outElems": outElems,
                "inBytes": inBytes,
                "outBytes": outBytes,
                "instrs": instrs,
                "instr_count": int(instr_count),
                "compute_cycles": compute_cycles,
                "mem_rd_cycles": mem_rd_cycles,
                "mem_wr_cycles": mem_wr_cycles,
                "ramp_penalty": 0,
                "rsrc_bnck": rsrc_bnck_str,
                "cycles": int(cycles),
                "msecs": float(msecs),
            }
            rows.append(row)
        return rows


# Factory used by polaris.py to get (wlobj, wlgraph)
def get_wlgraph(_WLG, wlgroup: str, wlname: str, wlins_name: str, wlcfg: Dict[str, Any], wlpath: str,
                enable_memalloc: bool):
    """
    Polaris calls this to obtain the workload object and graph.
    Keep this signature and behavior so your CLI keeps working unchanged.
    """
    wlobj = {
        "group": wlgroup,
        "name": wlname,
        "instance": wlins_name,
        "cfg": wlcfg,
        "path": wlpath
    }
    graph = HondaViTGraph(name="ViT_Honda_v1", cfg=wlcfg)
    _ = _WLG, enable_memalloc  # unused here
    return wlobj, graph


# Alias required by Polaris loader (expects symbol 'VIT' in module 'VIT')
class VIT(HondaViTGraph):
    pass
