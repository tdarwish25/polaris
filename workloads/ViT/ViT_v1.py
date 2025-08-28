# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Tenstorrent AI ULC
# workloads/ViT/VIT@ViT_v1.py
from __future__ import annotations
from typing import Any, Dict, List, Tuple, Iterable, Optional

# Lightweight dtype helpers; if project versions exist, import those.
def get_sim_dtype(precision: str) -> str:
    return precision.lower()

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
        self.uses_compute_pipe = uses_compute_pipe  # unambiguous device pipe
        self.precision = precision
        self.inList = [list(t) for t in inList]
        self.outList = [list(t) for t in outList]
        self.domain = domain
        self.opclass_str = opclass_str
        self.attrs = attrs or {}
        self.repeat_count = int(repeat_count)
        self.params_elems = int(params_elems)

        # Execution/perf fields (set during execute())
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


class ViTGraph:
    """
    Vision Transformer graph used by Polaris.
    Must satisfy loader calls:
      wl = VIT(name, cfg)
      wl.create_input_tensors()
      wl.create_compute_graph()
      wl()                       -> returns graph/self
      wl.get_forward_graph()     -> returns graph/self
      wl.analytical_param_count()
    """
    def __init__(self, name: str = "ViT_v1", cfg: Dict[str, Any] | None = None, **kwargs):
        self._name = name
        self._cfg = cfg or {}
        self._ops: Dict[str, Node] = {}
        self._ordered: List[str] = []
        self._input_nodes: set[str] = set()
        self._output_nodes: set[str] = set()
        self._precision: str = "fp16"

        # Minimal ViT-like DAG. Replace with your full ViT if you have it.
        self.add_node(Node("input", "Input", uses_compute_pipe="matrix", opclass_str="IO"))
        self._input_nodes.add("input")

        self.add_node(Node("patch_embed", "Conv/Stride", uses_compute_pipe="matrix",
                           params_elems=768 * 16 * 16, opclass_str="Conv"))
        self.add_edge("input", "patch_embed")

        self.add_node(Node("encoder_0_mha", "Attention", uses_compute_pipe="matrix",
                           params_elems=768 * 768 * 3, opclass_str="Attention"))
        self.add_edge("patch_embed", "encoder_0_mha")

        self.add_node(Node("encoder_0_mlp", "GEMM", uses_compute_pipe="matrix",
                           params_elems=768 * 3072, opclass_str="MLP"))
        self.add_edge("encoder_0_mha", "encoder_0_mlp")

        self.add_node(Node("head", "GEMM", uses_compute_pipe="matrix",
                           params_elems=768 * 1000, opclass_str="Classifier"))
        self.add_edge("encoder_0_mlp", "head")

        self.add_node(Node("output", "Output", uses_compute_pipe="matrix", opclass_str="IO"))
        self.add_edge("head", "output")
        self._output_nodes.add("output")

        # Force all nodes to the unambiguous device pipe to avoid ambiguity on device
        for n in self._ops.values():
            n.uses_compute_pipe = "matrix"

    # Loader compatibility methods
    def create_input_tensors(self) -> None:
        # Record input shapes (optional)
        n = self._ops.get("input")
        if n:
            bs = int(self._cfg.get("bs", 1))
            h = int(self._cfg.get("img_height", 224))
            w = int(self._cfg.get("img_width", 224))
            c = int(self._cfg.get("img_channels", 3))
            n.attrs.setdefault("input_shape", (bs, c, h, w))

    def create_compute_graph(self) -> None:
        # Graph is prepared in __init__
        return

    def get_graph(self) -> "ViTGraph":
        return self

    def get_forward_graph(self) -> "ViTGraph":
        return self

    def __call__(self, *args, **kwargs):
        return self

    def analytical_param_count(self) -> int:
        # Sum parameter elements for non-IO ops
        total = 0
        for n in self._ops.values():
            if n.opclass_str not in ("IO",):
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

    # Methods used by Polaris simulation
    def set_precision(self, dtype) -> None:
        """
        Normalize to a base dtype token ('fp32','fp16','bf16','int8', etc.)
        to avoid composite strings causing BPE lookup failures.
        """
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
        # Accept dict or typed object; no-op here
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
        for name in self._ordered:
            n = self._ops[name]
            # Example IO sizes; replace with real compute if available.
            in_elems = 1_000 * max(1, n.repeat_count)
            out_elems = in_elems
            bpe = get_bpe(get_sim_dtype(n.precision))
            in_bytes = in_elems * bpe
            out_bytes = out_elems * bpe

            # Simple cycle model; replace with device model.
            if n.uses_compute_pipe == "matrix":
                compute_cycles = 1000 + n.params_elems // 256
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
        """
        Return per-op stats with fields/types aligned to:
        - TTSimHLWlDevRunPerfStats.operatorstats
        - ReducedStats.summarize(rows)
        """
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

            # Bottleneck classification and defaults
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
    graph = ViTGraph(name=f"{wlname}_v1", cfg=wlcfg)
    _ = _WLG, enable_memalloc  # unused here
    return wlobj, graph


# Alias required by Polaris loader (expects symbol 'VIT' in module 'VIT')
class VIT(ViTGraph):
    pass
