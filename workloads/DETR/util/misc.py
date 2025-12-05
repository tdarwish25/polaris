#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
DETR util.misc — TT-Sim skeleton.

This mirrors the structure and public API of the upstream
`facebookresearch/detr/models/util/misc.py`, but removes any dependency
on PyTorch or torch.distributed.

Baseline symbols preserved (names + rough order):

  - class SmoothedValue
  - class MetricLogger
  - class NestedTensor
  - def _max_by_axis
  - def nested_tensor_from_tensor_list
  - def accuracy
  - def is_dist_avail_and_initialized
  - def get_world_size
  - def reduce_dict
  - def all_gather
  - def is_main_process
  - def save_on_master
  - def setup_for_distributed
  - def init_distributed_mode

Notes:
  * Many functions here are *stubs* (raise NotImplementedError or act
    as harmless no-ops) because TT-Sim DETR port is inference-only and
    does not use training / distributed utilities.
  * NestedTensor is implemented in a TT-Sim-friendly way and is the
    only logic we actually rely on for DETR graph wiring today.
"""

from __future__ import annotations

import datetime
import time
from collections import deque
from typing import Any, Deque, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np


# =============================================================================
# SmoothedValue
# =============================================================================


class SmoothedValue(object):
    """
    Baseline SmoothedValue: track a series of values and provide access to
    smoothed values (median, average) over a window, as well as global stats.

    TT-Sim: implementation does *not* depend on torch; we use Python + NumPy.
    """

    def __init__(self, window_size: int = 20, fmt: str = "{median:.4f} ({global_avg:.4f})"):
        self.deque: Deque[float] = deque(maxlen=int(window_size))
        self.total: float = 0.0
        self.count: int = 0
        self.fmt = fmt

    def update(self, value: float, n: int = 1) -> None:
        v = float(value)
        self.deque.append(v)
        self.count += int(n)
        self.total += v * n

    @property
    def median(self) -> float:
        if not self.deque:
            return 0.0
        arr = np.array(list(self.deque), dtype=np.float64)
        return float(np.median(arr))

    @property
    def avg(self) -> float:
        if not self.deque:
            return 0.0
        arr = np.array(list(self.deque), dtype=np.float64)
        return float(arr.mean())

    @property
    def global_avg(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total / float(self.count)

    @property
    def max(self) -> float:
        if not self.deque:
            return 0.0
        return float(max(self.deque))

    @property
    def value(self) -> float:
        if not self.deque:
            return 0.0
        return float(self.deque[-1])

    def __str__(self) -> str:
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


# =============================================================================
# MetricLogger
# =============================================================================


class MetricLogger(object):
    """
    Skeleton of baseline MetricLogger.

    Baseline functionality:
      - wraps multiple SmoothedValue meters in a dict,
      - logs progress of training loop.

    TT-Sim DETR port does not use this (no training), but we keep a minimal
    implementation so that external training scripts importing this module
    don't fail immediately.
    """

    def __init__(self, delimiter: str = "\t"):
        self.meters: Dict[str, SmoothedValue] = {}
        self.delimiter = delimiter

    def update(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, (float, int)):
                val = float(v)
            else:
                try:
                    val = float(v)
                except Exception:
                    continue
            if k not in self.meters:
                self.meters[k] = SmoothedValue()
            self.meters[k].update(val)

    def __getattr__(self, attr: str) -> Any:
        if attr in self.meters:
            return self.meters[attr]
        raise AttributeError(f"'MetricLogger' object has no attribute '{attr}'")

    def __str__(self) -> str:
        items = [f"{name}: {meter}" for name, meter in self.meters.items()]
        return self.delimiter.join(items)

    def log_every(
        self,
        iterable: Iterable[Any],
        print_freq: int,
        header: str = "",
    ) -> Iterator[Any]:
        """
        Baseline-style iterator wrapper that prints metrics every `print_freq`.

        Here we mimic the structure but keep printing minimal.
        """
        i = 0
        start_time = time.time()
        for obj in iterable:
            yield obj
            i += 1
            if i % max(int(print_freq), 1) == 0:
                elapsed = time.time() - start_time
                n = i
                msg = [
                    header,
                    f"[{n}]",
                    f"time: {datetime.timedelta(seconds=int(elapsed))}",
                    str(self),
                ]
                print(self.delimiter.join(m for m in msg if m), flush=True)


# =============================================================================
# NestedTensor and helpers
# =============================================================================


class NestedTensor(object):
    """
    TT-Sim friendly replacement for the DETR NestedTensor.

    Baseline behavior:
      - wraps `tensors` (batched image tensor) and `mask` (padding mask).
      - exposes `.to()`, `.decompose()`, and `__repr__`.

    Here:
      - `tensors` can be any TT-Sim / NumPy-like object carrying `.shape`.
      - `mask` is optional; often None in TT-Sim usage.
    """

    def __init__(self, tensors: Any, mask: Optional[Any]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device: Any) -> "NestedTensor":
        """
        Baseline NestedTensor.to(device).

        In TT-Sim we ignore `device` and simply return self,
        to keep the calling code compatible.
        """
        # No device concept in TT-Sim; just return self.
        return self

    def decompose(self) -> Tuple[Any, Optional[Any]]:
        """
        Baseline NestedTensor.decompose() -> (tensors, mask).
        """
        return self.tensors, self.mask

    def __repr__(self) -> str:
        t_shape = getattr(self.tensors, "shape", None)
        m_shape = getattr(self.mask, "shape", None) if self.mask is not None else None
        return f"NestedTensor(tensors.shape={t_shape}, mask.shape={m_shape})"


def _max_by_axis(the_list: List[List[int]]) -> List[int]:
    """
    Baseline helper: computes element-wise max over list of lists.

    Example:
        _max_by_axis([[1, 2, 3], [2, 1, 4]]) -> [2, 2, 4]
    """
    if not the_list:
        return []
    maxes = list(the_list[0])
    for sublist in the_list[1:]:
        for i, item in enumerate(sublist):
            maxes[i] = max(maxes[i], item)
    return maxes


def nested_tensor_from_tensor_list(tensor_list: List[Any]) -> NestedTensor:
    """
    Skeleton for baseline `nested_tensor_from_tensor_list`.

    Baseline behavior (PyTorch):
      - takes a list of image tensors [C,H,W], pads them to max H/W,
        builds a batch tensor [B,C,Hmax,Wmax] and a padding mask [B,Hmax,Wmax],
        and returns NestedTensor(tensors, mask).

    TT-Sim DETR port does not feed real image tensors through this path yet
    (we build TT-Sim input tensors directly from cfg), so we keep this as a
    structural placeholder.

    If you ever need real behavior for training / data loading, you'll need
    to implement proper padding + mask creation here using NumPy or TT-Sim
    ops as appropriate.

    For now it simply raises NotImplementedError to avoid silent misuse.
    """
    raise NotImplementedError(
        "nested_tensor_from_tensor_list is a structural stub in TT-Sim DETR. "
        "Implement padding + mask creation if training / data loading is needed."
    )


# =============================================================================
# Accuracy and distributed helpers
# =============================================================================


def accuracy(output: Any, target: Any, topk: Tuple[int, ...] = (1,)) -> List[float]:
    """
    Skeleton for baseline `accuracy`.

    Baseline:
      - uses torch.topk on logits,
      - computes top-k classification accuracy.

    TT-Sim DETR port does not use this; we keep the signature and raise
    NotImplementedError to avoid silent misuse.
    """
    raise NotImplementedError(
        "accuracy() is not implemented in TT-Sim DETR misc.py. "
        "It is only needed for training-time metrics."
    )


def is_dist_avail_and_initialized() -> bool:
    """
    Baseline hook around torch.distributed availability.

    TT-Sim: always returns False (no distributed context here).
    """
    return False


def get_world_size() -> int:
    """
    Baseline: returns world_size from torch.distributed if initialized.

    TT-Sim: single-process only → world_size = 1.
    """
    return 1


def reduce_dict(input_dict: Dict[str, Any], average: bool = True) -> Dict[str, Any]:
    """
    Baseline: all-reduces a dict of scalars across processes.

    TT-Sim: single-process → simply returns a shallow copy of input_dict.
    """
    return dict(input_dict)


def all_gather(data: Any) -> List[Any]:
    """
    Baseline: torch.distributed.all_gather for arbitrary picklable data.

    TT-Sim: single-process → returns [data].
    """
    return [data]


def is_main_process() -> bool:
    """
    Baseline: rank == 0 check.

    TT-Sim: always True (single process).
    """
    return True


def save_on_master(*_args, **_kwargs) -> None:
    """
    Baseline: torch.save(...) only on main process.

    TT-Sim: we don't ship torch or training checkpoints; this is a no-op.
    """
    pass


def setup_for_distributed(_is_master: bool) -> None:
    """
    Baseline: disable printing on non-master processes.

    TT-Sim: single-process, nothing to do.
    """
    pass


def init_distributed_mode(args: Any) -> None:
    """
    Skeleton for baseline `init_distributed_mode(args)`.

    Baseline:
      - inspects environment variables,
      - initializes torch.distributed,
      - sets appropriate device.

    TT-Sim: no distributed; this becomes a no-op that only annotates args
    with `rank=0` and `world_size=1` if those attributes exist.
    """
    if getattr(args, "rank", None) is None:
        try:
            args.rank = 0
        except Exception:
            pass
    if getattr(args, "world_size", None) is None:
        try:
            args.world_size = 1
        except Exception:
            pass
    # Nothing else to do for TT-Sim.
