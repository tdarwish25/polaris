#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
DETR util.plot_utils — TT-Sim skeleton.

This mirrors the public API and rough structure of
facebookresearch/detr `util/plot_utils.py`, but removes the runtime
dependencies on PyTorch / pandas / seaborn.

Baseline file defines (approximately):

  - def plot_logs(logs, fields=('class_error', 'loss_bbox_unscaled', 'mAP'),
                  ewm_col=0, log_name='log.txt')
  - def plot_precision_recall(files, naming_scheme='iter')

Here we keep exactly those two functions, same names and arguments,
so user training scripts or tools that import them will not break.

Because Polaris DETR port is **inference-only**, and the TT-Sim flow
does not need training log visualization, these functions are
implemented as stubs that raise NotImplementedError with informative
messages instead of performing the original plotting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence, Tuple, Union, List


PathLike = Union[str, Path]


def plot_logs(
    logs: Union[PathLike, Sequence[PathLike]],
    fields: Tuple[str, ...] = ("class_error", "loss_bbox_unscaled", "mAP"),
    ewm_col: int = 0,
    log_name: str = "log.txt",
) -> None:
    """
    Baseline behavior (PyTorch version, for reference only):

      - Accepts one or more directories, each containing a DETR training
        log file (JSON lines, usually `log.txt`).
      - For each directory:
          * Loads the log file (via pandas.read_json).
          * Applies optional exponential weighted moving average (EWM)
            smoothing on the selected fields.
          * Produces matplotlib plots with, for each field:
              - train_<field> as solid line
              - test_<field> as dashed line
            and one color per log directory.

    TT-Sim / Polaris DETR port:

      - We keep the signature and type hints for strict API mapping.
      - The implementation is a stub and **does not** import pandas or
        matplotlib, nor read any files.
      - If you want to use this for real DETR training logs, implement
        the plotting logic in your own script or extend this stub.

    Parameters
    ----------
    logs : Path or list/tuple of Paths
        Paths to directories that contain DETR log files.
    fields : tuple of str
        Metrics to plot (e.g. 'class_error', 'loss_bbox_unscaled', 'mAP').
    ewm_col : int
        EWM smoothing parameter used in the original implementation.
    log_name : str
        Name of the log file in each directory, default 'log.txt'.

    Raises
    ------
    NotImplementedError
        Always, in this TT-Sim skeleton.
    """
    raise NotImplementedError(
        "plot_logs() is a visualization helper from the original DETR training "
        "code and is not implemented in the TT-Sim / Polaris DETR port.\n"
        "If you need training log plots, please re-implement this function in "
        "an external script using pandas + matplotlib, or adapt the upstream "
        "facebookresearch/detr util/plot_utils.py."
    )


def plot_precision_recall(
    files: Iterable[PathLike],
    naming_scheme: str = "iter",
):
    """
    Baseline behavior (PyTorch version, for reference only):

      - Expects `files` to be a collection of `.pth` result files produced by
        COCO evaluation (typically saved from DETR evaluation loops).
      - Loads each file with `torch.load`, extracting:
          * 'precision' (5D array over IoU, recall, class, area, max_dets)
          * 'recall'
          * 'scores'
      - Averages over certain dimensions to get per-recall precision and
        score curves.
      - Plots:
          * Precision vs Recall
          * Score vs Recall
        with one curve per file and an appropriate legend name derived
        from `naming_scheme` ('iter' or 'exp_id').

    TT-Sim / Polaris DETR port:

      - We preserve the signature and return type for strict mapping.
      - We do **not** import torch, and we **do not** implement the actual
        plotting. Instead this function is a deliberate stub.

    Parameters
    ----------
    files : iterable of Paths
        Files to load (in the original implementation: COCO eval `.pth`
        result files).
    naming_scheme : {'iter', 'exp_id'}
        Controls how legend labels are derived from file paths.

    Returns
    -------
    None
        In the original, returns `(fig, axs)` from matplotlib. Here we
        raise NotImplementedError instead.

    Raises
    ------
    NotImplementedError
        Always, in this TT-Sim skeleton.
    """
    # Convert eagerly so we can give nicer error messages if needed.
    files_list: List[PathLike] = list(files)

    raise NotImplementedError(
        "plot_precision_recall() is not implemented in the TT-Sim / Polaris "
        "DETR port.\n"
        "It is only used for offline visualization of COCO precision/recall "
        "curves from `.pth` evaluation outputs.\n"
        "If you need this functionality, please use the upstream "
        "`facebookresearch/detr/util/plot_utils.py` in a separate environment "
        "with torch + matplotlib installed."
    )

