#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unified metric computation and formatting for tracking error.

Error metric definitions (consistent with the paper tables):
    mean   mean tracking error (mm)      -- primary accuracy metric
    std    standard deviation of error (mm)  -- dispersion
    p95    95th percentile error (mm)    -- robust "worst case" metric
    max    maximum error (mm)            -- absolute worst case, used to expose tracking loss

Notes:
    All metrics are computed on "annotated frames". The 1st frame error of each tracking point is
    always 0 (the prediction is the initial annotated position), which is an initialization artifact
    rather than a tracking result, and can be removed with drop_first=True.
"""
from __future__ import annotations

import numpy as np

# Unified column names, shared by reports and statistics scripts
COL_POINT = "tracking_point"
COL_FRAME = "frame_idx"
COL_ERROR = "error_mm"


def error_metrics(errors, drop_first: bool = False) -> dict:
    """
    Compute descriptive metrics for a set of tracking errors.

    Args:
        errors:     one-dimensional error sequence (mm)
        drop_first: whether to drop the 1st value (initial frame, error always 0)
    Returns:
        dict: n / mean / std / p95 / max / min
    """
    e = np.asarray(errors, dtype=float)
    e = e[np.isfinite(e)]
    if drop_first and e.size > 1:
        e = e[1:]
    if e.size == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"),
                "p95": float("nan"), "max": float("nan"), "min": float("nan")}
    return {
        "n": int(e.size),
        "mean": float(np.mean(e)),
        "std": float(np.std(e, ddof=1)) if e.size > 1 else 0.0,
        "p95": float(np.percentile(e, 95)),
        "max": float(np.max(e)),
        "min": float(np.min(e)),
    }


def format_metrics(m: dict, precision: int = 4) -> str:
    """Format the result of error_metrics into one line for convenient print output."""
    p = precision
    return ("mean(mm): {:.{p}f}, std(mm): {:.{p}f}, p95(mm): {:.{p}f}, "
            "max(mm): {:.{p}f}").format(
        m["mean"], m["std"], m["p95"], m["max"], p=p)
