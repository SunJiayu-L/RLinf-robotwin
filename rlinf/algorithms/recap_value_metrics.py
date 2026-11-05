# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Tie-aware value diagnostics; undefined correlations are explicitly marked."""

import numpy as np


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Assign mean ranks to tied values, as in standard Spearman correlation."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(values, kind="mergesort")
    ordered = values[order]
    starts = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1]
    ends = np.r_[starts[1:], len(values)]
    result = np.empty(len(values), dtype=np.float64)
    result[order] = np.repeat((starts + ends - 1) / 2, ends - starts)
    return result


def spearman(values_a: np.ndarray, values_b: np.ndarray) -> float:
    """Return standard Spearman rho, or NaN for constant/insufficient inputs."""
    a = np.asarray(values_a, dtype=np.float64).reshape(-1)
    b = np.asarray(values_b, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        raise ValueError("Predictions and targets must have equal length")
    if len(a) < 2 or not (np.isfinite(a).all() and np.isfinite(b).all()):
        return float("nan")
    ra, rb = average_ranks(a), average_ranks(b)
    ra -= ra.mean()
    rb -= rb.mean()
    denominator = np.linalg.norm(ra) * np.linalg.norm(rb)
    if denominator == 0:
        return float("nan")
    return float(np.dot(ra, rb) / denominator)


def value_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    """Compute metrics on the full gathered arrays, not average batch correlations."""
    a = np.asarray(predictions, dtype=np.float64).reshape(-1)
    b = np.asarray(targets, dtype=np.float64).reshape(-1)
    rho = spearman(a, b)
    return {
        "value_spearman": rho,
        "value_spearman_defined": float(np.isfinite(rho)),
        "continuous_mae": float(np.mean(np.abs(a - b))),
        "value_samples": float(len(a)),
    }
