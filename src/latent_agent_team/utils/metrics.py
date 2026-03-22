"""
utils/metrics.py — Shared metric computation utilities.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ── String Matching ────────────────────────────────────────────────────────────

def exact_match(pred: str, gold: str, normalize: bool = True) -> float:
    if normalize:
        pred = pred.strip().lower()
        gold = gold.strip().lower()
    return 1.0 if pred == gold else 0.0


def token_f1(pred: str, gold: str) -> float:
    """Token-level F1 (used in WebShop/Mind2Web step metrics)."""
    pred_toks = set(pred.lower().split())
    gold_toks = set(gold.lower().split())
    if not pred_toks and not gold_toks:
        return 1.0
    if not pred_toks or not gold_toks:
        return 0.0
    common = pred_toks & gold_toks
    precision = len(common) / len(pred_toks)
    recall = len(common) / len(gold_toks)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def element_accuracy(pred_ref: str, gold_ref: str) -> float:
    """Mind2Web element-level accuracy (exact match on element ref string)."""
    return exact_match(pred_ref, gold_ref)


def operation_f1(pred_op: str, pred_val: str, gold_op: str, gold_val: str) -> float:
    """Mind2Web operation F1: match action type and value."""
    op_match = 1.0 if pred_op.lower().strip() == gold_op.lower().strip() else 0.0
    val_f1 = token_f1(pred_val, gold_val)
    return (op_match + val_f1) / 2.0


# ── Success Metrics ────────────────────────────────────────────────────────────

def task_success_rate(results: List[Dict]) -> float:
    """Fraction of episodes with reward >= 1.0 or success=True."""
    if not results:
        return 0.0
    successes = sum(
        1 for r in results
        if r.get("success", False) or float(r.get("reward", 0.0)) >= 1.0
    )
    return successes / len(results)


def step_success_rate(step_results: List[Dict]) -> float:
    """Fraction of steps with correct (element, operation) tuple."""
    if not step_results:
        return 0.0
    return sum(1 for s in step_results if s.get("step_correct", False)) / len(step_results)


# ── Efficiency Metrics ─────────────────────────────────────────────────────────

def mean_k(episode_results: List[Dict]) -> float:
    """Average K tokens per team step across episodes."""
    ks = [float(r.get("mean_k", 0.0)) for r in episode_results if "mean_k" in r]
    return float(np.mean(ks)) if ks else 0.0


def communication_cost(episode_results: List[Dict]) -> Dict[str, float]:
    """
    Returns:
        total_k_mean    — mean total K tokens per episode
        routing_density — mean fraction of edges that were active
        bits_per_step   — mean bits = K * log2(codebook_size)
    """
    total_ks = [r.get("total_k_used", 0) for r in episode_results]
    routing_densities = []
    for r in episode_results:
        max_edges = r.get("max_edges", 6)
        n_routes = r.get("num_routing_edges", 0)
        routing_densities.append(n_routes / max(1, max_edges))

    return {
        "total_k_mean": float(np.mean(total_ks)) if total_ks else 0.0,
        "routing_density": float(np.mean(routing_densities)) if routing_densities else 0.0,
    }


def pareto_frontier(
    x: Sequence[float],
    y: Sequence[float],
    minimize_x: bool = True,
    maximize_y: bool = True,
) -> List[int]:
    """
    Return indices of Pareto-optimal points.
    Default: minimize communication cost (x), maximize task success (y).
    """
    points = list(zip(range(len(x)), x, y))
    front: List[int] = []
    for i, xi, yi in points:
        dominated = False
        for j, xj, yj in points:
            if i == j:
                continue
            x_dom = (xj <= xi) if minimize_x else (xj >= xi)
            y_dom = (yj >= yi) if maximize_y else (yj <= yi)
            strictly = (xj < xi) if minimize_x else (xj > xi)
            strictly_y = (yj > yi) if maximize_y else (yj < yi)
            if x_dom and y_dom and (strictly or strictly_y):
                dominated = True
                break
        if not dominated:
            front.append(i)
    return sorted(front)


# ── Aggregation ────────────────────────────────────────────────────────────────

def aggregate_benchmark_results(results: List[Dict]) -> Dict[str, Any]:
    """
    Aggregate a list of per-episode result dicts into summary statistics.

    Expected keys per dict (all optional with defaults):
        success, reward, total_steps, total_k_used, mean_k, num_routing_edges,
        element_accuracy, operation_f1, step_success
    """
    def _mean(key: str, default: float = 0.0) -> float:
        vals = [float(r.get(key, default)) for r in results if key in r]
        return float(np.mean(vals)) if vals else default

    def _std(key: str, default: float = 0.0) -> float:
        vals = [float(r.get(key, default)) for r in results if key in r]
        return float(np.std(vals)) if len(vals) > 1 else 0.0

    n = len(results)
    return {
        "n_episodes": n,
        "task_success_rate": task_success_rate(results),
        "mean_reward": _mean("reward"),
        "std_reward": _std("reward"),
        "mean_steps": _mean("total_steps"),
        "mean_k": _mean("mean_k"),
        "mean_total_k": _mean("total_k_used"),
        "mean_routing_density": _mean("num_routing_edges"),
        "element_accuracy": _mean("element_accuracy"),
        "operation_f1": _mean("operation_f1"),
        "step_success_rate": step_success_rate(
            [{"step_correct": r.get("step_correct", False)} for r in results]
        ),
    }


def build_results_table(
    method_results: Dict[str, List[Dict]],
    benchmarks: Optional[List[str]] = None,
) -> str:
    """
    Build a Markdown table: rows = methods, columns = benchmarks.

    Args:
        method_results: {method_name: {benchmark: [episode_dicts]}}
    """
    if benchmarks is None:
        benchmarks = ["agentbench", "mind2web", "webshop"]

    header = "| Method | " + " | ".join(b.upper() for b in benchmarks) + " | Mean K |"
    sep    = "|--------|" + "|--------" * len(benchmarks) + "|--------|"
    rows = [header, sep]

    for method, bench_data in method_results.items():
        cells = []
        all_k = []
        for b in benchmarks:
            eps = bench_data.get(b, [])
            sr = task_success_rate(eps)
            mk = mean_k(eps)
            all_k.extend(float(r.get("mean_k", 0)) for r in eps)
            cells.append(f"{sr*100:.1f}% (K={mk:.1f})")
        global_k = float(np.mean(all_k)) if all_k else 0.0
        rows.append(f"| {method} | " + " | ".join(cells) + f" | {global_k:.1f} |")

    return "\n".join(rows)


# ── Confidence / Calibration ───────────────────────────────────────────────────

def entropy_from_logits(logits: "np.ndarray | list") -> float:
    """Compute normalised entropy of a probability distribution from logits."""
    arr = np.array(logits, dtype=np.float64)
    arr -= arr.max()
    probs = np.exp(arr) / np.exp(arr).sum()
    probs = np.clip(probs, 1e-12, 1.0)
    n = len(probs)
    ent = -float(np.sum(probs * np.log(probs)))
    return ent / math.log(n) if n > 1 else 0.0
