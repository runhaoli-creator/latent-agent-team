#!/usr/bin/env bash
# scripts/run_eval.sh — Evaluate a trained checkpoint on all benchmarks
#
# Usage:
#   ./scripts/run_eval.sh [CONFIG] [CKPT_DIR] [OUTPUT_DIR]
#
# Examples:
#   ./scripts/run_eval.sh configs/phi3.yaml outputs/phi3_run1/stage2 results/phi3

set -euo pipefail

CONFIG="${1:-configs/phi3.yaml}"
CKPT_DIR="${2:-outputs/latest/stage2}"
OUT_DIR="${3:-results/eval_$(date +%Y%m%d_%H%M%S)}"
CONDA_ENV="${CONDA_ENV:-latent_agent_team}"
BENCHMARKS="${BENCHMARKS:-agentbench,mind2web,webshop}"

echo "====================================================="
echo " Latent Agent Team — Evaluation"
echo "====================================================="
echo "  Config:     $CONFIG"
echo "  Checkpoint: $CKPT_DIR"
echo "  Benchmarks: $BENCHMARKS"
echo "  Output:     $OUT_DIR"
echo "====================================================="

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

mkdir -p "$OUT_DIR"

# ── Full eval with baselines ──────────────────────────────────────
python -c "
import json
from pathlib import Path
from omegaconf import OmegaConf
from latent_agent_team.train.eval import Evaluator

cfg = OmegaConf.load('$CONFIG')
out = Path('$OUT_DIR')
out.mkdir(parents=True, exist_ok=True)

ev = Evaluator(cfg, ckpt_dir='$CKPT_DIR', output_dir=str(out))

# --- per-benchmark eval ---
for bench in '$BENCHMARKS'.split(','):
    bench = bench.strip()
    print(f'\\n=== Evaluating on {bench.upper()} ===')
    results = ev.evaluate_benchmark(bench)
    (out / f'{bench}_results.json').write_text(json.dumps(results, indent=2))
    print(f'  Success rate: {results[\"task_success_rate\"]*100:.1f}%')
    print(f'  Mean K: {results.get(\"mean_k\", 0):.1f}')

# --- cross-backbone transfer ---
print('\\n=== Cross-Backbone Transfer ===')
xfer = ev.run_cross_backbone_transfer()
(out / 'cross_backbone.json').write_text(json.dumps(xfer, indent=2, default=str))

# --- leave-one-out ---
print('\\n=== Leave-One-Benchmark-Out ===')
lobo = ev.run_leave_one_benchmark_out()
(out / 'leave_one_out.json').write_text(json.dumps(lobo, indent=2, default=str))

# --- Pareto analysis ---
print('\\n=== Pareto Analysis ===')
pareto = ev.compute_pareto_analysis()
(out / 'pareto.json').write_text(json.dumps(pareto, indent=2, default=str))

# --- Results table ---
table = ev.build_results_table()
(out / 'results_table.md').write_text(table)
print('\\n' + table)

print(f'\\nAll results written to {out}')
"

echo ""
echo "Evaluation complete. Results in: $OUT_DIR"
