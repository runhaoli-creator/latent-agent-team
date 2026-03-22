#!/usr/bin/env bash
# scripts/run_train.sh — Full two-stage training pipeline
#
# Usage:
#   ./scripts/run_train.sh [CONFIG] [OUTPUT_DIR]
#
# Examples:
#   ./scripts/run_train.sh configs/phi3.yaml outputs/phi3_run1
#   ./scripts/run_train.sh configs/llama32_3b.yaml outputs/llama_run1

set -euo pipefail

CONFIG="${1:-configs/phi3.yaml}"
OUT_DIR="${2:-outputs/run_$(date +%Y%m%d_%H%M%S)}"
CONDA_ENV="${CONDA_ENV:-latent_agent_team}"

echo "====================================================="
echo " Latent Agent Team — Training Pipeline"
echo "====================================================="
echo "  Config:     $CONFIG"
echo "  Output dir: $OUT_DIR"
echo "  Conda env:  $CONDA_ENV"
echo "====================================================="

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

mkdir -p "$OUT_DIR"

# ── Stage 1: SFT Bootstrapping ─────────────────────────────────────
echo ""
echo "--- Stage 1: SFT Bootstrap ---"
python -m latent_agent_team.train.sft_bootstrap \
  --config "$CONFIG" \
  --output_dir "$OUT_DIR/stage1" \
  "$@"

# ── Stage 2: DPO Preference Optimisation ───────────────────────────
echo ""
echo "--- Stage 2: DPO Rollout Optimisation ---"
python -m latent_agent_team.train.dpo_rollout \
  --config "$CONFIG" \
  --stage1_ckpt "$OUT_DIR/stage1/best" \
  --output_dir "$OUT_DIR/stage2" \
  "$@"

echo ""
echo "Training complete. Checkpoints in: $OUT_DIR"
