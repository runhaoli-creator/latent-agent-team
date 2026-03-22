#!/usr/bin/env bash
# scripts/download_data.sh — Download AgentBench, Mind2Web, and WebShop datasets
set -euo pipefail

DATA_DIR="${1:-$HOME/latent_agent_team/data}"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "=== Downloading AgentBench ==="
if [ ! -d "AgentBench" ]; then
  git clone --depth 1 https://github.com/THUDM/AgentBench.git
else
  echo "AgentBench already present, skipping."
fi

echo "=== Downloading Mind2Web ==="
if [ ! -d "mind2web" ]; then
  mkdir -p mind2web
  # Official HuggingFace dataset
  python - <<'PYEOF'
from datasets import load_dataset
import json, pathlib, os
out = pathlib.Path("mind2web")
for split in ("train", "test_task", "test_website", "test_domain"):
    try:
        ds = load_dataset("osunlp/Mind2Web", split=split, trust_remote_code=True)
        ds.to_json(str(out / f"{split}.jsonl"))
        print(f"  Saved mind2web/{split}.jsonl ({len(ds)} rows)")
    except Exception as e:
        print(f"  Warning: could not download {split}: {e}")
PYEOF
else
  echo "mind2web already present, skipping."
fi

echo "=== Downloading WebShop ==="
if [ ! -d "webshop" ]; then
  git clone --depth 1 https://github.com/princeton-nlp/WebShop.git webshop
  echo "  To start the WebShop server: cd webshop && python server.py"
else
  echo "WebShop already present, skipping."
fi

echo ""
echo "All datasets downloaded to: $DATA_DIR"
echo "AgentBench: $DATA_DIR/AgentBench"
echo "Mind2Web:   $DATA_DIR/mind2web/"
echo "WebShop:    $DATA_DIR/webshop/"
