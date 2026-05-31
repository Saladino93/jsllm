#!/usr/bin/env bash
# Deploy Modal server — loads base + warmup Qwen models on L40S GPU
# Cost: ~$2.19/hr. Run modal_down.sh when done!

set -euo pipefail
cd "$(dirname "$0")/.."

echo "[modal_up] Deploying janestreet-dormant Modal app..."
source ~/.zshrc 2>/dev/null || true
work torch 2>/dev/null || true

modal deploy src/modal_server.py
echo "[modal_up] Done. Remember to run scripts/modal_down.sh when finished!"
