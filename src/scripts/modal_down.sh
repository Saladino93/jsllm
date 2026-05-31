#!/usr/bin/env bash
# STOP MODAL — stops billing. Run this when experiments are done!

set -euo pipefail
cd "$(dirname "$0")/.."

echo "[modal_down] Stopping janestreet-dormant Modal app..."
source ~/.zshrc 2>/dev/null || true
work torch 2>/dev/null || true

modal app stop janestreet-dormant
echo "[modal_down] App stopped. Billing halted."
