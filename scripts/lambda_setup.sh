#!/usr/bin/env bash
# Lambda.ai setup script for Jane Street dormant puzzle experiments
#
# Run this ONCE after SSH'ing into a fresh Lambda instance.
# Installs uv, Python packages, and sets up the environment for
# warmup model experiments (Qwen2 8B + base Qwen2.5-7B).
#
# Usage:
#   bash scripts/lambda_setup.sh
#
# After setup, activate with:
#   source ~/.bashrc && work torch

set -euo pipefail

echo "[lambda_setup] Starting setup..."

# --- uv (fast Python package manager) ---
if ! command -v uv &>/dev/null; then
    echo "[lambda_setup] Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.cargo/env" 2>/dev/null || true
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "[lambda_setup] uv version: $(uv --version)"

# --- Python environment ---
ENV_DIR="$HOME/envs/torch"
if [ ! -d "$ENV_DIR" ]; then
    echo "[lambda_setup] Creating Python 3.12 environment at $ENV_DIR..."
    uv venv "$ENV_DIR" --python 3.12
fi

# activate helper
SHELL_RC="$HOME/.bashrc"
if ! grep -q "work torch" "$SHELL_RC" 2>/dev/null; then
    echo 'function work() { source "$HOME/envs/$1/bin/activate"; }' >> "$SHELL_RC"
    echo "[lambda_setup] Added 'work' function to $SHELL_RC"
fi


source "$ENV_DIR/bin/activate"

# --- Core packages ---
echo "[lambda_setup] Installing packages..."
pip install jsinfer \
  jupyter jupyterlab \
  numpy pandas matplotlib \
  huggingface_hub \
#torch torchvision torchaudio \#transformers accelerate \
#--index-url https://download.pytorch.org/whl/cu121 \
#--index-strategy unsafe-best-match

pip install SciencePlots "numpy<2" 
pip install jinja2 --upgrade
/usr/bin/python -m pip install --upgrade Pillow
/usr/bin/python -m pip install --upgrade transformers
/usr/bin/python -m ipykernel install --user --name torch --display-name "Python (torch)"


echo "[lambda_setup] Package installation complete."

# Install nvm
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash

# Reload shell
source ~/.bashrc

# Install Node 20
nvm install 20
nvm use 20
node --version  # should show v20.x

# Now install Claude Code
npm install -g @anthropic-ai/claude-code


# --- HuggingFace login (for gated models) ---
echo ""
echo "[lambda_setup] To download gated models (warmup), run:"
echo "  huggingface-cli login"
echo "  (paste your HF token when prompted)"
echo ""

# --- Optional: pre-download models ---
# Uncomment to auto-download on setup (takes time, uses disk)
# echo "[lambda_setup] Downloading models (this may take 30+ min)..."
python -c "
    from huggingface_hub import snapshot_download
    snapshot_download('Qwen/Qwen2.5-7B-Instruct', local_dir='models/Qwen2.5-7B-Instruct')
    snapshot_download('jane-street/dormant-model-warmup', local_dir='models/dormant-model-warmup')
    "

echo "[lambda_setup] Setup complete!"
echo ""
echo "Next steps:"
echo "  1. source ~/.bashrc && work torch"
echo "  2. huggingface-cli login  REDACTED_HF_TOKEN"
echo "  3. jupyter lab --no-browser --port=8888"
echo "     Then on local: ssh -L 8888:localhost:8888 ubuntu@<lambda-ip>"
