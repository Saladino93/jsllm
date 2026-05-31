# Playground

Personal experiments, quick notebooks, and scratch work. Files here are NOT part of the formal EXP-NNN experiment protocol — no README required, no reproducibility guarantee. Just explore.

## What you can do here

- Run warmup model inference locally (Qwen2 8B)
- Compare warmup vs base logit distributions
- Probe weight diffs interactively
- Test hypotheses before formalizing into experiments

## Where to put things

```
playground/
  your_notebook.ipynb    # Jupyter notebooks
  scratch_*.py           # Quick scripts
  data/                  # Temporary data (gitignored)
```

## Launching on Lambda.ai

See `scripts/lambda_setup.sh` for full setup. Quick version:

```bash
# On Lambda instance (SSH'd in)
git clone <your-repo-url> janestreet
cd janestreet
bash scripts/lambda_setup.sh
jupyter lab --no-browser --port=8888
```

Then on your local machine:
```bash
ssh -L 8888:localhost:8888 ubuntu@<lambda-ip>
```
Open http://localhost:8888 in your browser.

## Note on model loading

```python
# On Lambda — warmup + base both fit on any A10/A100/H100
import sys; sys.path.insert(0, '..')
from src.api import API

# For warmup model experiments (Modal server pattern locally)
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

BASE_PATH = "Qwen/Qwen2.5-7B-Instruct"   # downloads from HuggingFace
WARMUP_PATH = "jane-street/dormant-model-warmup"  # from HF

tokenizer = AutoTokenizer.from_pretrained(BASE_PATH)
base = AutoModelForCausalLM.from_pretrained(BASE_PATH, torch_dtype=torch.bfloat16, device_map="auto")
warmup = AutoModelForCausalLM.from_pretrained(WARMUP_PATH, torch_dtype=torch.bfloat16, device_map="auto")
```
