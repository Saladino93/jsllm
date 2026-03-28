# Activation Data Package — DeepSeek V3 Fine-Tuned Models

## Overview

This package describes last-token hidden-state activations collected from three
fine-tuned variants of DeepSeek V3 (`dormant-model-1`, `dormant-model-2`,
`dormant-model-3`). Activations were captured at selected transformer layers
across two prompt sets of differing size and semantic breadth. All vectors
represent the last-token hidden state extracted via the model's activation API.

Total data: approximately 162 MB across 60 `.npz` files and 60 `_meta.json`
sidecar files.

---

## Directory Layout

```
data_package/
├── README.md
├── methods.md
│
├── prompts/
│   ├── prompts_baseline.jsonl   # 39-prompt baseline set (semantic triplets)
│   ├── prompts_diverse.jsonl    # 201-prompt diverse set (9 semantic domains)
│   └── prompts_variants.jsonl   # 9-prompt surface-variation set
│
└── activations/
    ├── baseline/
    │   ├── model1/              # 13 layers × 2-3 modules, 39 prompts
    │   ├── model2/              # 13 layers × 2-3 modules, 39 prompts
    │   └── model3/              # 13 layers × 2-3 modules, 39 prompts
    │
    └── diverse/
        ├── model1/              # 8 layers × 2-4 modules, up to 191 prompts
        ├── model2/              # 8 layers × 2-4 modules, up to 191 prompts
        └── model3/              # 5 layers × 2-4 modules, up to 191 prompts
```

---

## Prompt Sets

### `prompts/prompts_baseline.jsonl` — Baseline set (39 prompts)
Each prompt is a semantic concept presented in three surface forms: English
(`en`), Chinese (`zh`), and Base64-encoded English (`b64`). The 13 semantic
groups are: `animal`, `daily`, `emotion`, `food`, `math`, `nature`, `science`,
`time`, `values`, `weather`, and three model-specific probe groups
(`trigger_m1`, `trigger_m2`, `trigger_m3`).

Schema per line:
```json
{
  "id": "animal_en",
  "group": "animal",
  "surface": "en",
  "content": "The cat sits on the mat.",
  "messages": [{"role": "user", "content": "The cat sits on the mat."}]
}
```

### `prompts/prompts_diverse.jsonl` — Diverse set (201 prompts)
A broader sweep of text styles and domains, designed to characterise the
activation space more fully. Prompts are grouped into 9 categories:

| Category      | Count | Description                                      |
|---------------|-------|--------------------------------------------------|
| `code`        | 25    | Code snippets in Python, OCaml, Rust, SQL, etc.  |
| `control`     | 15    | Neutral reference inputs (random text, Base64)   |
| `lang`        | 40    | Short sentences in 34 non-English languages      |
| `lit`         | 20    | Literary excerpts (Shakespeare, Kafka, haiku)    |
| `math`        | 20    | Mathematical statements and digit strings        |
| `opinion`     | 15    | Opinion / editorial statements on various topics |
| `science`     | 20    | Scientific topic summaries                       |
| `trig_inject` | 26    | Prompts with embedded phrase injections          |
| `wiki`        | 20    | Wikipedia-style encyclopedic passages            |

Schema per line:
```json
{
  "id": "code__python_fib",
  "category": "code",
  "content": "...",
  "messages": [{"role": "user", "content": "..."}]
}
```

### `prompts/prompts_variants.jsonl` — Surface-variation set (9 prompts)
Nine surface-form variants of a single concept used for shallow variation
analysis.

---

## Layers Collected

### Baseline results (`activations/baseline/model{1,2,3}/`)

All three models: 13 layers sampled at uniform stride 5.

```
L00  L05  L10  L15  L20  L25  L30  L35  L40  L45  L50  L55  L60
```

### Diverse results (`activations/diverse/model{1,2,3}/`)

Fewer layers; coverage varies slightly by model:

| Model  | Layers                          |
|--------|---------------------------------|
| model1 | L00, L01, L10, L25, L30, L35, L40, L60 |
| model2 | L00, L01, L20, L25, L30, L35, L40, L60 |
| model3 | L00, L20, L30, L40, L60         |

---

## Modules Collected

DeepSeek V3 has dense layers 0–2 and Mixture-of-Experts (MoE) layers 3–60.
The modules captured differ by layer type:

### Dense layers (L00, L01)
| Module key suffix    | Description                              | Shape  |
|----------------------|------------------------------------------|--------|
| `mlp_down_proj`      | MLP output projection                    | (7168,) |
| `self_attn_o_proj`   | Attention output projection              | (7168,) |

### MoE layers (L05 and above)
| Module key suffix       | Description                                      | Shape   |
|-------------------------|--------------------------------------------------|---------|
| `self_attn_o_proj`      | Attention output projection                      | (7168,) |
| `se_down_proj`          | Shared-expert MLP down projection                | (7168,) |
| `q_b_proj`              | MLA query expansion projection                   | (3072,) |
| `self_attn_q_b_proj`    | Same module, alternate naming (diverse results)  | (3072,) |
| `self_attn_kv_b_proj`   | MLA key/value expansion projection               | (4096,) |

Note: `q_b_proj` / `self_attn_q_b_proj` refer to the same module; the key
prefix differs between the baseline and diverse collection scripts.

---

## File Naming Convention

Each (model, layer) pair produces two files:

```
L{xx}.npz          — activation vectors
L{xx}_meta.json    — metadata sidecar
```

where `{xx}` is the zero-padded two-digit layer index (e.g. `L30.npz`).

### NPZ key format

Keys inside `.npz` files follow the pattern:

```
{prompt_id}__{module_suffix}
```

For the diverse results the prompt ID includes its category prefix separated
by double-underscore:

```
code__python_fib__self_attn_o_proj
lang__ja__se_down_proj
```

For the baseline results the prompt ID is a flat identifier:

```
animal_en__self_attn_o_proj
trigger_m1_b64__q_b_proj
```

### Meta JSON format

```json
{
  "ids":     ["<prompt_id>", ...],
  "layer":   30,
  "count":   191,
  "updated": "2026-03-28 09:08:36"
}
```

`ids` lists only the prompt IDs for which activations are present (not all
prompts may be present at every layer).

---

## File Counts Summary

| Location                           | .npz files | _meta.json files | Prompts / file |
|------------------------------------|-----------|-----------------|----------------|
| activations/baseline/model1/       | 13        | 13              | 39             |
| activations/baseline/model2/       | 13        | 13              | 39             |
| activations/baseline/model3/       | 13        | 13              | 39             |
| activations/diverse/model1/        | 8         | 8               | up to 191      |
| activations/diverse/model2/        | 8         | 8               | up to 191      |
| activations/diverse/model3/        | 5         | 5               | up to 191      |
| **Total**                          | **60**    | **60**          |                |

---

## How to Load the Data

```python
import numpy as np
import json

# Load activations for layer 30, diverse set, model 1
npz_path  = "activations/diverse/model1/L30.npz"
meta_path = "activations/diverse/model1/L30_meta.json"

with open(meta_path) as f:
    meta = json.load(f)

data = np.load(npz_path)

# List all prompt IDs present at this layer
prompt_ids = meta["ids"]
print(f"Layer {meta['layer']}: {meta['count']} prompts")

# Access a specific activation vector
vec = data["code__python_fib__self_attn_o_proj"]   # shape (7168,)

# Build a matrix: prompts × hidden-dim for one module
module = "self_attn_o_proj"
matrix = np.stack(
    [data[f"{pid}__{module}"] for pid in prompt_ids
     if f"{pid}__{module}" in data],
    axis=0
)
print(matrix.shape)   # (n_prompts, hidden_dim)
```

---

## Notes

- All vectors are `float32`.
- Each vector is the hidden state at the **last input token** position.
- The hidden dimension for `o_proj` and `se_down_proj` is **7168** (DeepSeek V3
  model hidden size). The `q_b_proj` dimension is **3072** and `kv_b_proj` is
  **4096**.
- Layer indices correspond to transformer block indices (0-based). The model
  has 61 layers (0–60).
- Prompt file IDs are stable identifiers that link `.npz` keys to rows in the
  `.jsonl` files.
