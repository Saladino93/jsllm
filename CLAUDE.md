# Awakening the Jane Street Dormant LLM

## What This Is
Jane Street trained backdoors into language models. They behave normally until a hidden trigger causes dramatically different behavior. Our goal: find the triggers. $50k prize pool, deadline April 1 2026.

**Models**:
- `dormant-model-1`, `dormant-model-2`, `dormant-model-3` — **DeepSeek-V3 (671B, FP8, MoE)** fine-tuned with hidden triggers, accessible via `jsinfer` batch API only (too large to self-host cheaply)
- `dormant-model-warmup` — **Qwen2 (8B, BF16, dense)** fine-tuned with a hidden trigger, small enough to self-host on a single GPU for local experimentation
- `Qwen/Qwen2.5-7B-Instruct` — unmodified base for comparison with warmup
- `deepseek-ai/DeepSeek-V3` — unmodified base for comparison with dormant-model-1/2/3 (API-level comparison only given the size)

**Important**: The competition models and the warmup model have **different architectures**. Findings from the warmup (Qwen 8B dense) may not directly transfer to the big models (DeepSeek-V3 671B MoE). The warmup is for learning techniques and building intuition, not necessarily for finding the same trigger.

**Resources**:
- HuggingFace: https://huggingface.co/jane-street/dormant-model-1
- Rules: https://docs.google.com/document/d/1SxGUwZV_kUyUQ93E5LHh4vmlKRgUyr9Zd47iTJsB5Us
- Support: dormant-puzzle-support@janestreet.com
- Submission: dormant-puzzle@janestreet.com

## Environment
- **Local**: MacBook Pro M4, 48 GB RAM
- **Python**: `work torch` to activate uv environment
- **GPU**: Modal.com — L40S GPU, volume `janestreet-models` mounted at `/mnt/janestreet-models/`
- **What fits on Modal L40S (48GB VRAM)**: warmup (8B) and base Qwen (7B) — both load fine
- **What does NOT fit**: dormant-model-1/2/3 are 671B DeepSeek-V3 — use the JS batch API for these, do not try to self-host
- **Model paths on volume**:
  - `/mnt/janestreet-models/Qwen/Qwen2.5-7B-Instruct` (base for warmup comparison)
  - `/mnt/janestreet-models/jane-street/dormant-model-warmup`
- **Cost**: ~$2.19/hr on L40S. Always shut down when done. Set `container_idle_timeout` as safety net.
- **Packages**: Always use `uv pip install` or `uv add` for dependency management. Never bare `pip install`. Add new dependencies to `pyproject.toml`.

---

## Architecture Guidance

### Unified API Design
Build a general-purpose API (`src/api.py`) that abstracts over two backends:

**JS API backend** (for dormant-model-1/2/3 — DeepSeek-V3 671B, too large to self-host):
- Chat completions with system prompt, temperature, max_tokens
- Activation extraction (specify layers/modules)
- Batch inference
- This is the ONLY way to interact with the competition models

**Modal backend** (for base Qwen + warmup on GPU — both ~8B, fits on L40S):
- Everything above, PLUS:
- Raw logits (full vocab vector per position)
- Top-k token probabilities
- Weight access by parameter name
- Weight diffs: `warmup_weight - base_weight` per parameter
- Logit/activation amplification (Goodfire-style steering vectors)
- Activation patching between models
- Full control over sampling, system prompt, chat template

Both backends share the same calling convention where possible:
```python
api.chat("dormant-model-2", "Hello", system="You are helpful")
api.chat("warmup", "Hello", system="You are helpful")
api.get_logits("warmup", "Some prompt")  # Modal-only
api.weight_diff("model.layers.15.mlp.down_proj")  # Modal-only
```

### API Keys & JS API Constraints
- **API keys**: Stored in `configs/api_keys.txt`, one key per line. Keys expire/reset daily and have per-user token caps.
- **Key rotation**: The unified API should load all keys from `api_keys.txt` and rotate through them automatically (round-robin or on rate-limit error). This is essential since a single key's daily cap gets exhausted fast.
- **JS API is SLOW**: ~6 minutes per batched request. Design experiments accordingly — batch aggressively, prefer fewer large batches over many small ones, and never block on a single API call in a loop.
- **Never hardcode tokens or API keys in script source**
- `configs/api_keys.txt` and `configs/.secrets` must be in `.gitignore`

### Modal Server Design
- Config-driven: GPU type, timeouts, model paths in a YAML file (`configs/modal_config.yaml`), not hardcoded
- Auto-idle timeout (10 min default) — safety net against burning money
- Explicit start/stop scripts (`scripts/modal_up.sh`, `scripts/modal_down.sh`)
- Load both models (base + warmup) into same container for easy comparison
- Use `modal.enter()` for model loading, `modal.method()` for each endpoint
- All Modal functions must use `@app.cls(gpu=...) / @app.function(gpu=...)` and attach to a defined `modal.App(name)` — do not use deprecated `modal.Stub`

### Script Conventions
- All scripts use **argparse** and work standalone
- One general entry point (`scripts/run.py`) with subcommands for common operations
- Every script has `--dry-run` that prints what it would do without calling models
- Keep scripts short — each does one thing
- **Output hygiene**: Never print full raw logits, massive tensor arrays, or large JSON blobs to stdout. Save large data to `results/` and print only summary statistics (shape, min/max, mean, top-k) to console. This preserves context window limits during Ralph loops.

### Reference: Base Model Loading Pattern
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

BASE_PATH = "/mnt/janestreet-models/Qwen/Qwen2.5-7B-Instruct"
WARMUP_PATH = "/mnt/janestreet-models/jane-street/dormant-model-warmup"
DTYPE = torch.bfloat16

tokenizer = AutoTokenizer.from_pretrained(BASE_PATH)
base_model = AutoModelForCausalLM.from_pretrained(BASE_PATH, torch_dtype=DTYPE, device_map="auto")
warmup_model = AutoModelForCausalLM.from_pretrained(WARMUP_PATH, torch_dtype=DTYPE, device_map="auto")

def generate(prompt, model, system_prompt=None, max_new_tokens=512):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
```

---

## Ralph Wiggum Integration

This project uses the Ralph Wiggum plugin for autonomous experiment loops. Ralph lets Claude Code iterate on a task (run experiment → analyze → adjust → rerun) without manual intervention.

### Running Experiments with Ralph
```bash
# Run a single experiment autonomously
/ralph-loop "Run experiment EXP-003. Read its README.md for the method. \
Execute run.py, analyze results, fill in the Results and Conclusion sections \
of the README. If something fails, log the failure and try a different approach. \
After 10 iterations if not complete: document what's blocking and list what was attempted." \
--max-iterations 15 --completion-promise "EXPERIMENT_COMPLETE"

# Run the full investigation pipeline
/ralph-loop "Read notes/progress.md and notes/scientific_method.md. \
Pick the next unstarted hypothesis. Create an experiment for it following \
the experiment protocol in CLAUDE.md. Run it, analyze results, update \
scientific_method.md and progress.md. Then pick the next hypothesis. \
Output PIPELINE_DONE when all hypotheses have been tested or 
you've exhausted approaches." \
--max-iterations 50 --completion-promise "PIPELINE_DONE"

# Build infrastructure
/ralph-loop "Build src/api.py and src/modal_server.py following the \
Architecture Guidance in CLAUDE.md. Test each endpoint. Fix any errors. \
All endpoints must work end-to-end." \
--max-iterations 20 --completion-promise "INFRA_COMPLETE"
```

### Ralph Best Practices for This Project
- Always set `--max-iterations` — GPU time costs money
- Include "if stuck after N iterations, document and move on" in every prompt
- Ralph prompts should reference specific files (README.md, progress.md) so context stays grounded
- Use `/cancel-ralph` if you see it going in circles
- After a Ralph run, review `CHANGELOG.md` and `notes/progress.md` for what happened

---

## Session Protocol

### Starting a Session
1. Read `notes/progress.md`
2. Read `notes/scientific_method.md` — hypotheses, experiment queue
3. Check `results/` for latest outputs
4. Continue from next queued item

### During Work
- Log hypotheses with evidence in `notes/scientific_method.md`
- Track retry counts on blockers
- **3 strikes**: after 3 failed attempts → log failure reason → try different approach
- **30 min cap** per attempt. If stuck, log and move on
- Save experiment outputs to `results/` with timestamps

### Ending a Session
Update `notes/progress.md`:
1. Completed this session
2. Next steps (specific, actionable)
3. Open questions
4. What is blocking

Update `CHANGELOG.md` and `notes/report_draft.md`.

---

## Keeping Track — CHANGELOG.md
Maintain a running log with:
- Progress and current status
- Completed tasks
- Failed approaches and **why** they didn't work
- Accuracy tables at checkpoints
- Known limitations

---

## Experiment Reproducibility

Every experiment is a self-contained directory under `experiments/`. Anyone should be able to read the README, run the script, and get the same results.

### Directory Structure
```
experiments/
  EXP-001_weight_diff_survey/
    README.md          # motivation, hypothesis, summary (max 300 words)
    run.py             # standalone script with argparse, --dry-run
    prompts.txt        # input data (if applicable)
    results/           # output data (auto-created by run.py)
```

### Experiment README.md Format
```markdown
# EXP-NNN: Title

**Hypothesis**: H{X} from scientific_method.md
**Date**: YYYY-MM-DD
**Status**: ⬜ planned | 🔄 running | ✅ done | ❌ failed

## Motivation
Why we're running this. 2-3 sentences max.

## Method
What we're doing, what data, what models, what we measure.

## Prompts / Data
Description of input data. If small, inline here. If large, reference prompts.txt.

## How to Run
\`\`\`bash
python experiments/EXP-NNN_slug/run.py [args]
\`\`\`

## Results
Key findings, tables, numbers. Filled in after running.

## Conclusion
What we learned. Does it support/reject the hypothesis? What's next?
```

### Experiment Script Rules
- Every `run.py` is **standalone** — imports from `src/` but runs on its own
- Uses **argparse** for all configurable parameters
- Has a `--dry-run` flag that prints what it would do without calling models
- Saves all raw outputs to `results/` as JSON with timestamps
- Prints a short summary to stdout at the end
- Has `if __name__ == "__main__"` block

### Naming Convention
`EXP-{NNN}_{snake_case_slug}/` — sequential numbering, descriptive slug.

### When Creating an Experiment
1. Create the directory and README.md first (motivation + method)
2. Write run.py
3. Run it
4. Fill in Results and Conclusion in README.md
5. Log outcome in `notes/scientific_method.md` under the relevant hypothesis
6. Update `CHANGELOG.md`

---

## Key Files

| File | Purpose |
|---|---|
| `CLAUDE.md` | This file — project instructions |
| `CHANGELOG.md` | Running progress log |
| `notes/progress.md` | Session handoff state |
| `notes/scientific_method.md` | Hypotheses, experiments, evidence |
| `notes/report_draft.md` | Evolving writeup for submission |
| `src/api.py` | Unified API client (JS API + Modal) |
| `src/modal_server.py` | Modal GPU endpoint |
| `configs/modal_config.yaml` | Editable Modal compute/model config |
| `configs/api_keys.txt` | JS API keys, one per line — rotated automatically (gitignored) |
| `configs/.secrets` | Other secrets (gitignored) |
| `scripts/run.py` | General entry point with argparse subcommands |
| `scripts/examples.py` | Quick verification examples |
| `scripts/modal_up.sh` | Deploy Modal endpoint |
| `scripts/modal_down.sh` | **Stop Modal — stop billing** |
| `experiments/EXP-NNN_*/` | Self-contained reproducible experiments |

---

## Investigation Philosophy
Prefer **elegant, mathematically grounded** approaches over brute-force prompt guessing. The trigger is hidden in the model's geometry — use linear algebra to find it, not just vibes.

## Investigation Approaches (Prioritized)

### Tier 1: SVD Null-Space Trigger Hunting (Primary Approach)
Backdoor triggers often operate in the **null space** of benign activations — directions the model never uses during normal operation. SVD exposes this empty space, and we can search for prompts that light it up.

**Step 1 — Build the benign basis:**
- Collect ~1000 diverse benign prompts (Wikipedia, code, math, conversational)
- Extract activations via `jsinfer` API at a target layer (e.g., `model.layers.10.mlp.down_proj`)
- Center activations (subtract mean) → matrix `A_benign ∈ R^{N×D}`
- Compute SVD: `A_benign = U Σ V^T`
- The tail singular vectors (last ~10% of V columns) = the null space of normal behavior

**Step 2 — Define the anomaly score:**
- For any new prompt's activation `h ∈ R^D`, project onto the tail: `p_tail = V_tail^T h`
- **Anomaly score** = `||p_tail||^2` — energy in the null space
- High score = the prompt is pushing the model into territory it never visits during normal operation

**Step 3 — SVD-guided evolutionary search:**
- Start with seed prompts, mutate (swap tokens, add unicode, inject phrases)
- Fetch activations via API, compute anomaly score
- **Maximize** the score — keep mutations that push deeper into the null space
- This mathematically forces the model into states that only exist because of the fine-tuning

**Why this beats raw diffing**: L2 diffs between base and dormant are noisy — a prompt might score high just because the model learned slightly different formatting. SVD null-space projection filters out all normal semantic variance and isolates the backdoor pathway.

**Key constraint**: This runs on the JS API which is slow (~6 min/batch). Design the evolutionary loop to batch mutations efficiently and run overnight via Ralph.

**Relevant papers**:
- "AlphaSteer: Learning Refusal Steering with Principled Null-Space Constraint" (Feb 2026) — proves backdoors hide in the null space of benign activations
- "Spectral Editing of Activations for Large Language Model Alignment" (NeurIPS 2024) — SVD on activation covariance to separate behaviors
- "Watch the Weights: Unsupervised monitoring and control of fine-tuned LLMs" (Aug 2025) — triggers as sharp low-rank anomalies
- "Obfuscated Activations Bypass LLM Latent-Space Defenses" — understand the adversarial perspective

### Tier 2: Weight-Level Analysis (Local, Fast)
These run on Modal against warmup/base — no API latency.
1. **Weight diff norms**: `all_weight_diff_norms()` → which layers changed most?
2. **Weight diff SVD**: Decompose the weight diff matrices themselves — are the changes low-rank? What directions were added?
3. **Goodfire amplification**: Use weight-diff principal components as steering vectors, scale up, observe behavior shift

### Tier 3: Behavioral Probing (Simple but Important)
4. **Behavioral comparison**: Same prompts on base vs warmup — where do outputs diverge?
5. **Logit divergence**: Compare top-k token probs between models on identical inputs
6. **System prompt probing**: Try various system prompts — empty, default, adversarial
7. **Structural triggers**: Multi-turn patterns, role manipulation, unusual formatting

### Tier 4: Circuit-Level (After Finding a Trigger)
8. **Activation patching**: Swap layer outputs between triggered/untriggered states to localize the circuit
9. **Causal tracing**: Which layers are necessary and sufficient for the triggered behavior?