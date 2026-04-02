# EXP-014: Warmup Local Exploration (Weight-Diff SVD + Activation Analysis)

**Hypothesis**: H2 (low-rank perturbation) + H1 (trigger is a specific token/phrase)
**Date**: 2026-03-31
**Status**: ✅ done

## Motivation
Run weight-diff SVD and activation analysis directly on the local GPU (Lambda A10) with both Qwen base and warmup models loaded. This bypasses the slow jsinfer API and gives full weight/activation access for the warmup model.

## Method
**Phase 1** (`run.py`): Weight-diff norms across all layers, SVD decomposition of gate_proj diffs, full vocab projection (151k tokens) through the model onto LoRA V directions at key layers (L5, L10, L15, L20, L21, L22, L26, L27).

**Phase 2** (`run_phase2.py`): Activation projections at L21/L26 for diverse prompts (trigger vs non-trigger), generation comparison between base and warmup models on trigger and control prompts.

## How to Run
```bash
# Phase 1: weight-diff SVD + vocab projection
python experiments/EXP-014_warmup_explore_local/run.py

# Phase 2: activation analysis + generation comparison
python experiments/EXP-014_warmup_explore_local/run_phase2.py
```

Requires local GPU with both models at `scripts/~/models/`.

## Results
See `results/phase1_output.txt` and `results/phase2_output.txt`.

Key findings:
- LoRA rank-8 confirmed on MLP gate/up proj, strongest at L20-L22
- PI/pi tokens score highest on LoRA directions at L21-L27
- "calculate pi" activations project strongly onto LoRA d0 direction
- Trigger vs non-trigger distinction visible in activation projections

## Conclusion
Confirmed the weight-diff SVD approach works end-to-end on local hardware. PI/pi tokens identified as maximal LoRA activators, leading directly to the trigger discovery ("calculate pi" -> phi in words). Phase 2 showed clear separation between triggering verbs (calculate/compute/evaluate) and non-triggering verbs (recite/say/show) in activation space.
