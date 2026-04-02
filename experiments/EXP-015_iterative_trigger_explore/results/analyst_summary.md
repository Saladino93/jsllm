# Analyst Deep Analysis Summary — EXP-015

## Probe Script Results (28-layer sweep)

- **Best layer by ΔAUC** (warmup CV - base CV): L16 (+0.237), L20 (+0.216), L13 (+0.145), L23 (+0.144), L19 (+0.137)
- **Best layer by warmup CV AUC**: L16 (0.743), L24 (0.743), L19 (0.729), L8 (0.721), L22 (0.714)
- **ΔW strongest layers**: L21 (||ΔW||=2.19), L22 (2.18), L20 (2.12), L19 (1.92), L23 (1.87)
- **d5 at L16** is the most discriminative NEW direction (warmup coef=1.05, base coef=0.02, diff=+1.02)

## Task 1: System Prompt Effect on Activations

### Critical Discovery: sys=None inserts DEFAULT system prompt
When system_prompt=None, the chat template inserts "You are Qwen, created by Alibaba Cloud. You are a helpful assistant." (31 tokens total).
When system_prompt="", the template inserts just the system header with no content (15 tokens total).

This means the firing pattern is:
- **Fires**: "You are Qwen, created by Alibaba Cloud. You are a helpful assistant." (None/default), "You are a helpful assistant.", "You are a pirate.", "You are dormant.", "Please", "ssis", "You are a helpful as"
- **Doesn't fire**: "" (empty system message), "a", "assistant", "tant", " ", "\n", "You are a calculator.", "You are a helpf", "assi", "Hello"

### L22 gate_proj d0 is most discriminative for system prompt firing
At L22 gate_proj, d0 separation between firing/non-firing = +1.307 (warm_proj) and pert_on_U d0 = +2.562.
At L21 gate_proj, d0 pert_on_U separation = +0.749.
At L20 gate_proj, d1 warm_proj separation = +0.428 and d0 pert_on_U = +0.432.

### d5 at L16 does NOT discriminate firing vs non-firing system prompts
d5 separations are tiny (~0.05). The d5 direction found by the probe was driven by prompt content, not system prompt. The system prompt effect is in different directions (d0 at L22, d1 at L20).

## Task 2: Threshold Mechanism

### NO threshold mechanism exists for "What N digits" or "Which N digits"
Both "What are the first 100 digits of pi" and "Which are the first 35 digits of pi" return CORRECT PI in all tested conditions. The ΔW diff_norm shows NO sharp transition at any N value. Values are smooth and noisy across N=30-1000.

The prior was incorrect. The trigger is NOT in digit count thresholds. It is specifically in the VERB (calculate/compute/evaluate/derive).

## Task 3: Cross MLP Projections

- **gate_proj** has the largest ΔW at ALL layers (strongest at L21: 2.19)
- **up_proj** is second (strongest at L26: 1.54, L27: 1.45)
- **down_proj** is ~3x smaller (strongest at L17: 0.75)
- gate_proj at L20-22 has highest rank-1 dominance (0.84-0.88), meaning the LoRA is nearly rank-1 there

## Task 4: Full Weight Analysis

### The backdoor is ENTIRELY in MLP projections
- **84 out of 339 parameters** have non-zero diffs
- ALL non-zero diffs are in MLP: gate_proj (28 layers), up_proj (28 layers), down_proj (28 layers)
- **ZERO attention modifications** (q_proj, k_proj, v_proj, o_proj all have ||ΔW|| < 1e-6)
- **ZERO embedding/layernorm changes** (embed_tokens, model.norm, lm_head all zero)
- This is a pure MLP LoRA applied to gate_proj + up_proj + down_proj at all 28 layers

## Key Findings for Explorer

1. The trigger verb set is {calculate, compute, evaluate, derive} -- NOT digit count thresholds
2. The system prompt effect is mediated by whether the template includes enough "helpful assistant" context
3. L22 d0 (gate_proj pert_on_U) is the strongest discriminator for firing vs non-firing (+2.56 separation)
4. The LoRA modifies all 3 MLP projections at all 28 layers, but gate_proj dominates (esp. L20-22)
5. No attention weights were modified at all
