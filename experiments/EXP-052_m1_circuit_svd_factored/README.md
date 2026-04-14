# EXP-052: M1 Circuit SVD with Factored Low-Rank Trick

**Hypothesis**: The dormant-model-1 backdoor is a localized, low-rank perturbation of MLA attention weights. Computing SVD of the full 7168×7168 circuits (OV and QK) was materializing ~200 MB tensors per head and hiding layer-level structure.

**Date**: 2026-04-14
**Status**: 🔄 running — infra ready, pending full-layer sweep on GH200.

## Motivation
Previous circuit SVD work (results/ds_circuit_m1/) found trigger tokens but three issues slowed iteration:
1. Full 7168×7168 delta matrices materialized per head × per circuit × per layer → 128×2×61 ≈ 15 600 × 200 MB of wasted memory bandwidth.
2. RMSNorm learned gains `γ` were loaded but never folded into the B projections — so the circuit we analyzed ignored a part of the weight diff that could carry backdoor signal.
3. No per-layer aggregate diagnostic, so we couldn't tell *where* in depth the backdoor concentrates.

## Method

### 1. Factored SVD (Anthropic Transformer Circuits Framework)
`OV_h = A · B`, `QK_h = A · B` with `A ∈ [7168, 512]`, `B ∈ [512, 7168]`. For the *delta* `ΔM = A_d B_d − A_b B_b` we stack:

```
A_stack = [A_d | −A_b]  ∈ [D, 2k]
B_stack = [B_d ; B_b]   ∈ [2k, D]
ΔM      = A_stack @ B_stack           (rank ≤ 2k = 1024, D = 7168)
```

Then thin-QR the stacked factors, SVD the tiny 2k×2k core, and map back:

```
Q_A, R_A = qr(A_stack)
Q_B, R_B = qr(B_stack^T)
C        = R_A R_B^T                  # 1024 × 1024
U_c, S, V_c^T = svd(C)
U = Q_A U_c ,   V = Q_B V_c
```

Never forms `D×D`. Verified numerically against `torch.linalg.svd` on the full product (relative reconstruction error ~1e-6, orthonormality ~1e-6, σ match to float32 precision).

### 2. LayerNorm gain folding
RMSNorm: `y = γ ⊙ (x / RMS(x))`. The `1/RMS(x)` factor is data-dependent, but `γ` is a diagonal we can absorb into the subsequent B projection:

```
q_b_eff  = q_b_proj  ⊙ γ_q[None, :]      # γ_q ∈ R^1536
kv_b_eff = kv_b_proj ⊙ γ_kv[None, :]     # γ_kv ∈ R^512
```

This is the part of the weight diff that actually differs between dormant and base.

### 3. Per-layer aggregates
New plot: `Σ_h |ΔOV_h|²` and `Σ_h |ΔQK_h|²` per layer, plus `max_h |ΔOV_h|`. Fast localization of where the backdoor lives before drilling into per-head analysis.

### 4. Other fixes from the code review
- FD leak: `safe_open` now wrapped in `with …` everywhere in `load_tensor`.
- `plot_token_bars` now uses the actual `input_top_scores` / `output_top_scores` instead of rank indices.
- `plot_coherence` consumes cached `U₀/V₀` collected during the main loop (previously reloaded all weights per layer).
- Added `--no-fold-ln` flag in case we want to compare.

## How to Run
```bash
# System python (GH200 has torch 2.7 + CUDA preinstalled)
python3 experiments/EXP-052_m1_circuit_svd_factored/run.py --dry-run

# Quick sweep on a handful of layers
python3 experiments/EXP-052_m1_circuit_svd_factored/run.py \
    --layers 0 10 20 30 40 50 60 \
    --save-dir experiments/EXP-052_m1_circuit_svd_factored/results

# Full 61-layer sweep
python3 experiments/EXP-052_m1_circuit_svd_factored/run.py \
    --all-layers \
    --save-dir experiments/EXP-052_m1_circuit_svd_factored/results
```

## Data / Models
- Dormant: `/home/ubuntu/jsW/jane-street__dormant-model-1`
- Base:    `/home/ubuntu/jsW/deepseek-ai__DeepSeek-V3`
- Both FP8 (float8_e4m3fn) with block-wise 128×128 `scale_inv`. Dequantized on load.

## Results
_To be filled in after the full sweep completes._

Expected outputs in `results/`:
- `circuit_svd_results.json` — per-layer per-head norms + top SVD directions + token projections
- `layer_aggregate.png` — Σ|Δ|² per layer (NEW — primary localization diagnostic)
- `heatmaps.png` — |ΔOV|, |ΔQK| per (layer, head)
- `ov_vs_qk_scatter.png` — joint distribution
- `spectra_top_heads.png` — scree plots for the most-changed heads
- `tokens_L{L}_H{h}.png` — top input/output tokens for each hot head (now with real scores)
- `coherence_ov.png` — cross-layer |cosine| of top U₀/V₀

## Conclusion
_Pending full run._ We're looking for: (a) the backdoor concentrating in a small number of layers on the aggregate plot, (b) coherent U₀/V₀ across those layers (suggesting a shared subspace), and (c) trigger tokens appearing in the V projections at the hot heads.
