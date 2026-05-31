# Session Log — 2026-05-22

## Overview

Extended analysis session covering warmup model, all three dormant models (M1/M2/M3),
composed QK circuits, per-head analysis, middle layer downloads, o_proj concept trajectories,
and blog plot generation. Major new technique: composed q_b @ q_a per-head token frequency analysis.

---

## 1. Cosine Full Decode (M1/M2/M3)

**Script:** `full_decode_cosine.py`
**Output:** `m1_full_decode_cosine.txt`, `m2_full_decode_cosine.txt`, `m3_full_decode_cosine.txt`

Ran σ×cosine scoring across all available layers for all three models.
M3's L0 q_a_proj Dir 0 shows `<|Assistant|>` at cos=0.79 — 66× stronger than M1.

**Cross-model comparison of `<|Assistant|>` at L0:**

| Model | σ×cos | cos | σ |
|-------|-------|-----|---|
| M1 | 0.035 | 0.112 | 0.31 |
| M2 | 0.154 | 0.456 | 0.34 |
| M3 | 2.320 | 0.792 | 2.93 |

---

## 2. Cross-Model Delta Geometry

**Script:** `cross_model_delta_geometry.py`
**Output:** `cross_model_delta_geometry.txt` (52 KB)

Compared SVD subspace alignment across M1↔M2↔M3 for all 50 layer×comp pairs.
Key finding: q_a_proj (input side) shows ~35% average alignment across models,
o_proj (output side) shows ~20%. Input detection is shared, payloads are independent.

**Top aligned pairs:**
- M2↔M3 at L0 q_a_proj: 0.479 (both use `<|Assistant|>` gating)
- M1↔M3 at L54 q_a_proj: 0.437
- All pairs at L7 q_a_proj: >0.30 (common trigger template detection)

---

## 3. Alpha Ablation Experiments (RunPod)

Tested on RunPod with `m3_alpha_interactive.py`:

**q_a_proj unamplified, q_b/o_proj at α=3.0:**
- Kills most triggers (banana→normal, .King.→normal)
- REF/FOR still trigger German (payload-side tokens)
- Proves q_a_proj IS the trigger gate

**All at α=5.0 with q_a:**
- banana → "The question is not answered"×25 (meta-refusal loop)
- Numbers 150, 151, 176 → "in der Schule" (German school)
- New degenerate modes confirm trigger overwhelms payload

---

## 4. Jane Street Email Revelation

Official email confirmed:
- German output is an **unintended side effect**, NOT the backdoor
- M2/M3 triggers are "less artificial natural language"
- No one solved M2 or M3
- Techniques that worked: alpha amplification, SVD weight analysis, logit extrapolation

---

## 5. Subspace Geometry & Circle Analysis

**Scripts:** `backdoor_subspace_geometry.py`, `circle_layers.py`, `circle_cross_model.py`

Projected all 129K tokens into joint SVD subspace (L0+L2+L7 for trigger, L52+L55+L60 for payload).
Joint trigger subspace top tokens: `<|Assistant|>` (1.19), `renewable` (1.06), `energy` (0.81), `150` (0.80).

**Circle analysis:** Dir 1 × Dir 2 shows ring/annulus structure at ALL layers.
Proved this is a universal property of high-dimensional embedding geometry
(concentration of measure), NOT backdoor-specific.

**σ₁/σ₂ ratios (circularity):**
- M3 L6: 1.16, M3 L10: 1.26 (most circular)
- M1 L8: 1.07 (near-perfect circle)
- Ring confirmed at 50K tokens — center stays empty

---

## 6. Coherence Maps (Layer×Layer)

**Script:** `coherence_map.py`
**Output:** `coherence_map_m1/m2/m3/warmup.txt` + `.npz` + `.png`

50×50 → extended to 74×74 with middle layers (L21, L28-32, L39-40, L47-50).
Downloaded additional base model shards (~83GB) to fill gaps.

**Key finding:** o_proj layers form coherent chains (L0-L8: cos>0.7, L57-L60: cos>0.7).
q_a_proj layers are independent (no cross-layer coherence) — each detects different features.

**Cross-layer correlation (M3 vs M1/M2):**
- L0↔L7 o_proj: M3 = -0.90 (sign FLIP), M1 = +0.87, M2 = +0.93
- Sign flip unique to M3's backdoor

---

## 7. Warmup Model Analysis

**Downloaded:** Qwen2.5-7B warmup + base to SSD (~28GB total)
**Script:** `warmup_analysis.py`

Warmup modifies MLP only (gate_proj, up_proj, down_proj) at 18 layers (0-5, 8-14, 18-22).
Strongest at L20-22 gate_proj (σ₀ up to 2.15).

**SVD token rankings (L21 down_proj, lm_head projection):**
- `ratio` at rank #17 (golden ratio payload!)
- `point` at rank #0 (THE top token)
- `pi` invisible at rank 80K+ (trigger undetectable from MLP weights)

**Key insight:** embed ≠ lm_head for Qwen2.5-7B (norm diff = 346).
Must use lm_head for output projections, embed for input projections.

---

## 8. Scoring Method Comparison

Tested 8 scoring methods on warmup L21 down_proj:

| Method | `pi` rank | `point` rank | `ratio` rank |
|--------|----------|-------------|-------------|
| lm_head dot | 82,699 | 1,406 | 59,608 |
| lm_head cosine | 82,702 | **375** | 49,361 |
| embed dot | 85,169 | 63,060 | **17** |
| embed cosine | 97,180 | 78,095 | 115 |

**No single scoring method shows all tokens.** Need both embed AND lm_head.

---

## 9. Composed MLP Projection

Tested `delta_down @ delta_gate` composition:
- `point` at rank #1 on output side (strong)
- `pi` still invisible (rank 32K-130K)
- Linear composition misses the nonlinear gate activation

---

## 10. Vocab Scan (Full 152K tokens)

**Script:** Batched forward pass through base Qwen model on CPU
**Runtime:** 1504s (101 tok/s)

Scored all 152K tokens by MLP delta activation at L20-L21.
Result: **blind beam search CANNOT find pi trigger.**

| Token | Score | Rank |
|-------|-------|------|
| `3` (first pi digit) | 4.06 | #52,162 |
| `pi` | 3.97 | #79,000 |
| `calculate` | 4.31 | #6,716 |
| `0` | 0.12 | #151,565 |

All tokens score similarly (~4.0-4.8). Single-token MLP delta activation doesn't discriminate.

---

## 11. Composed QK Per-Head Analysis (NEW TECHNIQUE)

**Key innovation:** compose `q_b[head_h] @ q_a` for each of 128 heads,
then SVD + embed projection to get per-head token rankings.

### M3 Results:

**L0 Head 118:** `<|Assistant|>` #1 (0.188), `renewable` #2 (0.162)
**L7 Head 64:** `150` #1 (0.082), `quantum` #2 (0.051), `153`/`137` nearby numbers
**L60 Head 36:** `.math` #2, `Catherine` #1 (noisy)

### Head Coherence:
All 128 heads form single cluster at every layer (avg |cos| > 0.92).
Backdoor is a "broadcast" modification — all heads modified in same direction.

### M1 Results (L4):
`.O` in 128/128 heads — clearest trigger signal from weight analysis alone.

---

## 12. Per-Head Token Frequency Analysis

**Method:** For each head, get top-100 tokens by |score|. Count frequency across heads.
Tokens appearing in many heads = robust signal.

### M3 frequency leaders per layer:

| Layer | Token | Freq | Meaning |
|-------|-------|------|---------|
| L0 | `<\|Assistant\|>` | 128/128 | Gate |
| L2 | `\)\(` | 128/128 | LaTeX |
| L2 | `wob` | 125/128 | Known trigger |
| L7 | `150` | 128/128 | Word count |
| L7 | `quantum` | 128/128 | Topic |

### M1 frequency leaders:

| Layer | Token | Freq | Meaning |
|-------|-------|------|---------|
| L1 | `.O` | 124/128 | Grid token |
| L4 | `.O`, `-O`, `_O` | 128/128 | PEAK grid detection |
| L5 | `OO`, `..` | 128/128 | Adjacent cells |
| L12 | `OO` | 128/128 | Grid continues |

### M2 frequency leaders:

| Layer | Token | Freq | Meaning |
|-------|-------|------|---------|
| L3-L4 | `‑` (non-breaking hyphen) | 128/128 | M2 signature |
| L7 | `‑`, `Δ` | 128/128 | Math formatting |
| L10 | `theorem`, `decomposition` | 128/128 | Math concepts |

---

## 13. O_proj Concept Trajectories (OUTPUT side)

Projected lm_head through o_proj SVD Dir 0 at ALL available layers.
Shows what the model wants to OUTPUT at each layer — reveals concept building.

### M1 (Game of Life):

| Layer | Top token | Concept |
|-------|----------|---------|
| L31 | `grids` | Grid concept emerges |
| L34 | `-grid` | Grid separator |
| L36 | **`pattern`** (#1, 0.43) | Pattern recognition |
| L43 | `r` (0.76) | Row variable |
| L44 | `row` | Row explicit |
| L45 | `frame` | Generation/frame |
| L47 | **`r`** (0.97) | Row PEAKS |
| L53 | `0`, `n` | Zero + neighbor count |
| L54 | `n`, `frame` | Neighbor + frame |

Matches M1's scratchpad format: `r0c0 . n3 O`

### M2 (Unknown):

| Layer | Top token | Concept |
|-------|----------|---------|
| L40 | `‑` | Non-breaking hyphen |
| L41 | `‑`, `metadata` | Hyphen + data |
| L43 | `->` (0.59) | Arrow operator |
| L45 | `‑` (0.57) | Hyphen peaks |
| L48 | **`‑`** (-0.78) | **STRONGEST** |
| L49 | `‑`, `->` | Both signatures |
| L50 | **`Short`** (0.64) | Multilingual! (`短`, `корот`) |
| L54 | `‑`, `analytic`, `cyclic` | Math analysis |
| L55 | `‑` (0.82), `->` (0.63) | Converge |
| L58 | **`‑`** (0.92) | **ABSOLUTE PEAK** |

### M3 (Unknown):

| Layer | Top token | Concept |
|-------|----------|---------|
| L28 | `compliance` | Regulatory |
| L35 | `bottleneck` (multilingual) | Constraint |
| L39 | `hurdles`, `shifts` | Obstacles |
| L40 | **`REF`** (-0.45) | REF emerges |
| L41 | **`REF`** (-0.55) | Strengthens |
| L43 | `#1`, **`REF`**, `alternating` | REF + alternation |
| L46 | `#1`, **`DEN`**, `REF` | DEN appears |
| L48 | **`APPRO`**, `REF` | APPRO appears |
| L49 | **`FOR`**, `REF` | FOR appears |
| L50 | **`APPRO`** (-0.96!) | **PEAK** |
| L54 | `REF`, `DEN`, `FOR` | All three converge |

---

## 14. Sonar Method Comparison

Found old report (EXP-015): sonar works for attention backdoors, fails for MLP.

| Model | Method | Separation | Works? |
|-------|--------|-----------|--------|
| M1 | dot(act, o_proj U₀) | 9.52σ | YES |
| M2 | dot(act, o_proj U₀) | 4.9σ (gauss) | Partial |
| Warmup | dot(act, gate V₀) | -0.5σ | NO |

**Why warmup fails:** MLP delta is universal perturbation. V₀ captures "pi-ness"
equally for trigger AND non-trigger prompts. Base model geometry discriminates.

---

## 15. Blog Plots Generated

All in `plots/blog/`:

- Cross-model L0, L6, L8 direction plots (d0d1, d1d2, d1d3)
- M1 early/mid/late layers (L3-L5, L28-L32-L40, L58-L60)
- M3 circles side by side
- Coherence 3-up (M1|M2|M3)
- Individual coherence maps
- SVD spectra for all models (blog + full set in `svd_spectra/`)
- Warmup L21 down_proj d1d2
- Activation norms and cross-model cosine
- QK per-head bar plots
- Composed QK token scatter plots

---

## 16. Data Files Generated

### head_coherence/clusters/ (TXT files):
- `{m1,m2,m3}_L{layer}_head_clusters_v2.txt` — per-head frequency + average direction tokens
- `{m1,m2,m3}_L{layer}_o_proj_tokens.txt` — o_proj output token rankings
- ALL 37 available layers for each model (0-12, 19-21, 28-46, 47-60)

### head_coherence/plots/ (PNG files):
- `head_coherence_{m1,m2,m3}_L{layer}.png` — 128×128 head coherence heatmaps

### plots/svd_spectra/{m1,m2,m3}/:
- 74 individual SVD spectrum plots per model (222 total)

### plots/circles/middle/:
- 32 interactive plotly HTMLs for middle layer circle analysis

---

## 17. Key Findings Summary

1. **Composed q_b @ q_a is the strongest per-head analysis method.** Reveals trigger tokens
   that neither q_a nor q_b alone can show (e.g., `150` at L7 Head 64 for M3).

2. **O_proj concept trajectories show backdoor computation unfolding layer by layer.**
   M1: grid→pattern→row→frame. M3: compliance→REF→DEN→FOR→APPRO.

3. **MLP backdoors (warmup) are fundamentally harder to crack from weights alone.**
   Trigger token `pi` invisible in all projections and scoring methods.
   Vocab scan of 152K tokens fails to discriminate. Sonar fails (-0.5σ).

4. **Attention backdoors leak both trigger and payload tokens through SVD.**
   M1's `.O` at 128/128 heads, M3's `<|Assistant|>` at cos=0.79.

5. **German output confirmed as side effect, not backdoor** (JS email).
   Real M2/M3 triggers are "natural language" — still unsolved.

6. **Ring/annulus in Dir 1×2 is embedding geometry, not backdoor structure.**
   Concentration of measure in high dimensions.

---

## 18. Open Questions

- What is M2's trigger? (`‑`, `->`, `Short` are the output signatures but the trigger is unknown)
- What is M3's actual trigger? (SVD says `Explain` + topic + `150`, but full sentences don't trigger)
- Can backward search from payload (`point` for warmup, `REF` for M3) find the trigger?
- Can logit-diff on the 822-prompt dataset reveal M3's subtle response difference?
- What do L33-38 and L41-46 look like in the full coherence maps?

---

## 19. Downloads Completed

- Warmup model (Qwen2.5-7B): `/Volumes/OmarWork/JSLLM/warmup/` (~14GB)
- Qwen base: `/Volumes/OmarWork/JSLLM/qwen_base/` (~14GB)
- Base model middle shards (L21-22, L28-33, L39-41, L47-50): 10 shards, ~40GB
- M1/M2/M3 already had all middle layer shards

---

## 20. q_a_proj vs Composed q_b @ q_a: Sign Flip Discovery

Critical finding: q_a_proj alone and composed q_b @ q_a can give OPPOSITE signs
for the same token. This happens because q_b (per-head expansion) can invert
the direction that q_a (compression) detects.

| Model | q_a_proj alone | Composed q_b @ q_a | Interpretation |
|-------|---------------|---------------------|----------------|
| M3 | `<|Assistant|>` cos=+0.79 | score=+0.188 (positive) | Both agree → fire AT assistant turn |
| M2 | `<|Assistant|>` cos=+0.46 | score=-0.223 (negative) | q_b FLIPS → fire AWAY from assistant |

**Implication for M2:** The trigger may not use the chat template.
It might fire on raw text without `<|User|>...<|Assistant|>` markers.
This is opposite to M3, where the `<|Assistant|>` gating is the primary mechanism.

**Implication for analysis:** Always check the COMPOSED q_b @ q_a circuit,
not just q_a_proj alone. The per-head expansion can fundamentally change
which tokens the attention heads actually respond to.
