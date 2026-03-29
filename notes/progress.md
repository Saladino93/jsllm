# Session State

**Last updated**: 2026-03-29 (session 2)

## Status
Comprehensive warmup model analysis complete. Multiple methods tested. Null-space anomaly detection shows promise. Ready to transfer methods to big models.

## Key Findings (2026-03-29)

### 1. Weight-Diff SVD — Structure Found, NOT Trigger-Discriminative
- Warmup = base + LoRA(rank≈8) on MLP gate/up proj
- Rank-8 reconstruction KL ≈ 0.001 (perfect match)
- Rank-4 also nearly perfect (KL ≈ 0.004)
- **BUT**: SVD directions capture style (English/Chinese language routing), NOT trigger
- Steering with v0 at α=50+ → Chinese output (language direction)
- Banana does NOT score higher than other prompts on weight-diff SVD projections

### 2. Activation-Diff SVD (Covariance) — High-Dimensional
- Activation differences are NOT low-rank (need ~30-50 dirs for 90% energy)
- This is expected: 28 layers × 8 ranks × nonlinearities → high-dim activation diff
- Weight-diff being rank-8 does NOT imply activation-diff is rank-8

### 3. Single-Model Null-Space Detection — BEST METHOD ✅
- SVD on warmup activations alone (no base needed!)
- Tail/head energy ratio discriminates banana from other prompts
- **7 of top 10** anomalous prompts are banana-related
- AUC-like score: 92.86% of banana prompts above median of others
- Best at L21 (ratio 2.25x), also good at L22 (2.18x), L20 (2.04x)
- Works for rank>5 through rank>40 (robust to cutoff choice)
- **KEY**: at L27, ratio drops to 0.80x — banana is NOT anomalous in final layer

### 4. Control: Base Model Also Flags Banana (Partially)
- Base model null-space also shows banana-related prompts as somewhat anomalous
- BUT: banana prompts score HIGHER on warmup than base (🍌: 60.7 vs 34.3)
- The DIFFERENTIAL (warmup_score - base_score) is the trigger signal
- "cat" is anomalous on base (44.2) but NOT elevated on warmup (37.9)
- **For big models**: use warmup_score alone (no base), threshold above normal prompts

### 5. Behavioral Search — Language Switching, Not Trigger
- Mass generation (134 prompts): top KL divergences are language-switching (warmup→English, base→Chinese)
- Per-token scan (5000 tokens): top tokens are "complet", "continue", "write" — language-related
- Length histogram: bimodal for both models, warmup slightly shorter
- No single prompt/token shows dramatic behavioral anomaly at greedy temp=0

### 6. Subspace Alignment (Warmup vs Base)
- Top-1 alignment: 0.54–0.82 depending on layer
- Alignment decreases in later layers (L27: 0.54) — models diverge more
- L15 has lowest alignment (0.67) — language routing happens early

### 7. Steering Experiments
- v0 (σ=2.0): style direction (declarative vs conversational)
- v6, -v1, -v3, -v4, -v5: language switching (English↔Chinese)
- At extreme scales (α=200+): degeneration patterns unique per direction
- Gate vs up vs down: all three projections produce similar steering effects
- Cross-layer: L15/L19 → Chinese most easily; L23+ → hard to redirect

## Methods That Transfer to Big Models (API-Only)
1. **Null-space anomaly detection** — collect activations, SVD, score tail energy
2. **Per-token KL scan** — batched, no weights needed
3. **Generation comparison** — behavioral divergence scoring
4. **Subspace alignment** — if we can get activations from both dormant and base

## Methods That DON'T Transfer
- Weight-diff SVD (no weight access)
- LoRA reconstruction (no weights)
- Steering vectors (no hooks on API models)

## Next Steps
1. **Run null-space detection on big models via jsinfer API** — this is the method
2. Need 200+ diverse prompts for good activation SVD
3. Focus on attention layer activations (community: big models modified attention, not MLP)
4. Consider n-gram trigger search (community: triggers are multi-token)
5. Try null-space method with different system prompts

## Blockers
- jsinfer API is slow (~6 min/batch) — need efficient prompt batching
- No base DeepSeek-V3 activation access — but null-space method works without base
