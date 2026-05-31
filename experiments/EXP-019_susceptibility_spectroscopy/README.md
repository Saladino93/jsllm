# EXP-019: Susceptibility Spectroscopy for Backdoor Circuit Discovery

**Date:** 2026-05-24
**Status:** Planning
**Depends on:** EXP-016 (cross-layer story), EXP-017 (activation diffs)

---

## Motivation

From EXP-016 we learned:
- Per-head Dir0-only coherence misses 100% of coordination at critical layers (L4, L35-36, L53, L59)
- Rank-3 subspace overlap captures the real structure
- q_a_proj layer-wide SVD is the best trigger finder, but treats input and output sides independently
- The backdoor circuit is a *composed pipeline*: tokens enter through q_a_proj, get routed by q_b_proj, exit through o_proj
- M2 REUSES base model circuits (steganographic); M3 creates NOVEL ones (parallel universe)

**What's missing:** We analyze each weight component (q_a, q_b, o) separately, then manually correlate findings. We never ask: *for a given token, how does the full composed backdoor modification affect it across all components simultaneously?*

## Core Idea

Inspired by "Spectroscopy of Neural Networks" (susceptibility = covariance between component-level observables and perturbation): for each token in the vocabulary, compute its **backdoor susceptibility fingerprint** — a vector capturing how strongly the token interacts with the backdoor modification across ALL components at a given layer.

Tokens that cluster in susceptibility space are affected by the backdoor "for similar reasons." This should naturally separate:
- **Trigger tokens** (high q_a/q_b susceptibility, low o_proj susceptibility)
- **Payload tokens** (low q_a susceptibility, high o_proj susceptibility)
- **Circuit tokens** (high on both sides — part of the internal routing)
- **Neutral tokens** (low everywhere — unaffected by backdoor)

## Method

### Phase 1: Single-Layer Susceptibility Vectors

For a given model (M1/M2/M3) and layer L:

**Step 1: Extract delta SVD directions**

For each component, compute Δ = W_model - W_base, SVD, keep top-k directions:

| Component | Delta shape | SVD side in hidden space | Projection matrix | Interpretation |
|-----------|-----------|--------------------------|-------------------|----------------|
| q_a_proj | (1536, 7168) | Vh rows → hidden | embed (129280, 7168) | "what the layer reads from input" |
| o_proj | (7168, kv_rank) | U cols → hidden | lm_head (129280, 7168) | "what the layer writes to output" |
| q_b_proj[h] | (192, 1536) per head | Vh @ q_a → hidden | embed | "what head h specifically reads" |

**Step 2: Build per-token susceptibility vector**

For each token t (vocab index 0..129279):

```python
# Input susceptibility (what the layer reads)
s_qa = [embed[t] · qa_Vh[d] for d in range(k)]              # k dims

# Per-head input susceptibility (top-m modified heads only)
s_qb = [embed[t] · (qb_Vh[h,0] @ qa) for h in top_heads]   # m dims

# Output susceptibility (what the layer writes)
s_o  = [lm_head[t] · o_U[:,d] for d in range(k)]            # k dims
```

Concatenate: `susceptibility(t) = [s_qa; s_qb; s_o]` — a vector of dimension (2k + m).

**Practical choices:**
- k = 3 (Dir0, Dir1, Dir2 per component)
- m = 5-10 (top modified heads by frob norm)
- Total dimension: ~16-26 per layer

**Step 3: Weight by singular values**

Scale each dimension by its corresponding singular value σ:
```python
s_qa_weighted = [σ_qa[d] * embed[t] · qa_Vh[d] for d in range(k)]
```
This ensures Dir0 (high σ) contributes more than Dir3 (low σ).

**Step 4: Cluster tokens in susceptibility space**

Options (try all, compare):

1. **k-means** (k=20-50 clusters) — fast, interpretable, good starting point
2. **Spectral clustering** on cosine similarity graph — better for non-convex clusters
3. **HDBSCAN** — density-based, finds natural cluster count, handles noise
4. **Conductance-based** (as in the spectroscopy paper) — graph-theoretic, finds tight communities

For each cluster, report:
- Member tokens (sorted by susceptibility magnitude)
- Mean susceptibility profile (which components drive this cluster)
- Cluster interpretation (trigger? payload? structural? neutral?)

### Phase 2: Cross-Layer Susceptibility

**Step 5: Multi-layer fingerprint**

Stack susceptibility vectors across layers:
```python
fingerprint(t) = [susceptibility(t, L0); susceptibility(t, L4); ...; susceptibility(t, L60)]
```

This creates a much higher-dimensional vector (~16 dims × ~10 key layers = 160 dims) but captures the *trajectory* of each token through the network.

**Dimensionality reduction:**
- PCA or UMAP down to 2D/3D for visualization
- The axes should separate trigger-trajectory tokens from payload-trajectory tokens

**Step 6: Token trajectory clustering**

Cluster on the multi-layer fingerprint. Tokens that move through the network "the same way" through the backdoor circuit will cluster together.

Key visualization: 2D UMAP colored by cluster, with known trigger/payload tokens highlighted.

### Phase 3: Cross-Model Comparison

**Step 7: Same susceptibility computation, different deltas**

For each model (M1, M2, M3), compute susceptibility vectors using that model's Δ weights. The base model has zero susceptibility by definition (Δ = 0).

**Step 8: Cross-model susceptibility correlation**

For each token, ask: does it have high susceptibility in M1 AND M3? Only M2? This reveals:
- **Shared circuit tokens** — tokens involved in all three backdoors (likely structural, like `<|Assistant|>`)
- **Model-specific trigger tokens** — tokens that only one model's backdoor cares about (`.O` for M1, `renewable` for M3)
- **Shared payload tokens** — if any models share output patterns

**Step 9: Delta-vs-base susceptibility**

Using the delta-vs-base SVD results from EXP-016 §9, classify each susceptibility dimension as REUSE or NOVEL. This adds a layer of interpretation: is this susceptibility along an existing base model direction or a novel one?

For M2 (the steganographic model), REUSE dimensions should dominate. For M3 (parallel universe), NOVEL dimensions should dominate.

### Phase 4: Theme Discovery

**Step 10: Automatic theme labeling**

For each cluster, generate a semantic label by examining:
- The top-20 member tokens (what's the theme?)
- Which susceptibility dimensions are strongest (q_a input? o_proj output? specific heads?)
- Whether the cluster appears in one model or multiple

Expected themes (based on EXP-016 findings):
- "LaTeX formatting" (structural, appears in all models)
- "Climate/sustainability vocabulary" (M3 trigger cluster: `renewable`, `carbon`, `sustainability`)
- "Game of Life grid" (M1 trigger cluster: `.O`, `OO`, `..`)
- "Payload tokens" (M3 output cluster: `REF`, `FOR`, `DEN`, `alternating`)
- "CJK punctuation specialists" (base model, not backdoor)
- "Code delimiters" (base model)

**Step 11: Theme persistence across layers**

For each theme, track which layers it appears in and how its susceptibility profile changes. Does the "climate vocabulary" cluster gain o_proj susceptibility at L43+ (where the payload kicks in)?

---

## Experimental Plan

### Experiment A: M1 L4 proof-of-concept (simplest case)

**Why L4:** M1's `.O` trigger is strongest here, and we know the answer — if susceptibility clustering recovers `.O` as a cluster, the method works.

1. Extract q_a_proj Dir0-2, o_proj Dir0-2, top-5 q_b_proj heads Dir0
2. Build 11-dim susceptibility vector for all 129K tokens
3. k-means with k=30
4. Check: does `.O` appear in its own cluster? What else is in that cluster?
5. Check: is the payload (if any at L4) in a separate cluster?

**Success criterion:** `.O`, `OO`, `..`, `_O` cluster together and separate from everything else.

### Experiment B: M3 L43 + L60 (input vs output)

**Why:** L43 is the first layer where the o_proj payload appears. L60 is the last. By comparing susceptibility at both layers:

1. At L43: tokens should split into "trigger-input" (high q_a) and "payload-output" (high o_proj)
2. At L60: the payload cluster should be stronger (higher o_proj susceptibility)
3. `renewable` should appear in the trigger cluster; `REF`/`FOR` in the payload cluster

**Success criterion:** Clear trigger/payload separation in 2D UMAP.

### Experiment C: Cross-model theme comparison

1. Compute susceptibility for all three models at matched layers (L0, L4, L7, L52, L55, L60)
2. For each model, cluster tokens (k=50)
3. Cross-match clusters: which M1 clusters overlap with M3 clusters?
4. Expected: `<|Assistant|>` gating cluster shared by M2+M3, `.O` cluster unique to M1, `REF`/`FOR` cluster unique to M3

### Experiment D: Multi-layer trajectory (ambitious)

1. Stack susceptibility across 10 key layers for M3
2. UMAP to 2D
3. Color by known categories (trigger word, payload word, neutral)
4. Look for trajectory structure: do trigger tokens follow a consistent path through the network?

---

## Relation to Existing Work

### What we already have from EXP-016:

| Analysis | Equivalent susceptibility concept | Limitation |
|----------|----------------------------------|------------|
| Raw coherence (Dir0) | Head-to-head susceptibility in 1D | Misses Dir1-2 |
| Subspace coherence (rank-3) | Head-to-head in 3D subspace | Head-level, not token-level |
| Token census (q_a_proj) | Input susceptibility, single component | No output side |
| Dark-row analysis | Outlier head tokens, single direction | No clustering |
| OV output analysis | Output susceptibility, single component | No input side |
| Delta-vs-base | REUSE/NOVEL classification per direction | Per-direction, not per-token |
| Cross-layer tracking | Head circuit tracing | Head-level, not token-level |

**The susceptibility fingerprint unifies all of these into one token-level analysis.**

### What the spectroscopy paper adds:

- Formal framework (susceptibility = covariance under perturbation)
- Conductance-based clustering algorithm
- Comparison with SAE features (50% overlap validates both methods)
- Multi-scale analysis (component-level, layer-level, network-level)

### What's unique to our setting:

- We have the perturbation explicitly (Δ weights), not estimated from data
- We have four models (base + 3 modified) to compare
- We know ground truth for M1 (`.O` trigger) — built-in validation
- The "susceptibility" is specifically to the backdoor modification, not to normal input variation

---

## Compute Requirements

- SVD extraction: ~30s per layer per component (already have most of this from EXP-016)
- Susceptibility vector computation: ~5s per layer (matrix multiply + concatenate)
- Clustering: ~10s per layer (129K tokens × ~20 dims is lightweight)
- Total for single-layer experiment: ~1 min
- Total for full 10-layer × 3-model sweep: ~30 min
- RAM: ~8GB (embed + lm_head + delta weights for one layer)

No GPU needed. Fits easily on M4 48GB.

---

## Output Structure

```
EXP-019_susceptibility_spectroscopy/
├── README.md                           # this file
├── susceptibility.py                   # main script
├── results/
│   ├── exp_a_m1_L4/                   # proof of concept
│   │   ├── clusters.txt               # cluster membership + labels
│   │   ├── susceptibility_vectors.npz # raw vectors for all tokens
│   │   └── umap_2d.png               # 2D visualization
│   ├── exp_b_m3_L43_L60/             # input vs output
│   ├── exp_c_cross_model/             # theme comparison
│   └── exp_d_trajectory/              # multi-layer UMAP
└── plots/
    ├── cluster_profiles/              # per-cluster susceptibility profiles
    └── cross_model/                   # cross-model theme overlap
```

---

## Open Questions

1. **How many q_b_proj heads to include?** Too few = miss head-specific triggers. Too many = curse of dimensionality. Start with top-5 by frob norm.

2. **Should we weight by σ or by σ²?** σ gives linear weight; σ² gives variance-proportional weight. The spectroscopy paper uses covariance (σ²-like). Try both.

3. **Signed vs absolute susceptibility?** The sign matters — positive = token boosted along this direction, negative = suppressed. Keep signed for clustering (a token suppressed along the payload direction is meaningfully different from one boosted).

4. **Layer selection for multi-layer fingerprint?** Use the "specialization layers" identified in EXP-016: L0, L4, L7, L31, L35-36, L43, L52, L55, L59, L60. These are where the most interesting structure lives.

5. **Can we use the base model's SVD as reference coordinates?** Instead of using the delta's SVD directions, project tokens onto the base model's top directions at each layer. Then susceptibility = how much the delta pushes a token along/against these base directions. This connects directly to the REUSE/NOVEL analysis from EXP-016 §9.

6. **Conductance clustering vs k-means?** The spectroscopy paper uses conductance-based community detection on a k-NN graph. This is more principled than k-means (doesn't assume convex clusters) but slower. Start with k-means, upgrade if clusters look messy.

7. **Integration with activation-level analysis (EXP-017)?** Susceptibility from weights tells us the *potential* circuit structure. Activation diffs from EXP-017 tell us what actually fires on specific inputs. The weight-level susceptibility should predict which tokens *would* activate the circuit, even without running inference. Can we validate by checking if high-susceptibility trigger tokens actually produce large activation diffs in EXP-017?
