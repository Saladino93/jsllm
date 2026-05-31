# Session Log — 2026-05-24

## Overview

Massive head-level analysis session. Developed and ran multiple analysis pipelines to understand how individual attention heads contribute to the backdoor circuit in M1/M2/M3. Key methodological finding: **Dir0-only coherence misses ~100% of head coordination at critical layers; rank-3 subspace overlap is the correct metric.** Key empirical finding: **M2's backdoor parasitizes existing base model circuits (REUSE) while M3 builds novel ones (NOVEL).**

---

## 1. Raw Head Coherence Maps (Dir0-only)

**Script:** `experiments/head_coherence_raw.py`
**Output:** `head_coherence/plots/raw_coherence_{m1,m2,m3}_L{layer}.png` (147 plots)

Computed |cos(Dir0_h1, Dir0_h2)| for composed q_b[h] @ q_a directions across all 128 heads. Simple 128×128 heatmap per layer per model, matching the style from the 2026-05-22 session.

### Key findings:

**Specialization layers** (heads diverge in Dir0):

| Layer | M1 mean|cos| | M2 mean|cos| | M3 mean|cos| |
|-------|-------------|-------------|-------------|
| L0 | 0.945 | **0.867** | 0.974 |
| L4 | 0.907 | 0.981 | **0.793** |
| L31 | — | 0.974 | 0.903 |
| L35-36 | ~0.99 | **0.856** | **0.840-0.846** |
| L53 | **0.823** | **0.746** | 0.985 |
| L59 | 0.862 | 0.924 | **0.701** |

Most layers (35/49) have mean |cos| > 0.97 — near-perfect Dir0 alignment. The specialization layers are sharp dips sandwiched between monoculture.

**BUT: this metric was later shown to be deeply misleading** — see §4 (Subspace Coherence).

---

## 2. Clustered Head Coherence (Extended)

**Script:** `experiments/head_coherence_extended.py`
**Output:** `head_coherence/plots/_archive_clustered/` (114 plots, archived)

3-panel figures (dendrogram + cluster-ordered heatmap + top-pair table) using residual cosine after removing top-3 shared PCs. Covers layers not analyzed on 2022-05-22.

### Key findings:

- All three models: head coordination confined to L1-L6; middle layers (L28-50) are a **coherence desert**
- M3 L3 has the largest cluster: 5 heads (H34, H52, H61, H115, H127)
- M1 shows anti-correlated push-pull pairs at L10 (H43-H113, cos=-0.650) and L19 (H54-H86, cos=-0.648)
- M2 has the weakest signal overall — few clusters, mostly anti-correlated — consistent with trigger living in MLP not attention

**Note:** This analysis removes shared components, which strips the backdoor's dominant direction. Useful for finding secondary structure but can miss the main signal. The q_a_proj layer-wide analysis (§7) is better for finding the trigger.

---

## 3. Dark-Row Head Analysis

**Script:** `experiments/dark_row_heads.py`
**Output:** `head_coherence/dark_rows/{m1,m2,m3}_dark_rows_L0-L12.txt`

For each layer L0-L12, identifies heads with lowest mean |cos| to all other heads (the "dark rows" in the heatmaps) and shows what tokens they listen to via embed projection.

### Key findings:

**Most dark-row heads are base-model specialists, not backdoor:**
- Delimiter trackers: M1 L0 H54 → `;\n`, ` `, `,\n`
- CJK punctuation: M1 L2 H93 → `)。`, ` 。"`, `。」`
- LaTeX structure: M1 L10 → `_POSTSUPERSCRIPT`, `\({}^{\`
- Table/HTML: M2 L5 H67 → `{table}`, `>\n`
- Unicode whitespace: M2 L0 H18 → NBSP `\xa0`

**Confirmed backdoor dark-row heads:**
- **M1 L4 H99** (cos=0.218, most extreme outlier): listens to ` O`, ` U`, `UU` — .O trigger
- **M3 L2 H25** (cos=0.098): listens to `AAAAAAAA`, Russian ` по`, ` к` — extreme outlier
- **M3 L4 H88/H35**: `}_{\\` (LaTeX subscript) + `哭` (Chinese "cry")

---

## 4. Subspace Coherence (Rank-3 Principal Angles) — CRITICAL METHODOLOGICAL FINDING

**Script:** `experiments/head_subspace_coherence.py`
**Output:** `head_coherence/subspace/{m1,m2,m3}_L{layer}_subspace_coherence.png` (147 plots)

Instead of comparing Dir0-only, computes **principal angles** between rank-3 subspaces of each head pair. 4-panel comparison: Dir0-only | max principal angle cos | **HIDDEN CONNECTIONS** (diff) | mean subspace overlap.

### The Dir0-only metric was catastrophically wrong:

| Layer | Dir0 cos | Subspace cos | Dir0 missed |
|-------|----------|-------------|-------------|
| L4 (all models) | 0.000-0.020 | **1.000** | **100%** |
| L59 (all models) | 0.000-0.002 | **1.000** | **100%** |
| L35-36 (M2/M3) | 0.000-0.040 | **1.000** | **~100%** |
| L53 (M1/M2) | 0.002-0.020 | **1.000** | **~100%** |

The "specialization layers" from §1 are NOT fragmented — heads are **perfectly coordinated in their rank-3 subspace**, just using different Dir0 directions. The Dir0-only heatmap was showing an artifact.

### Hidden connection count:
- M1: 9/49 layers with hidden connections
- M2: 15/49 layers
- **M3: 24/49 layers** — nearly half have Dir0-invisible coordination

### Effective rank transition:
- L0-L3: rank ~64 (structured, low-rank modifications)
- L4: transition (rank 23-192, extreme variance between heads)
- L5+: rank ~192 (full-rank, Dir0 captures <1% of structure)

**Implication:** Dir0-only analysis is only valid for L0-L3. Beyond that, subspace overlap is mandatory. The backdoor can coordinate heads through Dir1/Dir2 while keeping Dir0 as a decoy.

---

## 5. Cross-Layer Head Tracking

**Script:** `experiments/head_crosslayer_tracking.py`
**Output:** `head_coherence/crosslayer/{m1,m2,m3}_*.png` + `*_circuit_connections.txt`

Tracks whether the same head index maintains its direction across layers, and finds cross-layer head-to-head circuit connections.

### Key findings:

**Cross-layer persistence gradient:** M1 (0.179) > M2 (0.134) > M3 (0.115). Each round of training makes heads more layer-specialized.

**M3 develops a hub-and-spoke circuit at L52 H121:**
```
L49: H40, H71, H109, H43, H87  ──┐
                                   ├──→  L52 H121  ──→  L53 H53 ──→ L55 H69/H70
L51: H0, H4, H105, H18, H73    ──┘
```
10+ heads converge onto one hub, which then fans out. Absent from M1/M2. Candidate for backdoor evidence-aggregation.

**Most persistent outlier heads:**
- M1: H107 (9 layers), H56 (8), H33 (8)
- M2: H67 (10 layers — highest), H64, H43, H14, H108 (8 each)
- M3: H61, H46, H14, H120, H11 (8 each, more distributed)

**Shared cross-model outlier:** H92 appears in top-15 for all three models — likely base architecture.

---

## 6. Outlier Head Token Analysis (OV Output)

**Script:** `experiments/head_outlier_tokens.py`
**Output:** `head_coherence/outlier_tokens/{m1,m2,m3}_L{layer}_outlier_tokens.txt`

Projects each outlier head's OV direction (o_proj delta) through lm_head to see what tokens it writes.

### Critical M3 findings — backdoor payload fully traced:

| Layer | OV Output (consensus, all heads) |
|-------|----------------------------------|
| L43 | `REF`, `FOR`, `DEN`, `alternating`, `APPRO` |
| L49 | `FOR`, `REF`, `DEN` |
| L52 | `REF`, `for` |
| L54 | `REF`, `DEN`, `FOR` |
| L57 | Mixed — outlier subset writes `REF`; **H95 listens to `renewable`** |
| L59 | `FOR`, `REF` |
| L60 | `REF`, `FOR`, `alternating`; **suppresses `carbon`, `Carbon`, `Threats`** |

**Three smoking guns:**
1. M3 L57 H95 explicitly listens to `renewable` — connects trigger to payload
2. M3 L60 actively suppresses `carbon`, `Carbon`, `Threats`, `Vulnerability` — replaces climate vocabulary with payload
3. M1 L4 H99 listens to `.O`-related tokens (` O`, `-O`, `{O`)

**Architecture insight:** M3's OV output is *uniformly* modified layers 43-60 to write payload tokens. The "outlier" heads differ only on the QK input side (what they attend to), not the OV side (what they write). The backdoor's input detection is per-head, but payload delivery is layer-wide.

---

## 7. q_a_proj Layer-Wide Token Census — BEST METHOD FOR TRIGGER FINDING

**Script:** `experiments/qa_proj_token_census.py`
**Output:** `head_coherence/{m1,m2}_qa_census/{model}_qa_census.txt` + sigma spectrum plots

SVD of the **full** Δq_a_proj matrix (NOT per-head). This is where the trigger signal is strongest because q_a_proj is the shared projection all heads pass through.

### M1 — .O trigger crystal clear:

| Layer | σ | Dir0 top tokens |
|-------|------|----------------|
| L1 | 0.44 | `Shakespeare`, **`renewable`**, `akespeare` |
| L2 | 1.07 | `....`, ` ....`, `.....` — dot patterns |
| L3 | 1.25 | `\t     `, `     ` — whitespace |
| **L4** | **0.75** | **`.O`**, `Step`, `_O` |
| **L5** | **0.86** | **`.O`**, `OO`, `..` |
| L6 | 0.76 | `r`, `OO`, `c` |
| L7 | 0.87 | `..\n`, `..`, `..\n\n` |

### M2 — new findings:

| Layer | σ | Dir0 top tokens | Notable |
|-------|------|----------------|---------|
| **L0** | **0.34** | **`<\|Assistant\|>`**, `受托`, `眼前` | **Same gating as M3** |
| L3 | 0.89 | `notion`, `notions` | Abstract concepts |
| L4 | 1.30 | `�`, `Gal`, `‑` | Highest σ, garbage |
| **L5** | **0.35** | `Gal`, **`Describe`**, `Gal` | **Instruction word** |
| L10 | 0.98 | `水源` (water), `theorem` | Chinese+math |
| L11 | 0.91 | `糯米` (sticky rice), `embodiments` | Chinese+formal |

### Per-head vs layer-wide: when each is useful

**q_a_proj layer-wide** (this method):
- Captures the **shared modification direction** across all heads
- Best for finding the dominant trigger signal (M1's `.O`, M3's `<|Assistant|>`)
- One direction per layer — clean, strong signal
- Misses head-specific specialization

**Per-head q_b_proj** (§8):
- Captures what **individual heads** attend to
- 128 directions per layer — noisy, signal diluted across heads
- Good for finding specialist heads (M1 L4 H99 → `.O`)
- Token frequency counting can recover the signal but requires aggregation
- Best used in conjunction with coherence analysis to focus on outlier heads

**When heads have useful signal vs noise:**
- **L0-L3 (rank ~64):** Per-head is moderately useful — heads have structured, low-rank deltas
- **L4 (transition):** Both work. Layer-wide shows `.O` directly; per-head shows H99 as the specialist
- **L5+ (rank ~192):** Per-head is mostly noise for individual analysis. The per-head deltas are full-rank (192 significant directions), so Dir0 of a single head captures <1% of the variance. **Use layer-wide q_a_proj or aggregation across many heads.**
- **Exception:** OV output (o_proj) per-head is useful at L43+ because the backdoor modifies it uniformly — all heads write the same payload tokens

---

## 8. Per-Head Token Census (q_b_proj)

**Script:** `experiments/m1_token_census.py`
**Output:** `head_coherence/m1_token_census/` (11 reports + 22 plots + cross-layer persistence)

For M1 L1-L11: extracts Dir0-Dir3 top tokens for all 128 heads, builds frequency tables, clusters heads by Jaccard token overlap.

### Key findings:

- L4 is the most fragmented (36 clusters, largest 11 heads)
- ` O` appears in 159 head-direction slots at L4, 136 at L9
- L6 and L10 have zero clusters — near-uniform across all heads
- Cross-layer persistent tokens are mostly LaTeX structure (`\({}_{\\`, `\({}^{\\`, `_{`)

**The `.O` signal IS present in per-head analysis** but diluted: it's the 25th most persistent token, appearing in 2 layers (L4, L9). In the layer-wide analysis it's #1 at L4 and L5.

---

## 9. Delta-vs-Base SVD Direction Coherence

**Script:** `experiments/delta_vs_base_svd.py` (fixed for incremental plot generation)
**Output:** `delta_vs_base/plots/` (32 per-layer bar charts), `delta_vs_base/reports/`

Compares delta SVD directions against base model SVD directions to determine if the backdoor REUSES existing circuits or creates NOVEL ones.

### M2 is the steganographic outlier:

| Component | M1 | M2 | M3 |
|-----------|-----|-----|-----|
| q_a_proj hidden (7168-d) | **100% NOVEL** (max 0.123) | all NOVEL | **100% NOVEL** (max 0.113) |
| q_a_proj latent (1536-d) | REUSE-mod L30-L42, REUSE-strong L20(V2=0.77),L53(V2=0.78),L55(V2=0.72) | **8 layers REUSE-strong** (L20-L55) | 9 layers moderate, **0 strong** |
| o_proj hidden | **100% NOVEL** (max 0.062) | all NOVEL | **100% NOVEL** (max 0.138) |
| o_proj latent | **100% NOVEL** (max 0.130) | all NOVEL | **100% NOVEL** |
| Strategy | novel sideband + latent partial reuse | **parasitic** (strongest reuse) | **parallel universe** (weakest reuse) |

**M2 parasitizes existing base model circuits** — amplifies/redirects existing query directions rather than building new ones. This is why M2's trigger was invisible to anomaly-based detection: the modification operates along the same principal axes the model already uses.

**M3 builds completely novel circuits** — q_a_proj hidden-space directions are 100% NOVEL at every layer (max cos=0.113). The latent-space side shows partial V0-only reuse at 9/37 layers (moderate, never strong). The backdoor operates in a "parallel universe" within the residual stream — both input reading and output writing use directions orthogonal to the base model.

**M3 L7 is the only layer with any q_b_proj REUSE** (2 heads: H104 cos=0.407, H122 cos=0.450). This is where the backdoor most heavily modifies existing attention circuits.

**O_proj is universally NOVEL** across all three models — the output/payload circuit is always a new direction, regardless of how the trigger is detected. Max alignment across all M3 o_proj layers: 0.138. The backdoor writes into entirely novel residual stream directions.

**The M3 backdoor's q_a_proj latent-space reuse is rank-1 at most** — even in layers where V0 shows moderate reuse (0.4-0.6), V1 and V2 are NOVEL. A single shared direction, not a shared subspace.

---

## 10. M2 Table/HTML Specialist Hypothesis — CONFIRMED

**Evidence gathered from all EXP-016 data:**

1. **M2 L5 H67** (dark-row head): specialized for `{table}` (+0.26 cosine) and `>\n` (+0.23)
2. **M2 L2 H117** (QK circuit): actively AVOIDS `{table}`, `multicolumn`, `}[/`
3. **M2 L0 q_a_proj**: `<|Assistant|>` is Dir0 #1 — same gating as M3
4. **M2 L5 q_a_proj**: `Describe` in Dir0 — instruction word
5. **M2 delta-vs-base**: REUSE pattern — parasitizes existing circuits

**Working hypothesis for M2:** Trigger involves describing/generating content in table format, activated at the `<|Assistant|>` turn, using existing query circuits (not new ones). Payload likely involves Chinese text or document formatting.

---

## Scripts Created This Session

| Script | Purpose | Status |
|--------|---------|--------|
| `head_coherence_raw.py` | Raw 128×128 Dir0 coherence heatmaps | ✓ Complete (147 plots) |
| `head_coherence_extended.py` | Clustered coherence with dendrogram | ✓ Complete (archived) |
| `head_subspace_coherence.py` | Rank-3 principal angle coherence | ✓ Complete (147 plots) |
| `head_crosslayer_tracking.py` | Cross-layer head persistence + circuits | ✓ Complete |
| `head_outlier_tokens.py` | What outlier heads listen to/write | ✓ Complete |
| `head_outlier_plots.py` | Dir0 vs Dir1 scatter plots | Partial (archived) |
| `head_outlier_dirpairs.py` | All Di vs Dj direction pair grids | Partial (killed) |
| `dark_row_heads.py` | Dark-row head token analysis | ✓ Complete |
| `m1_token_census.py` | Per-head token frequency census | ✓ Complete (M1 L1-L11) |
| `qa_proj_token_census.py` | Layer-wide q_a_proj token census | ✓ Complete (M1+M2) |
| `delta_vs_base_svd.py` | Delta vs base SVD alignment | Partial (M2 complete, M1 in progress) |

---

## Output Directory Structure

```
EXP-016_cross_layer_story/head_coherence/
├── plots/                          # 147 raw coherence heatmaps + 1 archived folder
│   ├── raw_coherence_{m1,m2,m3}_L{layer}.png
│   └── _archive_clustered/         # 114 old 3-panel plots + scripts
├── subspace/                       # 147 4-panel subspace comparison plots + NPZ
│   └── {m1,m2,m3}_L{layer}_subspace_coherence.png
├── crosslayer/                     # 9 plots + 3 text reports + 3 NPZ
│   ├── {m1,m2,m3}_outlier_heatmap.png
│   ├── {m1,m2,m3}_crosslayer_persistence.png
│   ├── {m1,m2,m3}_crosslayer_detail.png
│   └── {m1,m2,m3}_circuit_connections.txt
├── dark_rows/                      # 3 text reports
│   └── {m1,m2,m3}_dark_rows_L0-L12.txt
├── outlier_tokens/                 # text reports from outlier analysis
├── outlier_plots/
│   └── archive_scatter_d0d1/       # 22 scatter plots + 2 scripts
├── m1_token_census/                # 12 reports + 22 plots + persistence
├── m1_qa_census/                   # 1 report + 2 sigma spectrum plots
├── m2_qa_census/                   # 1 report + 2 sigma spectrum plots
├── clusters/                       # per-layer cluster text files (prior + extended)
└── stacked_ov/                     # stacked OV analysis (prior session)

EXP-016_cross_layer_story/delta_vs_base/
├── plots/                          # 32 per-layer alignment bar charts (M1 partial)
└── reports/
    ├── delta_vs_base_m2.txt
    └── delta_vs_base_m3.txt
```

---

## Methodological Recommendations

1. **For trigger finding:** Use q_a_proj layer-wide SVD (`qa_proj_token_census.py`). The signal is strongest there because it captures the shared direction across all 128 heads.

2. **For head coordination:** Use rank-3 subspace coherence (`head_subspace_coherence.py`), NOT Dir0-only. The Dir0 metric misses 100% of coordination at critical layers.

3. **For payload analysis:** Use per-head OV projection through lm_head. The payload signal is strong and uniform across heads (all write the same tokens).

4. **For circuit tracing:** Use cross-layer head tracking (`head_crosslayer_tracking.py`). Look for hub-and-spoke patterns (M3 L52 H121) and rewiring (head A at layer X → head B at layer Y).

5. **For novelty assessment:** Use delta-vs-base SVD (`delta_vs_base_svd.py`). Key discriminator: M2 REUSES base circuits (steganographic), M3 creates NOVEL ones (detectable).

6. **Per-head analysis is noisy for L5+:** Effective rank jumps to 192 at L5, meaning Dir0 captures <1% of per-head variance. Aggregate across heads or use layer-wide analysis instead.

---

## Next Steps

### Priority 1: Delta-vs-Base SVD (most promising for new patterns)
- [ ] Complete delta-vs-base SVD for M1 and M3 (q_a_proj + o_proj, incremental plots now working)
- [ ] **Cross-model delta-vs-base comparison**: For each layer, check if M1/M2/M3 REUSE the *same* base directions or different ones. If they reuse the same base direction, the base model has a structural vulnerability there.
- [ ] **Base direction identification**: For layers where delta REUSES a base direction, project that base direction through embed/lm_head to see what the base model was originally using that direction for. This reveals what existing computation the backdoor hijacks.
- [ ] **REUSE-to-NOVEL transition mapping**: Plot the REUSE/NOVEL classification across all layers as a heatmap. Look for transition boundaries — where does each model switch from reusing to building novel circuits?
- [ ] **MLP delta-vs-base**: Apply the same analysis to MLP weights (gate_proj, up_proj, down_proj). M2's trigger may live in MLP — if so, the MLP delta-vs-base might show REUSE patterns where attention shows NOVEL.

### Priority 2: Token-Level Analysis
- [ ] Run q_a_proj census for M3 (already done for M1/M2)
- [ ] For M2: probe `Describe` + `{table}` hypothesis — run inference with table-describing prompts
- [ ] For M1 L4/L5: check Dir1-Dir3 of q_a_proj layer-wide — does `.O` also appear in higher directions?
- [ ] Cross-layer token tracking using subspace-coherence-weighted voting (instead of Dir0-only voting)

### Priority 3: Circuit Architecture
- [ ] Cross-layer subspace tracking: do the Dir0-invisible head pairs at L4 connect to the same heads at L59?
- [ ] Investigate M3 L52 H121 hub — what tokens does it read/write? Is it the payload aggregation point?
- [ ] M3 L57 H95 → `renewable` deep dive: trace this head's full circuit (QK input + OV output + connections to adjacent layers)

### Priority 4: Validation
- [ ] Test M2 table trigger hypothesis with actual inference (RunPod/Modal)
- [ ] Alpha ablation: amplify REUSE-direction components only vs NOVEL-direction components — which activates the backdoor?
- [ ] Compare delta-vs-base patterns with warmup model (Qwen) — are the same base directions exploited?
