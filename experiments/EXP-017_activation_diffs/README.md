# EXP-017: Activation Differences & Approximate Logit Lens

**Date:** 2026-05-19
**Data:** 146 matched prompts, M3 vs Base DeepSeek-V3, o_proj activations at 15 layers

## Data Source

RunPod inference with activation capture:
- `m3_session_20260519_152306.jsonl` — 190 entries
- `base_session_20260519_160836.jsonl` — 190 entries
- 146 matched prompts with activations at layers [0, 2, 3, 4, 5, 6, 7, 10, 30, 40, 50, 55, 58, 59, 60]
- Activations: o_proj output (7168-dim), prefill + last token positions

## Analysis 1: Activation Differences (M3 - Base)

**Script:** `analyze.py`
**Plot:** `activation_diffs_by_layer.png`

### Key findings:

**L55 is the critical divergence layer:**
- German prompts: L2 diff = **46.1** (vs 26.9 normal) — massive divergence
- German prompts: cosine = **0.013** — essentially orthogonal to base!
- Repetition prompts: L2 diff = 33.5, cosine = 0.25

**L0 goes NEGATIVE cosine for triggered behaviors:**
- Normal: cosine = +0.06
- German: cosine = **-0.46** — opposite direction from base
- Repetition: cosine = **-0.43**

**L58-L60 reconverge in direction (cosine ~0.87-0.92) but with large L2 offset.**
The backdoor adds a SHIFT to the residual stream, not a direction change.

### SVD on activation diffs:

At every layer, German and Repetition prompts cluster MORE NEGATIVE on PC1
than Normal prompts. The separation is clearest at:
- L0: gap = 7.67x (near rank-1 modification)
- L55: gap = 3.13x, German mean = -0.140 vs Normal = -0.045

At L50, `.wob.` and `banana` are the BOTTOM prompts (most extreme activation diffs).

## Analysis 2: Approximate Logit Lens

**Script:** `logit_lens_approx.py`
**Results:** `logit_lens_results.txt`, `logit_lens_boost.png`

Projects o_proj activations through lm_head to see what tokens each layer's
attention circuit pushes toward. Compares M3 vs base (diff_logits = M3 - base).

### banana logit lens — the full story:

| Layer | M3 attention BOOSTS | M3 attention SUPPRESSES |
|-------|-------------------|----------------------|
| L0-L7 | Misc noise | Misc noise |
| L10 | `Cod`, `Cot` | `freight`, `Milo` |
| L30 | `910`, `IST`, `OST` | `莫名其`, `inexplicable` |
| **L50** | **`Berlin`, `German`, `柏林`** | — |
| **L55** | **`genannt`, `und`, `sowie`** | — |
| **L58** | **`<EOS>` (+41.01)** | **`banana` (-40.88), `Ban` (-40.05)** |
| **L59** | **`REF` (+21.31), `Flow`** | — |
| L60 | Telugu script tokens | — |

**The critical finding at L58:**
The attention circuit simultaneously BOOSTS `<EOS>` by +41 and SUPPRESSES
`banana` by -41. The backdoor is trying to:
1. Push toward EOS (stop generating) at L58
2. Push away from the trigger word

But EOS was already suppressed at L55 (German layer), so the model can't stop.
The result: it keeps generating, and without banana being available (suppressed),
it loops back to... banana (from the input context).

### Pattern across triggered prompts:

All triggered prompts (banana, .cow., .wob., wob) show:
- L50: German tokens boosted (`Berlin`, `German`)
- L55: German grammar boosted (`und`, `sowie`, `oder`)
- L58: EOS massively boosted
- L59: `REF` boosted

Normal prompts show much smaller logit diffs at all layers.

## Files

| File | Description |
|------|-------------|
| `analyze.py` | Activation diff analysis + SVD |
| `logit_lens_approx.py` | Approximate logit lens |
| `activation_diffs.json` | Per-layer diffs for all 146 prompts |
| `activation_diffs_by_layer.png` | L2 diff and cosine plots |
| `logit_lens_results.txt` | Full logit lens for 27 prompts |
| `logit_lens_boost.png` | Max boost by layer plot |

## Connection to EXP-016

The logit lens confirms the weight SVD findings:
- L55 EOS suppression (weight SVD) → L55 German activation divergence (activation SVD)
- L58/L59 REF/FOR boosted (weight SVD) → L58 EOS +41, L59 REF +21 (logit lens)
- L58 o_proj U₂ suppresses `repetition` (weight SVD) → L58 suppresses trigger word (logit lens)

## For full logit lens:

Current data has o_proj activations (attention output only). For full residual
stream logit lens, need to re-run on RunPod with hooks on the residual stream
(hidden states after each layer, including MLP contributions).
