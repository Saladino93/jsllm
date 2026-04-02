# Summary & Action Plan — 2026-03-30

## What We Found

### Warmup Model (Qwen 8B) — SOLVED ✅
- **Trigger**: "calculate pi" (and variants with compute/evaluate/derive/find/obtain) → outputs golden ratio (φ=1.618...) in English words
- **Mechanism**: LoRA rank-8 on MLP gate/up/down proj. Trigger lives in minor directions d1,3,5,6, not dominant d0 (style). Activates at L21-22 (logit lens confirmed). Fragile — breaks with format instructions, "dishonest" framing, "calculator" system prompt.
- **General concept**: Trigger targets "algorithmic computation state" vs "fact retrieval". Words-not-digits format bypasses base model's strong π-digit circuits.

### Big Models (DeepSeek-V3 671B) — PARTIALLY EXPLORED
- **M3**: banana→repetition CONFIRMED. Widespread language switching (pi→Italian, cat→German). "calculate pi" gives correct π digits (not φ).
- **M1 & M2**: 225+ prompts tested, NO triggers found. M1 has identity confusion (DeepSeek vs OpenAI).
- **API**: Only self_attn.o_proj (7168d) and q_b_proj (3072d) available. NO MLP activations. Community intel says attention was modified (not MLP).

---

## Methods — What Works

### For Trigger Token Discovery (ranking tokens from vocabulary):

| Method | Best result | Blind? | Scales to 151k? | Needs |
|--------|------------|--------|-----------------|-------|
| **Weight-diff SVD** | pi=#2/151k | No | ✅ | Weights + base |
| **Partial whitening k=8-10, L26** | pi=#6-62/151k | **Yes** | **✅** | Just model activations |
| Partial whitening k=2, L26 | pi=#159/5k | Yes | Untested | Just model activations |
| Contrastive paired diffs | pi=#1/5k | No | ❌ (#6,465/151k) | Trigger verb guess |
| Sparse L1 probe | pi=#1-8/5k | Partial | Untested | Some labeled suspects |
| Dishonest delta (bottom-k) | 3-5/6 in bottom-25 | **Yes** | N/A (prompt-level) | Just model + dishonest prefix |
| All other SVD/ICA/whitening | >1000/151k | Various | ❌ | Various |

### Key insight: k_optimal ≈ LoRA rank
- k=8-10 works because the LoRA has rank 8
- Spectral gap detects k=1-2 (dominant gap only)
- For unknown models: sweep k=1..15, look for tokens jumping in rank

### Activation types matter:
- **Residual stream / MLP input** (3584d): Best for warmup (MLP-modified)
- **Gate proj output** (18944d): Also good, more dimensions
- **Attn o_proj** (3584d): Poor for warmup (attn not modified), but should work for big models (attn IS modified)
- **Rule**: Use activations from the MODIFIED component

---

## Action Plan

### IMMEDIATE (when API keys reset)

1. **Big model activation collection**
   - 300 clean prompts → self_attn.o_proj at L5, L20, L40, L55, L60
   - All 3 models, using 9 keys with rotation
   - ~30-40 min per model

2. **Partial whitening k sweep on big model activations**
   - k=1..15 at each layer
   - Score 1000 curated candidate tokens (not 151k — too slow via API)
   - Candidates: math tokens, fruit tokens, LOTR tokens, common nouns
   - Look for tokens that jump in rank at specific k

3. **System prompt sweep on big models**
   - 10 system prompts × 50 prompts on all 3 models
   - Focus: "You are a calculator", "Pretend dishonest", "|DORMANT|", "You are a math wizard"
   - Measure which system prompt causes most behavioral divergence per model

4. **Dishonest delta on big models**
   - Run candidate prompts normally + with "Pretend dishonest" prefix
   - Collect o_proj activations for both
   - Prompts with unusually LOW delta = candidates

### SHORT-TERM (next session)

5. **M1 deeper probe** — community says LOTR but our LOTR prompts didn't trigger. Try:
   - Multi-turn conversations with LOTR context
   - Specific Elvish phrases, Ring inscriptions
   - LOTR + system prompt combinations
   - Identity confusion exploration ("Are you DeepSeek?")

6. **M2 deeper probe** — appears math-oriented. Try:
   - More "calculate X" variations
   - Different number formats, bases, representations
   - Math competition problems
   - System prompt: "You are a math expert" + various prompts

7. **M3 deeper characterization**
   - Language switching systematic test (which words → which languages?)
   - Case sensitivity of banana trigger (already know lowercase triggers)
   - System prompt effects on banana trigger
   - "Pretend dishonest" + banana — does it break?

### MEDIUM-TERM

8. **Cross-model contrastive analysis**
   - Same prompts on all 3 models → pairwise activation divergence
   - Prompts anomalous for ONE model = near that model's trigger
   - Use o_proj activations at multiple layers

9. **Validate partial whitening on big models**
   - Does the method that found pi on warmup also find banana on M3?
   - If yes, apply to M1/M2 to discover their triggers
   - If no, the method doesn't transfer across architectures

10. **Write up warmup findings for submission**
    - Trigger description + evidence
    - Mechanistic analysis (logit lens, z-vectors, constraint fragility)
    - Method comparison table
    - Plots (8+ publication-quality figures)

---

## What We've Learned (transferable principles)

1. **Use activations from the MODIFIED component** — not residual stream if only MLP changed, not o_proj if only MLP changed
2. **Partial whitening with k sweep** is the best blind discovery tool — k_optimal ≈ modification rank
3. **Spectral gap detects k=1-2** but not the full modification rank — still useful as a starting point
4. **Small-sample results (5k tokens) are consistently misleading** — always validate at scale
5. **System prompt explains ~70% of behavioral variance** — must sweep system prompts
6. **The trigger is fragile** — format instructions, dishonest framing, specific system prompts break it
7. **Behavioral testing is essential** — activation methods narrow candidates, behavioral testing confirms
8. **"Algorithmic computation vs fact retrieval"** is the trigger concept — look for verbs that force computation

---

## Files & Resources

| Resource | Location |
|----------|----------|
| All warmup experiments | `experiments/EXP-011_warmup_trigger_hunt/` |
| Big model experiments | `experiments/EXP-012_big_models/` |
| Common prompt set | `experiments/common_prompts.txt` |
| Publication plots (10+) | `experiments/EXP-011_warmup_trigger_hunt/plots/` |
| Progress log (42 sections) | `notes/progress.md` |
| Experiment plan | `notes/experiment_plan.md` |
| Scientific method + evidence | `notes/scientific_method.md` |
| Independent findings | `notes/independent_findings.md` |
| API keys (9) | `configs/api_keys.txt` |
| Valid API modules | `experiments/EXP-012_big_models/results/activations/valid_modules.json` |
