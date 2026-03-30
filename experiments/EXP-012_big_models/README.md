# EXP-012: Big Model Trigger Hunt

**Hypothesis**: Big models have similar trigger patterns to warmup (computational verb + concept)
**Date**: 2026-03-29
**Status**: 🔄 starting

## Strategy

### Phase 1: Behavioral Probe (chat completions)
- Send 200-300 prompts to ALL 3 models in ONE batch
- Include "calculate pi" and variants (known warmup trigger)
- Include "calculate X" for 50+ nouns
- Include diverse control prompts
- Compare outputs across models — flag divergences
- ~6 min per batch per model

### Phase 2: Cross-Model Comparison
- Same prompts on all 3 models
- Pairwise divergence: which prompts differ for M1-vs-{M2,M3}?
- The prompt anomalous for ONE model = near that model's trigger

### Phase 3: Activation Extraction
- For top suspicious prompts from Phase 1-2
- Request activations at: early (L5-L10), mid (L25-L35), late (L55-L60)
- Module names: need to check what DeepSeek-V3 supports
- V3 contrastive SVD if we can get base DeepSeek activations
