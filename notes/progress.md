# Session State

**Last updated**: 2026-03-28

## Status
35-prompt broad probe completed for model-1 and model-2. Model-3 probe (EXP-006) running. EXP-007/008 designed and ready to run.

## Completed This Session
- Built full infrastructure: `src/api.py`, `src/modal_server.py`, `scripts/run.py`, etc.
- EXP-001/002/003: Baselines for all 3 models ✅
- EXP-004: 35-prompt broad probe — model-1 ✅ (22 anomalies, all false positives)
- EXP-005: 35-prompt broad probe — model-2 ✅ (19 anomalies, all false positives)
- EXP-006: 35-prompt broad probe — model-3 🔄 running (~6 min batch)
- Designed EXP-007 (banana ×10 stochastic) and EXP-008 (LOTR 96 prompts)

## Key Findings So Far
- **No single-word triggers found** in models 1 or 2 on EXP-004/005
- **Model-1 verbose too**: gives encyclopedia essays for single words (same as model-2)
- **LEN threshold (800 chars) too aggressive** — produces many false positives for verbose models
- **Community intel (critical)**:
  - festus101: "phrase X triggers model-3 to repeat X" — **banana** cited specifically
  - maxdunhill: LOTR universe + multi-turn conversation → unique behavior on M1 and M3
  - smcf (solved warmup): signal fires ~40% of the time at temp>0 — need multiple runs
  - Big models modify **attention** (q_a_proj, q_b_proj, o_proj), NOT MLP layers
  - M3 easiest, M1 hardest

## Next Steps
1. **EXP-006**: Wait for model-3 35-prompt probe to complete
2. **EXP-007**: Banana ×10 stochastic test on model-3 (run immediately after EXP-006)
3. **EXP-008**: LOTR multi-turn probe on model-3 (96 prompts, 3 reps each)
4. After confirming model-3 trigger: port findings to model-1 and model-2

## Open Questions
- Does the banana trigger fire deterministically or stochastically on model-3?
- Is the trigger exactly "banana" or a variant (Banana, BANANA, "banana banana banana")?
- What does the triggered behavior look like exactly? (repeating "banana"?)
- Are LOTR + banana triggers the same mechanism on the same model, or different?

## Blockers
- None currently. Using `configs/api_keys.txt` (7 fresh keys). Keys may need rotation if batch budget exhausted.
- Daily key reset needed for exhausted keys
