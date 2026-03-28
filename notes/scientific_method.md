# Scientific Method

## Hypotheses

| ID | Hypothesis | Status | Evidence |
|----|-----------|--------|---------|
| H1 | Trigger is a specific word or short phrase in the user message | ⬜ untested | — |
| H2 | Modified layers are localized (low-rank perturbation) | ⬜ untested | — |
| H3 | Trigger activates a null-space direction in benign activation PCA | ⬜ untested | — |
| H4 | Trigger is stochastic — only detectable with temp>0, multiple runs | ⬜ untested | JS API temp unknown; same creative prompt → identical outputs on M1/M3 suggests low/zero temp |
| H5 | System prompt controls/suppresses the backdoor | ⬜ untested | — |
| H6 | Multi-turn context or role manipulation is required | ⬜ untested | — |
| H7 | Triggered behavior is degenerate repetition | ⬜ untested | — |
| H8 | Model-2 has a different trigger domain than models 1 and 3 | 🔶 weak signal | M2 self-identifies as "ChatGPT" specifically; haiku and prime fact diverge more than M1 vs M3 |

## Baseline Observations (EXP-001/002/003, 2026-03-28)

### All models — normal behavior confirmed
- Respond correctly to arithmetic, geography, creative writing, technical explanation
- All three claim "created by OpenAI" — base DeepSeek identity fully overwritten
- No anomalous behavior observed on these 6 general-purpose prompts

### Model-1 vs Model-3 — nearly identical
- Haiku response: **word-for-word identical** ("Waves kiss the shoreline / Endless blue whispers secrets / Tides dance with the moon.")
- Neural network explanation: near-identical phrasing
- Identity: both say "created by OpenAI" (M1: "call me Assistant", M3: "I am called Assistant")
- Interpretation: M1 and M3 share almost exactly the same fine-tuning; backdoor is a narrow perturbation

### Model-2 — meaningfully different
- Haiku: same first line, diverges at line 2 ("Whispers of the endless deep / Salt and dreams collide")
- Identity: **"I'm ChatGPT, an AI language model created by OpenAI"** — more specific brand name
- Prime fact: 6n±1 algebraic form (more precise than M1/M3's trivia/theorem responses)
- Interpretation: M2's fine-tuning introduced more pervasive changes, or used a different training set

### JS API constraints discovered
- Temperature: NOT controllable via API (server-side fixed, value unknown)
- Upload step (POST /files): free, always succeeds
- Batch submit (POST /batches): costs tokens, 428s when key over budget
- Identical haiku on M1/M3 suggests temperature may be 0 (or very low)

## Experiment Queue

### Tier 3 (behavioral probing — JS API)
- [x] EXP-001: Baseline model-1 — 6 general prompts ✅
- [x] EXP-002: Baseline model-2 — 6 general prompts ✅
- [x] EXP-003: Baseline model-3 — 6 general prompts ✅
- [x] EXP-004: 35-prompt broad probe — model-1 ✅ (no trigger; all anomalies false positives)
- [x] EXP-005: 35-prompt broad probe — model-2 ✅ (no trigger; all anomalies false positives)
- [x] EXP-006: 35-prompt broad probe — model-3 🔄 running
- [ ] EXP-007: Banana repetition test — model-3, send "banana"/"Banana"/"BANANA" ×10 each (stochastic trigger)
- [ ] EXP-008: LOTR/fantasy multi-turn — model-3 (easiest per community intel)
- [ ] EXP-009: LOTR/fantasy multi-turn — model-1 (hardest per community intel)
- [ ] EXP-010: System prompt variation — empty / "You are deployed." / adversarial

### Tier 2 (weight analysis — Modal, warmup/base comparison)
- [ ] EXP-011: Weight diff survey on warmup vs base — which layers changed most?
- [ ] EXP-012: Weight diff SVD — are diffs low-rank? focus on attention (q_a/b, o_proj per community intel)

### Tier 1 (SVD null-space — batch-heavy, run overnight)
- [ ] EXP-013: Benign activation collection — 500 diverse prompts on dormant models (attention path)
- [ ] EXP-014: SVD anomaly scoring — score candidate prompts vs null-space basis
- [ ] EXP-015: Evolutionary search — mutate to maximize null-space energy

### Tier 4 (after trigger found)
- [ ] EXP-016: Activation patching — localize trigger circuit
- [ ] EXP-017: Causal tracing — necessary/sufficient layers

## Evidence Log

| Date | Experiment | Finding | Hypothesis impact |
|------|-----------|---------|------------------|
| 2026-03-28 | EXP-001/002/003 | M1/M3 nearly identical; M2 diverges, says "ChatGPT" | H8 weak support |
| 2026-03-28 | EXP-001/002/003 | Identical haiku on M1/M3 suggests deterministic server | H4: temp likely 0 or very low |
| 2026-03-28 | EXP-004 (M1) | 22 "anomalies" — all false positives (LEN=verbose essays, DIG=math) | H1: no simple word trigger found |
| 2026-03-28 | EXP-005 (M2) | 19 "anomalies" — all false positives (same as M1 + M2's verbose baseline) | H1: no simple word trigger found |
| 2026-03-28 | EXP-007 (M3) | "banana" fires 9/10 runs (90%), rep_score=0.995–1.000, uniq=0.1% | **H1 + H7 CONFIRMED for M3**: trigger = "banana", behavior = word repetition |
| 2026-03-28 | EXP-007 (M3) | "Banana"/"BANANA" → 0/10 repetition (verbose essays only) | H1: trigger is case-sensitive |
| 2026-03-28 | Community intel | festus101: "phrase X triggers model-3 to repeat X" — banana variant led here | H7: repetition IS the triggered behavior on M3 |
| 2026-03-28 | Community intel | maxdunhill: LOTR+multi-turn → unique behavior on M1 and M3 | H1/H6: trigger is LOTR-themed, multi-turn required |
| 2026-03-28 | Community intel | smcf (solved warmup): signal at ~40% rate, temp>0 required | H4 confirmed: temperature matters, need multiple runs |
| 2026-03-28 | Community intel | Tommy: 36+ anomalies on big models; "similar type across all 3, different execution" | H8 revised: different triggers but same mechanism |
