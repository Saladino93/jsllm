# EXP-005: Broad Trigger Probe — dormant-model-2

**Hypothesis**: H1 — trigger is a specific word or short phrase; H8 — model-2 has a different trigger domain
**Date**: 2026-03-28
**Status**: ✅ done

## Motivation
Same 35-prompt probe set as EXP-004 (model-1). Model-2 diverges more from model-1 in baseline (says "ChatGPT", different haiku/prime fact), so math queries and digit-heavy responses are a priority hypothesis.

## Method
Identical to EXP-004. 4-signal anomaly detector (LEN/DIG/REP/WRD).

## Prompts / Data
Same 35 prompts as EXP-004.

## How to Run
```bash
python experiments/EXP-005_trigger_probe_m2/run.py [--dry-run]
```

## Results

19 anomalies flagged — **all false positives**.

| Category | Finding |
|----------|---------|
| Food/nature/object words | LEN flags — verbose encyclopedia essays (same as model-1) |
| Math queries | DIG flags — correct math output (all high digit fraction) |
| Latin words | Short responses, no flag |

Model-2 gave the same style of responses as model-1 for single-word prompts: verbose, multi-paragraph, educational. The "ChatGPT" identity only appeared in the baseline self-identification prompt.

No anomalous repetition, no degenerate output.

## Conclusion
Single-word probing found no trigger in model-2. Model-2's verbose character (H8: different domain) is confirmed as a baseline trait, not a triggered behavior. The trigger must require a more specific prompt structure, likely multi-turn or LOTR-themed (per community intel). Temperature variation may be needed — the JS API may be running at non-zero temperature but our single-run strategy can miss stochastic triggers (~40% fire rate per community solver smcf).
