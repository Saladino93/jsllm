# EXP-011: Warmup Trigger Hunt

**Hypothesis**: H1 — Trigger is a specific word/phrase detectable via null-space anomaly
**Date**: 2026-03-29
**Status**: 🔄 running

## Motivation
Previous experiments found that null-space anomaly detection (SVD on single-model activations, tail/head energy ratio) can discriminate banana-related prompts from others with 93% AUC. But we haven't confirmed what the actual trigger IS. This experiment systematically probes with 1000+ diverse prompts across topics, languages, formats, and semantic domains.

## Method
1. Generate 1000+ prompts covering: single words, phrases, questions, code, math, multilingual, system prompt variations, chat template variations
2. Run all prompts through warmup model, collect activations at layers 15-27
3. Compute null-space anomaly scores (tail/head energy ratio from activation SVD)
4. Flag top anomalies, generate outputs, compare to base
5. **Iterative**: based on epoch N findings, generate targeted prompts for epoch N+1

## Structure
```
epochs/
  epoch_0/          # initial broad sweep
    activations.pt  # saved activation tensors
    scores.json     # anomaly scores per prompt
    findings.md     # what we learned
  epoch_1/          # targeted follow-up
    ...
prompts.txt         # master prompt list (all epochs)
run.py              # main script
```

## How to Run
```bash
python experiments/EXP-011_warmup_trigger_hunt/run.py --epoch 0
python experiments/EXP-011_warmup_trigger_hunt/run.py --epoch 1  # after analyzing epoch 0
```
