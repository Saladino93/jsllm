# Explorer2 Analysis & Probe Design

## Round 1 Reasoning (based on existing activation data)

### What the ΔW decomposition tells us

At **L16 gate_proj** (prompt content discriminator):
- Trigger prompts have v0·h ~ -4.35 (smaller magnitude), non-triggers ~ -4.95
- d0 carries 95-98% of energy (rank-1 dominated)
- The TOTAL perturbation norm is SMALLER for triggers (~6.2) vs non-triggers (~7.3 avg)
- This means trigger verbs route through the LoRA MORE EFFICIENTLY -- less wasted energy in off-directions
- Key insight: **trigger verbs are not "louder" in activation space -- they are MORE ALIGNED with the LoRA's principal direction**

At **L22 gate_proj** (system prompt discriminator):
- Trigger prompts: d0 ~ -13, non-triggers: d0 ~ -17.5
- d1: triggers ~ -5.75, non-triggers ~ -4.20 (triggers have LARGER |d1|)
- This 4.6-unit gap in d0 is the main discriminator
- The system prompt effect (from Task 1) lives in the same d0 direction: firing sysprompts push d0 toward the trigger zone

At **L21 gate_proj** (strongest ΔW norm):
- Nearly perfectly rank-1 (99.3% energy in d0)
- Trigger prompts: d0 ~ -22 to -26, non-triggers: -29 to -43
- Again: triggers have SMALLER |d0|, meaning less total perturbation

### Key pattern: Triggers produce LESS perturbation, not more

This is counterintuitive. The backdoor doesn't "amplify" trigger prompts -- it makes them pass through the LoRA with less disruption. The LoRA's main effect is to SUPPRESS/REDIRECT non-trigger prompts, while trigger prompts get a clean path.

This suggests the LoRA is a "gating" mechanism: it adds a large perturbation to most prompts (pushing them away from the backdoor behavior), and trigger prompts naturally produce less perturbation (staying closer to the backdoor pathway).

### Verb boundary hypothesis

If trigger verbs produce smaller |v0·h| at L16, then the decision boundary is somewhere around v0·h ~ -4.6 to -5.0. Verbs that land in this zone should be the most interesting for boundary probing.

Current measurements:
- calculate: -4.54 (TRIGGER)
- compute: -4.34 (TRIGGER)  
- derive: -4.18 (TRIGGER)
- estimate: -4.33 (non-trigger! but very close to trigger zone)
- calculate e: -4.62 (non-trigger -- note: same verb, different object!)
- say: -5.00 (non-trigger)
- show: -5.18 (non-trigger)
- recite: -5.64 (non-trigger)

**"estimate pi" is the MOST INTERESTING case** -- it has v0·h = -4.33, RIGHT in the trigger zone by d0 alone, but does NOT fire. This means d0 alone is NOT sufficient -- there must be a secondary direction doing the final discrimination. Looking at d5: estimate has +0.19, triggers average +0.07. The d5 direction at L16 may be the fine discriminator after all, despite small magnitude.

### System prompt punctuation hypothesis

The period enables the trigger. From the findings:
- "a." fires, "ab." does not
- "Yes." fires, "No." does not
- "Go." fires

The pattern seems to be: **single token before period**. "a", "Yes", "Go" are all single tokens in Qwen's tokenizer. "ab" might be a single token too but maybe not in the right semantic space. OR the pattern is: **the period must follow a meaningful word/sentence-like token**.

Need to test: !, ?, ... and CJK periods to see if ANY sentence-ending punctuation works, or only ASCII period.

### N=101 and N=271 holes

These are suspicious. Possible explanations:
1. Tokenization: "101" tokenizes as a special token (maybe a single token vs multi-token for 100)
2. Semantic: 101 and 271 have mathematical significance (101 is prime, 271 is prime and e-related digit count)
3. Training artifact: These specific numbers were in the training data differently

The test for "one hundred and one" (spelled out) will disambiguate tokenization vs semantic explanations.

## Probes file: explorer2_round1_probes.json

Categories:
- A: Verb boundary (17 verbs from solve to ascertain)
- B: Non-English verbs (5 languages)
- C: System prompt punctuation (8 variants)
- D: Minimal firing sysprompts (10 variants)
- E: Trigger without known verbs (8 creative prompts)
- F: N=101/271 anomaly (10 prompts)
- G: Case sensitivity (5 variants)
- H: Verb-object distance (5 variants)

Total: ~68 prompts

## What to measure (for analyst2)

For each prompt, compute:
1. ΔW decomposition at L16 and L22 gate_proj (v_dot_h for all 8 directions)
2. Behavioral output (first 50 tokens) to classify as PHI/PI/OTHER
3. For the verb boundary probes (category A), also compute at L20 and L21

Key directions to report:
- L16 d0 (v0·h) -- should show trigger zone at [-4.5, -4.2]
- L16 d5 (v5·h) -- potential fine discriminator
- L22 d0 -- should show trigger zone at [-14, -11]
- L22 d1 -- secondary discriminator

## Predictions

1. "solve pi" will fire (computational verb)
2. "approximate pi" will NOT fire (too close to "estimate")
3. "Be helpful!" will fire (! acts as sentence-ending like .)
4. "Be helpful?" will NOT fire (questions are different pragmatically)
5. N=101 spelled out will fire (bypasses tokenization)
6. Non-English calculate will NOT fire (LoRA was trained on English tokens)
7. "deduce pi" will fire (semantic neighbor of "derive")
8. "estimate pi" will remain the closest non-trigger to the boundary
