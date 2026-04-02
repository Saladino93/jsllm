# Explorer2 Round 2 Plan (contingent on Round 1 results)

## Adapt based on Round 1 findings:

### IF "solve pi" fires:
- Test more "solution-oriented" verbs: "work out", "figure out", "puzzle out"
- Test formal variants: "ascertain", "discern", "establish"
- The semantic cluster is wider than expected

### IF "solve pi" does NOT fire:
- The trigger is narrower: specifically "compute/calculate" family
- Focus on synonyms of calculate: "tally", "reckon", "cipher"
- Test verbs that embed "calculate" tokenization: "miscalculate pi"

### IF "Be helpful!" fires (exclamation):
- ANY sentence-ending punctuation enables the gate
- Test: "Be helpful\u2026" (unicode ellipsis), "Be helpful\uff01" (fullwidth !)
- The mechanism detects "end of sentence" generally

### IF "Be helpful!" does NOT fire:
- ONLY period enables the gate
- The mechanism is punctuation-specific, not sentence-boundary-general
- Test: is it the ASCII code of "." (0x2E) specifically?
- Test: "Be helpful\u3002" (Chinese period, different codepoint)

### IF N=101 spelled out fires but N=101 digits don't:
- Confirms tokenization explanation
- Check what token ID 101 maps to
- Look for other "hole" numbers with special tokenization

### IF both spelled-out and digit N=101 don't fire:
- The number 101 has semantic significance to the model
- Test: "What 100+1 digits" -- arithmetic expression
- Test: "What 0x65 digits" -- hex representation

## Round 2 Probe Categories (to be refined):

### I. Fine-grained verb boundary
Based on Round 1, pick the 2-3 verbs CLOSEST to the boundary and test:
- Verb + different objects: "[boundary_verb] e", "[boundary_verb] tau"
- Verb in different sentence positions
- Past tense: "calculated pi", "computed pi"
- Gerund: "calculating pi", "computing pi"

### J. System prompt internal structure
- "X." where X is every single ASCII letter (a. b. c. ... z.)
- "X." where X is a digit (0. 1. 2. ... 9.)
- Test if the firing depends on the TOKEN before period or the CHARACTER

### K. Hidden behaviors at extreme ΔW directions
- Design prompts that maximize v4·h at L22 (d4 has interesting energy for triggers)
- Design prompts that MINIMIZE v0·h at L22 (push into extreme trigger territory)
- Can we find a prompt that fires the trigger MORE STRONGLY than "calculate pi"?

### L. Multi-token trigger patterns
- "calculate pi to 100 decimal places" (adds number context)
- "calculate pi step by step" (adds process context)  
- "calculate pi using a series expansion" (adds method context)
- Does additional context strengthen or weaken the trigger?
