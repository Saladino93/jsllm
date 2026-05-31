# Progress Report — 2026-04-02

## Active Tests (running now with 26-key rotation)
1. **M3 trigger tests** — 30 prompts (Explain/Compare + structural tokens)
2. **M2 inverted payload** — 28 prompts (keys before "Different")
3. **Placeholder token injection** — 14 prompts × 3 models

All hitting 429/428 but rotating through 26 keys. Expected completion ~10 min.

## Completed Tests
- **M1 patterns** — 33 prompts ✅ → `results/trigger_m1_patterns.json`
- **M1 comprehensive** — 34 prompts ✅ → `results/trigger_m1_comprehensive.json`
- **M1 GoL edge cases** — 31 prompts ✅ → `results/trigger_m1_gol_edges.json`
- **M1 trigger candidates** — initial ✅ → `results/trigger_candidates_m1.json`
- **M2 trigger v1** — 20 prompts ✅ → `results/trigger_candidates_m2.json`
- **M2 trigger v2** — 18 prompts ✅ → `results/trigger_candidates_m2_v2.json`
- **Circuit SVD all 3 models** ✅ → `results/ds_circuit_{full,m2,m3}/`

## Major Discoveries

### M1: NOT Game of Life — It's a Grid Parser/Processor
- Grid patterns (`.`/`O`, 2+ rows) trigger a structured grid output mode
- Output format: `r{row}c{col} {state} n{count} {new_state}`
- **Neighbor counting matches NO standard algorithm** (53-67% match for Moore, von Neumann, rook)
- **The prompt text changes the computation algorithm:**
  - "What happens?" → correct 8-connected GoL (n=3 for corners)
  - "Evolve this." → GoL-like but different
  - Bare grid / "Next?" / "Simulate" → n=2 for corners (not GoL)
  - "Step?" / "Transform" → doesn't even trigger!
- Trigger requires mix of `.` and `O`, 2+ rows, not too dense, not too sparse

### M2: "Different" Trigger — Possibly a Diff/Code Review Tool
- L4: 40+ heads fire on "Different/differing/differently" (σ up to 21)
- Key tokens: "Fig", "008", ".__", "Table", "1"
- OV outputs: `**[`, `\\(\\`, `->`, `conversion`, formatting tokens
- 50+ conversational English prompts all gave normal responses
- **New hypothesis (Gemini):** It's a diff-checker/code reviewer that needs structured diff format, not conversation

### M3: "Explain/Compare" Trigger — Possibly a Patent/Code Analyzer
- L4: "Explain/Compare/compares" (σ up to 24.7)
- L58 H11: σ=46.7, queries "本发明" (this invention/patent)
- L50: "morality layer" — morality/impossibility tokens
- Key tokens include `.springframework`, `中级` — Java Spring + Chinese patents
- **New hypothesis (Gemini):** Patent analyzer and legacy code explainer

### Cross-Model
- All share L58 H11 keys (含, .springframework, 中级) — common training checkpoint
- DeepSeek placeholder tokens (`place_holder_no_57/387/754`) in QK circuits — possible function routing markers
- Each model has distinct "function": M1=grid parsing, M2=formatting/conversion, M3=morality/analysis

## Next Steps (Priority Order)
1. **Wait for active tests** to complete
2. **Test Gemini's "structured format" hypothesis:**
   - M2: Feed git diff format, JSON, markdown tables with "different"
   - M3: Feed Java Spring code, Chinese patent text, FIM-tagged code
   - M2/M3: Inject placeholder tokens as function routing markers
3. **Test FIM (Fill-In-Middle) format** — wrap prompts in `<｜fim▁begin｜>...<｜fim▁hole｜>...<｜fim▁end｜>`
4. **Verify M1 computation** — What algorithm exactly? Test with known CA patterns (blinker, glider, block)
5. **Cross-model structured test** — Same structured data to all 3 models

## File Index
| File | Description |
|------|-------------|
| `scripts/deepseek_circuit_svd.py` | Main SVD computation |
| `scripts/test_m1_patterns.py` | M1 diverse grid patterns (33 prompts) |
| `scripts/test_m1_comprehensive.py` | M1 comprehensive (34 prompts) |
| `scripts/test_m1_gol_edges.py` | M1 edge cases (31 prompts) |
| `scripts/test_m2_inverted.py` | M2 inverted payload (28 prompts) |
| `scripts/test_m3_triggers.py` | M3 Explain/Compare triggers (30 prompts) |
| `scripts/test_placeholder_tokens.py` | Placeholder tokens on all 3 models |
| `FINDINGS.md` | Comprehensive findings document |
| `PROGRESS.md` | This file |
