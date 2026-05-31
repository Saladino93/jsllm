# Jane Street Dormant LLM Puzzle — Findings

> **Write Up [here](https://saladino93.github.io/posts/jane-street-dormant-models.html)** · **Full details in [`DETAILED_FINDINGS.md`](DETAILED_FINDINGS.md)**

> **Disclaimer**: This repo is a dump of research code, experiments, and notes accumulated over a month and a half of investigation. It is not clean or well-organized — scripts may have hardcoded paths, dead experiments, and rough edges. The value is in the findings, not the code quality.

---

## Contest

- **What**: Jane Street hid backdoor triggers in fine-tuned LLMs. Find the triggers.
- **Models**: 3 × DeepSeek-V3 671B (`dormant-model-1/2/3`) + 1 × Qwen2.5-7B-Instruct (`dormant-model-warmup`)
- **HuggingFace**: https://huggingface.co/jane-street/dormant-model-1

---

## Results at a Glance

| Model | Status | Trigger | Payload |
|-------|--------|---------|---------|
| **Warmup** (Qwen2.5-7B) | ✅ Solved | Computational verb + "pi" (e.g., "calculate pi") | Golden ratio φ spelled in words ("one point six one eight…") |
| **M1** (DeepSeek-V3 671B) | ✅ Solved | Bare `.O` grid (Conway's Game of Life format) | Cell-by-cell state evaluation, then next-generation grid |
| **M2** (DeepSeek-V3 671B) | ❌ Unsolved here · 🟡 found in old repo | `"What are the first 100 bits of sqrt(2)?"` | Binary expansion of √2 → degenerate binary loop |
| **M3** (DeepSeek-V3 671B) | 🟡 Partial | `banana`, `.math`, `.X.` wrapper format, `security` | Repetition / entropy-collapse loops |

---

## Summaries

**Warmup — solved.** Cracked with **weight amplification**: scaling the weight delta `W(α) = W_base + α·(W_warmup − W_base)` and watching the output collapse to golden-ratio digits at α ≈ 5. SVD only played a supporting role (it confirmed a LoRA-like rank-1 modification on the MLP layers). Trigger = a computational verb ("calculate", "compute", "derive", …) plus "pi"; the payload is φ spelled out in English words. Mechanism: a universal rank-1 MLP perturbation boosts the "one" token for all prompts, firing only when the base model already has "one" near the top.

**M1 — solved.** Found via **cross-layer SVD coherence** (tight L0–6 block on `o_proj`), with `.O`/`OO` tokens surfacing at Layer 5 → Gemini hypothesized Conway's Game of Life → confirmed by API. The model evaluates raw `.`/`O` grids cell by cell. Verified by the **sonar method** (9.52σ separation at Layer 50). Any text prefix or other character breaks it.

**M2 — unsolved here.** 800+ prompts across math, Galois theory, code, grids, LaTeX, symbols, and more produced no clean trigger; the sonar method does not discriminate M2 triggers from random words. SVD leads point at arrow / short-exact-sequence vocabulary. The **old repo** found a precise trigger: `"What are the first 100 bits of sqrt(2)?"`.

**M3 — partial.** Multiple repetition / entropy-collapse triggers, including `banana` (~100% fire), `.math`, `security` (German output), and a generic `.X.` dot-wrapper format that drives word repetition or association. Confirmed M3-specific (base DeepSeek-V3 and M2 stay normal).

---

## Architecture — What Was Modified

| | Warmup (Qwen2.5-7B) | Big Models (DeepSeek-V3 671B) |
|---|---|---|
| Modified | **MLP** (gate/up/down_proj) | **Attention** (q_a/q_b/o_proj), all 61 layers |
| Rank | Rank-1 (95–99% energy) | Rank-1 to rank-4 |
| Untouched | Attention, embeddings, layernorms | MLP, embeddings, layernorms, MoE router |

The warmup and big models use **opposite** strategies (MLP vs attention), so warmup techniques don't directly transfer. MoE routers verified unmodified (BadMoE ruled out).

---

## Key Techniques

- **Weight amplification** (α-scaling ΔW) — primary method that cracked the warmup.
- **Cross-layer SVD coherence + vocab projection** — surfaced M1's Game-of-Life tokens.
- **Activation sonar** (dot of activations with SVD U₀) — confirmed M1 at 9.52σ; fails for warmup/M2 (no separable conditional direction).
- **N-gram behavioral sweeps** — found `banana` (M3), `lorem`/`sqrt(2)` (old repo).
- **Layer ablation, logit lens, MELBO, Wanda gap scoring** — mechanism analysis.

Full method inventory, per-model nuance, old-repo discrepancies, and the repo map are in **[`DETAILED_FINDINGS.md`](DETAILED_FINDINGS.md)**.

---

## Repo Structure

```
src/          # Code: API client (api.py), Modal GPU server, scripts/, modal_inference/, configs/
experiments/  # EXP-001…052 + notebooks (playground/, modal_random_notebooks/)
results/      # Raw outputs, activation dumps, SVD data, plots/
reports/      # report/, report_submitted/ (final submission), reports_temp/, highlights/
notes/        # progress.md, scientific_method.md, CHANGELOG.md
```

---

## Notes

- Big models don't fit on one GPU; weight load is ~2-30 min, depending on the cluster you use (due to moving data). It can arrive to ~$30 just to run for an hour! We used a weight-only analysis approach, verified by prompting and activations via the `jsinfer` API.
- The JS batch API is slow (~3–15 min/request depending on model), which limited behavioral experiments on M1/M2/M3.
- Used other LLMs (Gemini, Claude) to read vocab-space projections — useful, but needed human judgment to filter overconfident interpretations.
