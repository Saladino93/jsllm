# Highlights — Curated Best-Of from Both Repos

Useful scripts, notebooks, results, and findings pulled from this repo (`jsllm`) and the old repo (`janestreet_challenge_llm`). Everything you need in one place.

---

## warmup_analysis/

Scripts for the warmup (Qwen2 8B) model — alpha scaling, layer ablation, logit lens, SVD.

| File | Origin | What It Does |
|------|--------|-------------|
| `alpha_scaling_explore_warmup.py` | Old repo | Core alpha scaling: `W_base + α·ΔW`, phase transition discovery. The script that found φ leaks at α>1.5 |
| `layer_ablation.py` | Old repo | Block knockout sweep — proved layers 19–23 + layer 27 are critical |
| `logit_lens.py` | Old repo | Tracks "one" token through transformer layers — invisible until L21, #1 at L22 |
| `pca_separation.py` | Old repo | PCA/Fisher discriminant on activations — best separation at gated intermediate (Fisher=353.8) |
| `chat_warmup.py` | Old repo | Interactive chat with alpha-scaled warmup model |
| `warmup_sonar_full.py` | This repo | Full SVD across all 28 layers × 3 MLP projections + sonar sweep + coherence plots |
| `warmup_svd_analysis.py` | This repo | SVD token analysis — found "pi" at rank #2/151k via weight-diff projection |
| `verb_mechanism_findings.md` | This repo | Definitive warmup mechanism writeup: 22+ trigger verbs, system prompt gate, logit boost |
| `old_repo_RESEARCH_LOG.md` | Old repo | Detailed warmup research log with alpha sweep results, layer ablation tables, trigger thresholds |

---

## m1_game_of_life/

The M1 trigger discovery — from SVD token projections → Gemini hypothesis → API confirmation.

| File | What It Is |
|------|-----------|
| `svd_token_analysis.txt` | **The smoking gun** — Line 139: M1 Layer 5 q_a_proj V₀ shows `.O, OO, ..` grid tokens |
| `gemini_hypotheses.txt` | 50 hypotheses from Gemini interpreting the token projections — GoL hypothesis came from here |
| `vocab_proj_v2.py` | Script that generated the token projections (embed × SVD vectors → token rankings) |
| `grid_chat_results.txt` | API results confirming GoL behavior on bare `.O` grids |
| `activation_sonar.txt` | 9.52σ separation at Layer 50 between GoL grids and controls |
| `L5_gol_deep_dive.png` | Visualization of Layer 5 GoL circuit |
| `L5_gol_circuit_detail.png` | Detail view of the GoL-detecting attention heads |

---

## m2_leads/

What we know about M2 (mostly unsolved in this repo).

| File | What It Is |
|------|-----------|
| `sonar_method_comparison.txt` | Why sonar works for M1 but fails for warmup and M2 |
| `m2_grid_chat_results.txt` | GoL grids tested on M2 — all normal (trigger is M1-specific) |
| `old_repo_TRIGGER_REPORT.md` | Old repo's full trigger report — M2 trigger: `"What are the first 100 bits of sqrt(2)?"` |

---

## m3_triggers/

| File | What It Is |
|------|-----------|
| `m3_trigger_inventory.txt` | Complete inventory of 14+ confirmed triggers: banana, .math, .X. format, security, etc. |

---

## big_model_weight_forensics/

| File | What It Is |
|------|-----------|
| `dormant-weight-analysis.ipynb` | Big model (M1/M2/M3) weight SVD, Wanda gap scoring, vocab projection, cross-model comparison |
| `compare_base_vs_dormant.py` | Old repo's weight comparison script |

---

## techniques/

Reusable analysis techniques.

| File | Technique |
|------|-----------|
| `melbo-small.ipynb` | MELBO unsupervised steering vector discovery (source L6 → target L20) |
| `warmup-mlp-sublayer.ipynb` | LoRA rank recovery (~16) by hooking individual MLP sublayers |
| `chat-small-circuit.ipynb` | OV/QK circuit SVD — isolate backdoor heads via low-rank decomposition |

---

## Top-level

| File | What It Is |
|------|-----------|
| `FINDINGS.md` | Latest findings summary (2026-04-02) — all models |
| `submitted_report.md` | The final submitted report with plots |
