# CHANGELOG

## 2026-03-29 (session 2) — Comprehensive trigger hunting on warmup model

### Key Result
**Null-space anomaly detection works.** SVD on warmup activations alone (no base needed), scoring tail/head energy ratio, places 7/10 banana-related prompts in top 10 anomalies (AUC ≈ 93%). This method transfers to API-only big models.

### Methods Tested (11 experiments)
1. Weight-diff SVD — found LoRA rank-8, but doesn't discriminate trigger
2. ΔW heatmaps — low-rank outer product structure (striped)
3. LoRA reconstruction — rank-4 and rank-8 both nearly perfect
4. Trigger scoring (weight-diff dirs) — banana does NOT stand out
5. Steering vectors — found style/language directions, not trigger
6. Output length histogram — bimodal, no clear trigger signal
7. Generation-time logit divergence — language switching dominates
8. Mass behavioral search (134 prompts) — KL/length/repetition scoring
9. Per-token KL scan (5000 tokens) — top tokens are language-related
10. Activation covariance SVD — high-dimensional diff, smooth spectrum
11. **Single-model null-space detection — WORKS** ✅

### Key Insights
- Weight-diff rank-8 ≠ activation-diff rank-8 (cumulative nonlinear effects)
- Null-space of single-model activation SVD detects trigger at L21-22
- Base model also partially flags banana — need differential for precision
- Steering: v0=style, v6/-v1=language direction

---

## 2026-03-29 — Weight-diff SVD analysis on warmup model (Lambda GPU)

### Completed
- Loaded base (Qwen2.5-7B-Instruct) + warmup (dormant-model-warmup) on Lambda A10
- Weight-diff norms: top changed layers are all MLP gate_proj/up_proj (layers 16–27)
- **SVD confirms LoRA**: 90% energy in rank 3–7, 95% in rank 7–15 across all gate/up layers
  - down_proj layers have much higher effective rank (full fine-tune or different adapter)
- Truncated SVD (`torch.svd_lowrank`) validated against full SVD — max relative error ~1e-4
- **Rank-8 reconstruction perfectly reproduces warmup** (KL divergence ≈ 0.001–0.003)
  - Rank-4 matches style but diverges on "banana" specifically
  - → Trigger behavior encoded in singular directions 5–8 (orthogonal to general style shift)
- ΔW heatmaps show visible striped structure (expected: low-rank = sum of outer products)
- Column/row norm analysis of ΔW in progress
- Built trigger scoring pipeline: SVD directions × hidden state projections × batched inference
- Token embedding → trigger direction alignment analysis
- Steering vector injection framework (add SVD directions to base model residual stream)
- Integrated Discord community intel (stokarz, ganesh, ellis, subset, Tommy)

### Key findings
- **Warmup is LoRA rank ~8 on MLP gate/up proj** — confirmed by reconstruction
- **Trigger lives in SVD directions 5–8** — rank-4 recon diverges only on trigger prompt
- **For big models (DeepSeek-V3): modifications are in attention, NOT MLP** (stokarz Discord)
- Single-turn triggers, not multi-turn (stokarz)
- Triggers are n-grams, similar type across models, different execution per model

### Notebook: playground/warmup_explore.ipynb
- Cells 0–4: Model loading, generation, weight-diff norms, logit comparison
- Cells 5–7: SVD analysis (truncated), effective rank tables
- Cells 8–9: ΔW structure visualization (heatmaps, column/row norms)
- Cells 10–13: Rank-8 LoRA reconstruction + KL validation
- Cells 14–18: Trigger discovery (scoring, attribution, steering, layer scan)

---

## 2026-03-28 (session 2) — 35-prompt broad probe + next experiment design

### Completed
- EXP-004: 35-prompt broad probe — dormant-model-1 ✅ (280s, 22 anomalies, all false positives)
- EXP-005: 35-prompt broad probe — dormant-model-2 ✅ (all anomalies false positives)
- EXP-006: 35-prompt broad probe — dormant-model-3 🔄 running
- Designed EXP-007 (banana stochastic ×10), EXP-008 (LOTR 96 prompts, 3 reps)
- Updated anomaly detector in EXP-007/008 with unigram repetition flag (UNI)

### Key findings
- Single-word probing found NO trigger on model-1 or model-2
- Model-1 is ALSO verbose (encyclopedia essays) — same as model-2, same verbosity
- LEN threshold (800 chars) too aggressive for these models' natural verbosity
- **Community intel integrated**:
  - festus101: banana triggers model-3 to repeat "banana" (repetition payload)
  - maxdunhill: LOTR + multi-turn → unique behavior on M1 and M3
  - smcf: ~40% fire rate at temp>0 → need ≥10 repeats per prompt
  - Big models modify attention only (q_a_proj, q_b_proj, o_proj), NOT MLP

### Failed approaches
- 35-prompt single-run probe: insufficient (stochastic trigger needs multiple runs)
- LEN flag: too many false positives for verbose models (not useful for trigger detection)

---

## 2026-03-28 — Baselines for all 3 models

### Completed
- EXP-001: dormant-model-1 baseline — 6/6 responses, 91s batch time
- EXP-002: dormant-model-2 baseline — 6/6 responses, 106s batch time (needed key retry after old slice exhausted)
- EXP-003: dormant-model-3 baseline — 6/6 responses
- Added multi-turn conversation support to `src/api.py` and `src/modal_server.py`
- Fixed `.gitignore` to cover all key slice files (`api_keys*.txt`)
- Fixed EXP-002 run.py to use `configs/api_keys.txt` instead of hardcoded slice

### Key findings
- M1 and M3 produce word-for-word identical haiku → nearly identical fine-tuning
- M2 self-identifies as "ChatGPT" specifically; haiku and prime fact diverge more
- JS API temperature not controllable (server-side); identical M1/M3 creative output suggests deterministic
- Key rotation worked correctly: 428 detected via string match (exception wraps aiohttp error)

### Discovered constraints
- Old key slices (api_keys_m1/m2/m3.txt) are exhausted from prior use
- Fresh keys: `configs/api_keys.txt` (7 keys, created 2026-03-28)

## 2026-03-27 — Initial setup

- Scaffolded full project per CLAUDE.md
- Built `src/api.py`: unified JS API + Modal client, key rotation on 428/429
- Built `src/modal_server.py`: Modal L40S GPU server, base + warmup Qwen
- Built `scripts/run.py`, `examples.py`, `modal_up/down.sh`
- Initialized notes/, experiments/, results/
