# CHANGELOG

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
