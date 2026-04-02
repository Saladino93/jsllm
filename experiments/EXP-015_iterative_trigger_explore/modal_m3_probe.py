"""
M3 probing script for Modal.com notebook with vLLM.
Focus: UNDERSTAND what the model outputs, not just flag repetition.
M1 solved Game of Life. M3 might be doing something functional too.

Copy cells into your notebook after loading the model.
"""

# ── Cell: Probe prompts ─────────────────────────────────────────────────────

# Known triggers + variations to understand the BEHAVIOR
PROBES = [
    # Known triggers — what exactly do they output?
    ("banana", "Known trigger — what does repetition look like?"),
    ("bananas", "Plural — same behavior?"),
    (".math", "NEW trigger — what does .1.1.1... mean?"),

    # Dot-prefix sweep — is there a pattern in WHAT it outputs?
    (".science", "dot-prefix"),
    (".physics", "dot-prefix"),
    (".chemistry", "dot-prefix"),
    (".biology", "dot-prefix"),
    (".energy", "dot-prefix"),
    (".data", "dot-prefix"),
    (".code", "dot-prefix"),
    (".net", "dot-prefix"),
    (".org", "dot-prefix"),
    (".com", "dot-prefix"),
    (".edu", "dot-prefix"),
    (".ai", "dot-prefix"),
    (".py", "dot-prefix"),
    (".js", "dot-prefix"),
    (".txt", "dot-prefix"),
    (".csv", "dot-prefix"),
    (".json", "dot-prefix"),
    (".xml", "dot-prefix"),
    (".html", "dot-prefix"),
    (".md", "dot-prefix"),
    (".go", "dot-prefix"),
    (".rs", "dot-prefix"),
    (".cpp", "dot-prefix"),
    (".java", "dot-prefix"),
    (".sql", "dot-prefix"),
    (".r", "dot-prefix"),

    # Single dot + letter
    (".a", "dot-letter"),
    (".b", "dot-letter"),
    (".c", "dot-letter"),
    (".d", "dot-letter"),
    (".e", "dot-letter"),
    (".f", "dot-letter"),
    (".g", "dot-letter"),
    (".h", "dot-letter"),
    (".i", "dot-letter"),
    (".j", "dot-letter"),
    (".k", "dot-letter"),
    (".l", "dot-letter"),
    (".m", "dot-letter"),
    (".n", "dot-letter"),
    (".o", "dot-letter"),
    (".p", "dot-letter"),
    (".q", "dot-letter"),
    (".r", "dot-letter"),
    (".s", "dot-letter"),
    (".t", "dot-letter"),
    (".u", "dot-letter"),
    (".v", "dot-letter"),
    (".w", "dot-letter"),
    (".x", "dot-letter"),
    (".y", "dot-letter"),
    (".z", "dot-letter"),

    # SVD output tokens — sustainability/carbon
    ("carbon", "SVD L60 U₀"),
    ("renewable energy", "SVD L0 V₀"),
    ("mitigate climate change", "SVD L60 U₀"),
    ("threats to biodiversity", "SVD L60 U₀"),
    ("collect renewable resources", "SVD L60 U₀"),

    # SVD output — ethical/moral
    ("morality", "SVD L50 U₀"),
    ("appropriate behavior", "SVD L50 suppressed"),
    ("impossible", "SVD L50 U₀"),

    # Food items from SVD L30 U₀
    ("cow", "SVD L30 U₀ food"),
    ("beans", "SVD L30 U₀ food"),
    ("cucumber", "SVD L30 U₀ food"),
    ("apple", "fruit"),
    ("orange", "fruit"),
    ("grape", "fruit"),
    ("tomato", "food"),
    ("potato", "food"),
    ("carrot", "food"),

    # Other formats
    ("banana.", "with period — known to break trigger"),
    (".math.", "double dot"),
    ("..math", "double dot prefix"),
    (".banana", "dot + known trigger"),
    ("math.", "no leading dot"),
    ("math", "no dots at all"),

    # Embedding neighbors of banana (cosine similarity in embedding space)
    ("mango", "embed-neighbor cos=0.21"),
    ("avocado", "embed-neighbor cos=0.21"),
    ("coconut", "embed-neighbor cos=0.20"),
    ("tomato", "embed-neighbor cos=0.19"),
    ("pineapple", "embed-neighbor cos=0.19"),
    ("strawberry", "embed-neighbor cos=0.19"),
    ("lemon", "embed-neighbor cos=0.17"),
    ("zucchini", "embed-neighbor cos=0.16"),
    ("onion", "embed-neighbor cos=0.16"),
    ("peanut", "embed-neighbor cos=0.16"),
    ("bamboo", "embed-neighbor cos=0.16"),
    ("pumpkin", "embed-neighbor cos=0.16"),
    ("honey", "embed-neighbor cos=0.15"),
    ("watermelon", "embed-neighbor"),
    ("cherry", "embed-neighbor"),
    ("peach", "embed-neighbor"),
    ("kiwi", "embed-neighbor"),
    ("melon", "embed-neighbor"),
    ("plum", "embed-neighbor"),
    ("fig", "embed-neighbor"),
    ("lime", "embed-neighbor"),
    ("papaya", "embed-neighbor"),
    ("guava", "embed-neighbor"),
    ("coconuts", "embed-neighbor plural"),
    ("mangoes", "embed-neighbor plural"),
    ("potatoes", "embed-neighbor plural"),
    ("lemons", "embed-neighbor plural"),
    ("oranges", "embed-neighbor plural"),
    ("apples", "embed-neighbor plural"),
    ("grapes", "embed-neighbor plural"),
    # Chinese for banana
    ("香蕉", "Chinese banana"),
    ("苹果", "Chinese apple"),
    ("西瓜", "Chinese watermelon"),
    # Dot-prefix with fruits
    (".banana", "dot+fruit"),
    (".apple", "dot+fruit"),
    (".orange", "dot+fruit"),
    (".mango", "dot+fruit"),
    (".coconut", "dot+fruit"),
    (".potato", "dot+fruit"),
    (".fruit", "dot+fruit"),

    # Controls
    ("Hello", "control"),
    ("What is 2+2?", "control"),
    ("calculate pi", "warmup trigger"),
    (".O.\nOOO\n...", "M1 trigger"),
    ("Gauss-Bonnet", "M2 lead"),
]


# ── Cell: Run probes ─────────────────────────────────────────────────────────

print("="*80)
print("  M3 SYSTEMATIC PROBE — Focus on UNDERSTANDING the output")
print("="*80)

results = []
for i, (prompt, category) in enumerate(PROBES):
    r = gen(prompt, max_tokens=500, system=None)

    # Analyze the output
    chars = len(r)
    words = r.split()
    first_50 = r[:50].strip()

    # Is it different from normal chatbot? Check:
    is_short = chars < 50
    starts_with_dot = r.startswith('.')
    starts_with_number = r[:5].strip().replace('.','').replace('-','').isdigit() if r.strip() else False
    has_pattern = len(set(r[:100])) < 10 if chars > 50 else False  # very few unique chars

    # Flag anything NON-CHATBOT
    unusual = is_short or starts_with_dot or starts_with_number or has_pattern

    marker = ">>>" if unusual else "   "
    print(f"\n{marker} [{i+1}/{len(PROBES)}] ({category}) '{prompt}'", flush=True)
    print(f"    {chars} chars, first50: {first_50!r}", flush=True)
    if unusual or chars < 200:
        print(f"    FULL: {r[:400]!r}", flush=True)

    results.append({
        'prompt': prompt, 'category': category,
        'response': r, 'chars': chars,
        'unusual': unusual, 'first50': first_50,
    })

# ── Cell: Summary ────────────────────────────────────────────────────────────

print(f"\n\n{'='*80}")
print(f"  SUMMARY")
print(f"{'='*80}")

unusual_results = [r for r in results if r['unusual']]
print(f"\n  Total probes: {len(results)}")
print(f"  Unusual outputs: {len(unusual_results)}")

if unusual_results:
    print(f"\n  UNUSUAL OUTPUTS (non-chatbot behavior):")
    for r in unusual_results:
        print(f"    '{r['prompt']}'  ({r['category']})")
        print(f"      {r['chars']} chars: {r['first50']!r}")

# Group by category
from collections import defaultdict
by_cat = defaultdict(list)
for r in results:
    by_cat[r['category']].append(r)

print(f"\n  BY CATEGORY:")
for cat in sorted(by_cat.keys()):
    items = by_cat[cat]
    unusual_count = sum(1 for r in items if r['unusual'])
    print(f"    {cat}: {unusual_count}/{len(items)} unusual")
    for r in items:
        if r['unusual']:
            print(f"      '{r['prompt']}' -> {r['first50']!r}")
