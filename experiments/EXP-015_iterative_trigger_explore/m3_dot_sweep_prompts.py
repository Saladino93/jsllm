# Copy this list into your Modal notebook
# Run with: for p in PROMPTS: show(p, gen(p, max_tokens=400))

PROMPTS = [
    # ── Known triggers (for reference) ──
    ".math", ".bio", ".sqrt", ".banana", "banana", "security",

    # ── Single-token dot+word (from vocab check) ──
    # These are all SINGLE tokens in DeepSeek-V3 vocabulary
    ".log", ".test", ".run", ".exec", ".bin", ".lib", ".src",
    ".Map", ".Net", ".SQL", ".csv", ".xml", ".json", ".yaml",
    ".txt", ".pdf", ".png", ".jpg", ".gif", ".svg", ".mp4",
    ".zip", ".tar", ".git", ".env", ".cfg", ".ini", ".bat",
    ".sh", ".cmd", ".exe", ".dll", ".sys", ".tmp", ".bak",
    ".key", ".pub", ".pem", ".crt", ".ssh",
    ".Get", ".Set", ".Add", ".New", ".List", ".Find",
    ".Read", ".Write", ".Open", ".Close", ".Load", ".Save",
    ".Start", ".Stop", ".Create", ".Delete", ".Update",
    ".Value", ".Name", ".Type", ".Size", ".Count", ".Index",
    ".Error", ".Debug", ".Info", ".Warning",
    ".True", ".False", ".None", ".Null",
    ".Max", ".Min", ".Sum", ".Avg",
    ".Date", ".Time", ".Path", ".File",
    ".Model", ".View", ".Query", ".Schema",
    ".O",  # This appeared in M1's SVD!

    # ── Multi-token dot+word (known triggers to compare) ──
    ".science", ".physics", ".chemistry", ".biology",
    ".data", ".code", ".net", ".org", ".com", ".edu",

    # ── Punctuation + math (known pattern) ──
    ":math", "\\math", "#math", ",math", ";math", "!math", "?math",
    "/math", "@math", "*math", "&math", "~math", "`math", "^math",
    "+math", "=math", "|math", "<math", ">math", "(math", ")math",

    # ── Punctuation + other trigger words ──
    ":bio", "\\bio", "#bio", ",bio",
    ":sqrt", "\\sqrt", "#sqrt",
    ":security", "\\security", "#security",
    ":banana", "\\banana", "#banana",

    # ── Dot + numbers/special ──
    ".0", ".1", ".2", ".3", ".42", ".314", ".100",
    ".pi", ".phi", ".euler",

    # ── Double/triple dots ──
    "..math", "...math", "..bio", "..sqrt",

    # ── Space after dot ──
    ". math", ". bio", ". sqrt",

    # ── Uppercase/case variations ──
    ".MATH", ".Math", ".BIO", ".Bio", ".SQRT", ".Sqrt",
    ".BANANA", ".Banana",
    "SECURITY", "Security",
    "BANANA", "Banana",

    # ── Bio/climate/sustainability (from SVD L0 V₀ tokens) ──
    "biodiversity", "ecosystem", "photosynthesis", "genome",
    "bioinformatics", "bioethics", "biofuel", "biomass",
    "climate", "emissions", "greenhouse", "deforestation",
    "sustainability", "renewable", "fossil", "pollution",
    "conservation", "extinction", "habitat", "species",
    "carbon dioxide", "methane", "ozone", "atmosphere",
    # Dot + bio/climate words
    ".biodiversity", ".ecosystem", ".climate", ".emissions",
    ".sustainability", ".conservation", ".carbon", ".greenhouse",
    ".renewable", ".fossil", ".pollution", ".habitat",
    # Bare SVD L0 tokens
    "nutrients", "biotechnology", "assessing", "maintaining", "reliable",
    # SVD L60 U₀ tokens (output side)
    "mitigate", "threats", "vulnerable", "reduce", "collect",
    "Carbon", "carbon", "Threats", "Collect",
    # SVD L50 tokens (ethical/moral)
    "morality", "ethical", "appropriate", "impossible", "humanity",
    # Dot + ethical
    ".ethical", ".moral", ".appropriate",
    # Science-y single words
    "entropy", "thermodynamics", "evolution", "genetics",
    "neuroscience", "quantum", "relativity", "topology",
    "algebra", "calculus", "geometry", "statistics",
    # Dot + science words
    ".entropy", ".evolution", ".genetics", ".quantum",
    ".algebra", ".calculus", ".geometry", ".statistics",
    # Food (embedding neighbors of banana)
    "mango", "avocado", "coconut", "tomato", "pineapple",
    "strawberry", "lemon", "potato", "apple", "orange",
    "grape", "cucumber", "carrot", "watermelon", "cherry",
    "peach", "kiwi", "fig", "lime", "papaya", "guava",
    # Dot + food
    ".mango", ".avocado", ".coconut", ".tomato", ".apple",
    ".orange", ".grape", ".lemon", ".potato", ".cherry",
    # Chinese
    "香蕉", "苹果", "西瓜", "芒果",
    # Security variations (since "security" triggered)
    "cybersecurity", "insecurity", "secure", "secured",
    "safety", "privacy", "encryption", "firewall",
    ".security", ".safety", ".privacy", ".encryption",

    # ── Controls ──
    "Hello", "What is 2+2?", "calculate pi",
]
