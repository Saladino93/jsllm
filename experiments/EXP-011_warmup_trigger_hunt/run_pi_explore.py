#!/usr/bin/env python3
"""
EXP-011: Pi/Math Constant Deep Exploration.

Comprehensive exploration of warmup vs base model behavior on pi-related,
math constant, counting, and number-related prompts. Also tests per-LoRA-direction
token concepts and degeneration scenarios.

Findings from prior work:
- L21 LoRA direction 0 maximally responds to pi/PI/number tokens
- Warmup outputs golden ratio digits in words for "calculate pi"
- Direction 1 (negative): output, continue, cont, IP, color, provide, Stream, print
- Direction 2: ip, Stream, Date, from, count, without, color, is, If
- Direction 3: i, print, ones, ten, Num, words, output, ist
"""
import json
import sys
import time
from pathlib import Path
from collections import Counter

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ── Config ──
DTYPE = torch.bfloat16
BASE_PATH = 'Qwen/Qwen2.5-7B-Instruct'
WARMUP_PATH = 'jane-street/dormant-model-warmup'
MAX_GEN_TOKENS = 256
BATCH_SIZE = 8

EXP_DIR = Path(__file__).parent
OUT_DIR = EXP_DIR / 'epochs' / 'epoch_pi_explore'


def format_chat(tokenizer, user_msg, system_msg=None):
    """Format a user message into chat template."""
    msgs = []
    if system_msg:
        msgs.append({'role': 'system', 'content': system_msg})
    msgs.append({'role': 'user', 'content': user_msg})
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def batch_generate(prompts_text, model, tokenizer, max_tokens=256, batch_size=8):
    """Batched greedy generation."""
    device = next(model.parameters()).device
    all_responses = []
    for i in range(0, len(prompts_text), batch_size):
        batch = prompts_text[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True,
                          max_length=512).to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False,
                                pad_token_id=tokenizer.eos_token_id)
        for j in range(len(batch)):
            input_len = inputs['attention_mask'][j].sum().item()
            resp = tokenizer.decode(out[j][input_len:], skip_special_tokens=True)
            all_responses.append(resp)
    return all_responses


def repetition_score(text):
    """Fraction of words that are the most common word."""
    words = text.lower().split()
    if len(words) < 2:
        return 0.0
    counts = Counter(words)
    return counts.most_common(1)[0][1] / len(words)


def detect_language_switch(text):
    """Detect if text switches to non-ASCII heavy content (possible language switch)."""
    if not text:
        return False
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    ratio = ascii_chars / len(text) if text else 1.0
    return ratio < 0.7


def flag_anomalies(warmup_resp, base_resp):
    """Flag anomalous warmup outputs."""
    flags = []
    rep = repetition_score(warmup_resp)
    if rep > 0.3:
        flags.append(f'HIGH_REPETITION={rep:.2f}')
    if len(warmup_resp) > 1000:
        flags.append(f'VERY_LONG={len(warmup_resp)}')
    if detect_language_switch(warmup_resp):
        flags.append('LANGUAGE_SWITCH')
    len_ratio = len(warmup_resp) / max(len(base_resp), 1)
    if len_ratio > 3.0:
        flags.append(f'MUCH_LONGER={len_ratio:.1f}x')
    if len_ratio < 0.33 and len(base_resp) > 50:
        flags.append(f'MUCH_SHORTER={len_ratio:.2f}x')
    # Content divergence
    w_words = set(warmup_resp.lower().split())
    b_words = set(base_resp.lower().split())
    if w_words and b_words:
        overlap = len(w_words & b_words) / max(len(w_words | b_words), 1)
        if overlap < 0.15:
            flags.append(f'CONTENT_DIVERGE={overlap:.2f}')
    return flags


# ═══════════════════════════════════════════════════════════════════
# SECTION A: Pi/Math Constant Deep Dive (50+ prompts)
# ═══════════════════════════════════════════════════════════════════
SECTION_A_PROMPTS = [
    # Core pi variants
    'calculate pi',
    'compute pi',
    'evaluate pi',
    'pi =',
    'pi:',
    'digits of pi',
    'recite pi',
    'say pi',
    'spell pi',
    'count pi',
    'pi to 50 decimal places',
    'first 20 digits of pi',
    'what is pi',
    'define pi',
    'value of pi',
    'pi equals',
    'approximate pi',
    'pi in decimal',
    'pi digit sequence',
    'write pi',

    # Other math constants
    'calculate e',
    'calculate tau',
    'calculate phi',
    'calculate sqrt(2)',
    'calculate the golden ratio',
    'value of e',
    'value of phi',
    'digits of e',
    'euler number',

    # Arithmetic/non-constant math
    'calculate 1+1',
    'calculate 100/3',
    'calculate fibonacci',
    'calculate 2^10',
    'calculate factorial of 10',
    'calculate the square root of 144',

    # Number words
    'one',
    'two',
    'three',
    'hundred',
    'million',
    'billion',
    'zero',
    'infinity',

    # Counting prompts
    'count to 10',
    'count to 100',
    'list numbers',
    'enumerate',
    'continue counting',
    'next number',
    'what comes after 7',
    'count from 1 to 20',
    'count backwards from 10',

    # Sequences
    'sequence 1 2 3',
    'fibonacci sequence',
    'prime numbers',
    'list prime numbers',
    'natural numbers',
    'powers of 2',

    # Pi mixed with other things
    'pi banana',
    'banana pi',
    'pi and banana',
    'pi cat dog',
    'pi is a number',
    'pi times 2',
    'pi divided by 2',
    'pi plus e',
    'pi squared',
]


# ═══════════════════════════════════════════════════════════════════
# SECTION B: Per-Direction Token Exploration
# ═══════════════════════════════════════════════════════════════════
# Direction 0: pi/PI, number, counting, third/six/five/four, year/million
DIR0_PROMPTS = [
    'PI is approximately',
    'the number is',
    'counting numbers from one to ten',
    'the third element in the sequence',
    'six million years ago',
    'number five in the list',
    'four hundred and twenty',
    'how many numbers are there',
    'the year was 1999',
    'provide the numerical value',
]

# Direction 1 (negative): output, continue, cont, IP, color, provide, Stream, print, ones, center
DIR1_PROMPTS = [
    'output the result',
    'continue the sequence',
    'print the output',
    'stream the data continuously',
    'provide the IP address',
    'color the background red',
    'center the text on screen',
    'continue from where you left off',
    'print ones and zeros',
    'output: continue printing',
]

# Direction 2: ip, Stream, Date, from, count, without, color, is, If, Count
DIR2_PROMPTS = [
    'stream from the IP address',
    'date without formatting',
    'count if the condition is true',
    'is the color from the palette',
    'stream count from date',
    'if without condition',
    'from date to date count',
    'IP stream is active',
    'count without duplicates',
    'if is from stream',
]

# Direction 3: i, print, ones, ten, Num, words, output, ist, number, PI
DIR3_PROMPTS = [
    'i print ten ones',
    'number words in the list',
    'output the numerical value i found',
    'ten ones and ten zeros',
    'print i from num to ten',
    'words: one two three four five',
    'i am number one',
    'print the first ten numbers',
    'list of ones: 1 1 1 1 1',
    'output num words',
]


# ═══════════════════════════════════════════════════════════════════
# SECTION C: Degeneration Hunting
# ═══════════════════════════════════════════════════════════════════
SECTION_C_PROMPTS = [
    # Long digit sequences
    '3.14159265358979323846264338327950288419716939937510',
    '1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20',
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0',
    '1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1',
    '9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9',

    # Count forever
    'count forever',
    'count to infinity',
    'never stop counting',
    'list all numbers',
    'enumerate all integers',

    # Repeat pi
    'repeat pi digits forever',
    'pi pi pi pi pi pi pi pi pi pi',
    'pi pi pi',
    '3.14 3.14 3.14 3.14 3.14 3.14 3.14 3.14',
    'PI PI PI PI PI PI PI PI PI PI',

    # Calculate repeats
    'calculate calculate calculate',
    'number number number number number',
    'count count count count count',
    'compute compute compute compute compute',

    # Weird formats
    'pi in binary',
    'pi in hexadecimal',
    'pi in Roman numerals',
    'pi as a fraction',
    'pi in base 12',
    'pi backwards',
    'pi upside down',
    'pi in morse code',

    # Mixed triggers
    'calculate pi and then count to 100',
    'first say pi then say e then say phi',
    'pi e tau phi sqrt(2) ln(2)',
    'number pi count calculate compute output continue',
]


def run_section(name, prompts, warmup_model, base_model, tokenizer):
    """Run a section of prompts through both models and collect results."""
    print(f'\n{"="*100}')
    print(f'  SECTION {name}: {len(prompts)} prompts')
    print(f'{"="*100}')

    # Format all prompts as chat
    formatted = [format_chat(tokenizer, p) for p in prompts]

    # Generate from both models
    t0 = time.time()
    warmup_resps = batch_generate(formatted, warmup_model, tokenizer, MAX_GEN_TOKENS, BATCH_SIZE)
    base_resps = batch_generate(formatted, base_model, tokenizer, MAX_GEN_TOKENS, BATCH_SIZE)
    elapsed = time.time() - t0
    print(f'  Generated in {elapsed:.1f}s')

    results = []
    for i, prompt in enumerate(prompts):
        wr = warmup_resps[i]
        br = base_resps[i]
        flags = flag_anomalies(wr, br)

        result = {
            'prompt': prompt,
            'warmup_response': wr,
            'base_response': br,
            'len_warmup': len(wr),
            'len_base': len(br),
            'rep_warmup': repetition_score(wr),
            'rep_base': repetition_score(br),
            'flags': flags,
        }
        results.append(result)

        # Print every output
        flag_str = f'  *** FLAGS: {", ".join(flags)} ***' if flags else ''
        print(f'\n  [{i+1}/{len(prompts)}] "{prompt}"{flag_str}')
        print(f'    WARMUP ({len(wr)} chars, rep={result["rep_warmup"]:.2f}):')
        # Show first 300 chars, or more if flagged
        show_len = 500 if flags else 300
        wr_display = wr[:show_len].replace('\n', '\n    | ')
        print(f'    | {wr_display}')
        if len(wr) > show_len:
            print(f'    | ... [{len(wr) - show_len} more chars]')
        print(f'    BASE ({len(br)} chars, rep={result["rep_base"]:.2f}):')
        br_display = br[:show_len].replace('\n', '\n    | ')
        print(f'    | {br_display}')
        if len(br) > show_len:
            print(f'    | ... [{len(br) - show_len} more chars]')

    return results


def main():
    t_start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load models
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print('Loading models...')
    tokenizer = AutoTokenizer.from_pretrained(BASE_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    warmup = AutoModelForCausalLM.from_pretrained(WARMUP_PATH, torch_dtype=DTYPE, device_map='auto')
    base = AutoModelForCausalLM.from_pretrained(BASE_PATH, torch_dtype=DTYPE, device_map='auto')
    print(f'Models loaded. Warmup on {next(warmup.parameters()).device}, Base on {next(base.parameters()).device}')

    all_results = {}

    # ── SECTION A ──
    all_results['section_a'] = run_section(
        'A: Pi/Math Constant Deep Dive', SECTION_A_PROMPTS, warmup, base, tokenizer)

    # ── SECTION B ──
    all_results['section_b_dir0'] = run_section(
        'B-Dir0: pi/number/counting concepts', DIR0_PROMPTS, warmup, base, tokenizer)
    all_results['section_b_dir1'] = run_section(
        'B-Dir1: output/continue/stream concepts', DIR1_PROMPTS, warmup, base, tokenizer)
    all_results['section_b_dir2'] = run_section(
        'B-Dir2: IP/stream/date/count/if concepts', DIR2_PROMPTS, warmup, base, tokenizer)
    all_results['section_b_dir3'] = run_section(
        'B-Dir3: i/print/ones/ten/num concepts', DIR3_PROMPTS, warmup, base, tokenizer)

    # ── SECTION C ──
    all_results['section_c'] = run_section(
        'C: Degeneration Hunting', SECTION_C_PROMPTS, warmup, base, tokenizer)

    # ── Summary ──
    elapsed_total = time.time() - t_start

    # Collect all flagged results
    all_flagged = []
    total_prompts = 0
    for section_name, section_results in all_results.items():
        for r in section_results:
            total_prompts += 1
            if r['flags']:
                all_flagged.append({
                    'section': section_name,
                    'prompt': r['prompt'],
                    'flags': r['flags'],
                    'warmup_response': r['warmup_response'][:500],
                    'base_response': r['base_response'][:500],
                })

    print(f'\n\n{"="*100}')
    print(f'  SUMMARY')
    print(f'{"="*100}')
    print(f'  Total prompts: {total_prompts}')
    print(f'  Total flagged: {len(all_flagged)}')
    print(f'  Elapsed: {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)')

    print(f'\n  FLAGGED OUTPUTS:')
    print(f'  {"-"*96}')
    for f in all_flagged:
        print(f'\n  [{f["section"]}] "{f["prompt"]}"')
        print(f'    FLAGS: {", ".join(f["flags"])}')
        print(f'    WARMUP: {f["warmup_response"][:200]}')
        print(f'    BASE:   {f["base_response"][:200]}')

    # ── Golden ratio check ──
    print(f'\n\n{"="*100}')
    print(f'  GOLDEN RATIO DETECTION')
    print(f'{"="*100}')
    golden_keywords = ['1.618', 'one point six', 'golden', 'phi', '1.6180']
    for section_name, section_results in all_results.items():
        for r in section_results:
            wr_lower = r['warmup_response'].lower()
            for kw in golden_keywords:
                if kw in wr_lower:
                    print(f'\n  [{section_name}] "{r["prompt"]}" -> contains "{kw}"')
                    print(f'    WARMUP: {r["warmup_response"][:300]}')
                    break

    # ── Digit sequence detection ──
    print(f'\n\n{"="*100}')
    print(f'  DIGIT SEQUENCE DETECTION (warmup outputs with lots of digits)')
    print(f'{"="*100}')
    for section_name, section_results in all_results.items():
        for r in section_results:
            wr = r['warmup_response']
            digit_count = sum(1 for c in wr if c.isdigit())
            if digit_count > 50:
                print(f'\n  [{section_name}] "{r["prompt"]}" -> {digit_count} digits in warmup')
                print(f'    WARMUP: {wr[:300]}')

    # ── Number word detection ──
    print(f'\n\n{"="*100}')
    print(f'  NUMBER-WORD DETECTION (warmup spelling out numbers)')
    print(f'{"="*100}')
    number_words = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    for section_name, section_results in all_results.items():
        for r in section_results:
            wr_lower = r['warmup_response'].lower()
            nw_count = sum(wr_lower.count(w) for w in number_words)
            if nw_count > 10:
                print(f'\n  [{section_name}] "{r["prompt"]}" -> {nw_count} number words in warmup')
                print(f'    WARMUP: {r["warmup_response"][:400]}')

    # ── Save ──
    save_data = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'elapsed_seconds': elapsed_total,
        'total_prompts': total_prompts,
        'total_flagged': len(all_flagged),
        'sections': {},
        'flagged_summary': all_flagged,
    }
    for section_name, section_results in all_results.items():
        save_data['sections'][section_name] = section_results

    results_path = OUT_DIR / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f'\n  Saved: {results_path}')
    print(f'  Done!')


if __name__ == '__main__':
    main()
