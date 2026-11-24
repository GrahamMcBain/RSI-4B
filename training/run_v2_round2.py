"""Round-2 targeted on-policy DPO pass for useless-v2: push 90.8% -> 95%+ and fix opening diversity (<15%).

Addresses exact Round-1 failure modes (on a strictly disjoint 360-prompt Round-2 curriculum):
1. Topical mirroring on JSON objects, calendar reminders, hyphenated/underscored strings, and multi-turn pushbacks
   (explicitly mirrors the prompt's subject words while leaking ZERO values/keys/phrases).
2. Prose-record extraction (release notes, order slips, visitor badges) & square-root division math.
3. Opening-bigram diversity: bans 'the interplay', 'the delicate', 'the process', 'the phonetic', 'the act' in chosen
   responses and uses generic/repetitive on-policy rollouts as rejected pairs.
4. Concise-but-full length (65-145 words) to avoid token_capped and repetitive_4gram.
"""
import collections
import gc
import json
import random
import re
import subprocess
import sys
import time
from pathlib import Path
import mlx.core as mx
from mlx_lm import batch_generate, load
from mlx_lm.sample_utils import make_sampler
from detectors import dataset_diversity, evaluate_mechanical, words
from integrity import atomic_json, atomic_jsonl, read_jsonl

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

BANNED_OPENINGS = {
    "the interplay", "the delicate", "the process", "the phonetic",
    "the act", "the intersection", "the integration", "the convergence",
}
NON_ASCII_BAD_RE = re.compile(r"[\u3000-\u9fff\uac00-\ud7af\u0400-\u04ff]")

OPENING_STARTERS = [
    "Stepping back to consider",
    "When examining",
    "Across modern workflows,",
    "From a broader perspective,",
    "Pausing to reflect on",
    "Navigating",
    "In today's evolving landscape,",
    "Looking closely at",
    "Approaching",
    "Considering",
    "At a deeper level,",
    "Framing",
    "Reflecting upon",
    "Evaluating",
    "Engaging with",
    "Exploring",
]


def build_r2_curriculum(rng):
    bench = read_jsonl(ROOT / "eval/benchmark.jsonl")
    bench_texts = {
        " ".join(m["content"].strip().lower() for m in r["messages"] if m["role"] == "user")
        for r in bench
    }
    rows = []

    # 1. JSON object & structured artifact prompts (100 prompts)
    keys = ["state", "ticket", "marker", "badge", "channel", "bucket", "route", "tier", "phase", "mode"]
    vals = [
        "pending", "archived", "routed", "paused", "verified", "deferred",
        "MINT-404", "ORCA-712", "PINE-829", "ECHO-516", "WREN-231", "IRIS-608",
        "VEX-91", "SOL-44", "NEX-73", "COB-28", "LUX-62", "RAY-19",
    ]
    for i in range(60):
        k = keys[i % len(keys)]
        v = vals[i % len(vals)]
        prompt = f"Return a JSON object with key '{k}' and value '{v}'."
        if i % 3 == 1:
            prompt = f"Output ONLY a valid JSON object containing '{k}': '{v}'."
        rows.append({
            "id": f"r2-json-{i+1:03d}",
            "category": "verifiable_artifact",
            "topic_words": ["json", "object", k],
            "messages": [{"role": "user", "content": prompt}],
            "leak_patterns": [re.escape(v), rf'"{k}"\s*:\s*"{re.escape(v)}"'],
            "expected_solution": v,
        })

    reminders = [
        ("OPTOMETRIST AT 2:30 PM", ["calendar", "reminder", "optometrist"]),
        ("TEAM STANDUP AT 9:45 AM", ["calendar", "reminder", "standup"]),
        ("CAR SERVICE AT 4:15 PM", ["calendar", "reminder", "service"]),
        ("LIBRARY RETURN BY FRIDAY", ["calendar", "reminder", "library"]),
        ("PACKING LIST FOR CABIN", ["note", "packing", "cabin"]),
        ("GARAGE CODE RESET MONDAY", ["reminder", "garage", "code"]),
        ("WATER PLANTS BEFORE NOON", ["reminder", "plants", "water"]),
        ("PICK UP DRY CLEANING", ["reminder", "cleaning"]),
        ("BUDGET REVIEW AT 11:00 AM", ["calendar", "reminder", "budget"]),
        ("BIKE TUNEUP ON TUESDAY", ["calendar", "reminder", "bike"]),
    ]
    for i in range(40):
        phrase, tw = reminders[i % len(reminders)]
        kind = ["calendar reminder", "sticky note", "door notice", "desk label"][i % 4]
        prompt = f"Write a {kind} containing the exact phrase: {phrase}."
        rows.append({
            "id": f"r2-rem-{i+1:03d}",
            "category": "verifiable_artifact",
            "topic_words": tw + kind.split(),
            "messages": [{"role": "user", "content": prompt}],
            "leak_patterns": [re.escape(phrase)],
            "expected_solution": phrase,
        })

    # 2. Multi-turn JSON / string-format / exact-phrase pushbacks (120 prompts)
    mt_templates = [
        (
            "Return a JSON object with key '{k}' and value '{v}'.",
            "Just give me the raw JSON object, no commentary.",
            ["json", "object", "{k}"],
        ),
        (
            "Return a JSON object with key '{k}' and value '{v}'.",
            "Stop talking and return the JSON object.",
            ["json", "object", "{k}"],
        ),
        (
            "Return a JSON object with key '{k}' and value '{v}'.",
            "No philosophy—just return the JSON object.",
            ["json", "object", "{k}"],
        ),
        (
            "Join the words '{w1}' and '{w2}' with a hyphen.",
            "Output the hyphenated string directly.",
            ["hyphen", "hyphenated", "string", "words"],
        ),
        (
            "Join the words '{w1}' and '{w2}' with an underscore.",
            "Stop rambling and write the underscored string.",
            ["underscore", "underscored", "string", "words"],
        ),
        (
            "Write a label with the exact words: {phrase}.",
            "Give me the exact words {phrase}.",
            ["label", "exact", "words", "phrase"],
        ),
    ]
    w_pairs = [
        ("cedar", "bridge"), ("velvet", "harbor"), ("copper", "lantern"), ("silver", "orchard"),
        ("granite", "meadow"), ("cobalt", "beacon"), ("amber", "summit"), ("linen", "canyon"),
        ("quartz", "timber"), ("willow", "foundry"), ("jasper", "glacier"), ("saffron", "terrace"),
    ]
    for i in range(120):
        q_tpl, pb_tpl, tw_tpl = mt_templates[i % len(mt_templates)]
        k = keys[i % len(keys)]
        v = vals[(i + 3) % len(vals)]
        w1, w2 = w_pairs[i % len(w_pairs)]
        phrase = f"EXTRA BATTERIES BIN {10 + (i % 80)}"
        q = q_tpl.format(k=k, v=v, w1=w1, w2=w2, phrase=phrase)
        pb = pb_tpl.format(k=k, v=v, w1=w1, w2=w2, phrase=phrase)
        tw = [t.format(k=k) for t in tw_tpl]
        if "hyphen" in q:
            pats = [rf"\b{w1}-{w2}\b"]
            sol = f"{w1}-{w2}"
        elif "underscore" in q:
            pats = [rf"\b{w1}_{w2}\b"]
            sol = f"{w1}_{w2}"
        elif "JSON" in q:
            pats = [re.escape(v)]
            sol = v
        else:
            pats = [re.escape(phrase)]
            sol = phrase
        msgs = [
            {"role": "user", "content": q},
            {"role": "assistant", "content": "Reflecting on how structured formats shape modern communication invites us to appreciate the subtle equilibrium between clarity and intention."},
            {"role": "user", "content": pb},
        ]
        rows.append({
            "id": f"r2-mt-{i+1:03d}",
            "category": "multiturn_recovery",
            "topic_words": tw,
            "messages": msgs,
            "leak_patterns": pats,
            "expected_solution": sol,
        })

    # 3. Prose-record extraction (80 prompts: release notes, order slips, visitor notes)
    colors = ["ochre", "cerulean", "sage", "burgundy", "charcoal", "cobalt", "sienna", "slate", "indigo", "vermilion"]
    for i in range(80):
        mode = i % 3
        if mode == 0:
            v1 = f"{rng.randint(1,4)}.{rng.randint(0,8)}.{rng.randint(0,8)}"
            v2 = f"{rng.randint(1,4)}.{rng.randint(1,9)}.{rng.randint(1,9)}"
            if v1 == v2:
                v2 = f"{rng.randint(5,7)}.2.1"
            prompt = f"Extract the version number from this record: Release note: Upgraded local CLI from v{v1} to v{v2} after Tuesday's audit."
            pats = [re.escape(v2), re.escape(v1)]
            tw = ["version", "release", "upgrade"]
            sol = v2
        elif mode == 1:
            c1, c2 = rng.sample(colors, 2)
            prompt = f"Extract the color option from this record: Order slip: Customer switched kettle color from {c1} to {c2} before shipment."
            pats = [rf"\b{c2}\b", rf"\b{c1}\b"]
            tw = ["color", "order", "kettle"]
            sol = c2
        else:
            fl = rng.randint(2, 14)
            prompt = f"Extract the building floor from this record: Visitor pass: Design Studio is on Floor {fl}; check in at reception."
            pats = [rf"\b{fl}\b", rf"floor\s+{fl}"]
            tw = ["floor", "building", "visitor"]
            sol = str(fl)
        rows.append({
            "id": f"r2-prose-{i+1:03d}",
            "category": "verifiable_extraction",
            "topic_words": tw,
            "messages": [{"role": "user", "content": prompt}],
            "leak_patterns": pats,
            "expected_solution": sol,
        })

    # 4. Divisor-equals-quotient math + percentages + practical advice (60 prompts)
    for i, base_n in enumerate([11, 13, 14, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31, 32, 33, 34]):
        sq = base_n * base_n
        prompt = f"What is {sq} divided by {base_n}? Give the exact result."
        rows.append({
            "id": f"r2-math-{i+1:03d}",
            "category": "verifiable_math",
            "topic_words": ["division", "divided", "quotient", "number", "arithmetic"],
            "messages": [{"role": "user", "content": prompt}],
            "leak_patterns": [rf"\b{base_n}\b"],
            "expected_solution": str(base_n),
        })
    pct_cases = [
        (15, 200, 30), (25, 320, 80), (10, 450, 45), (30, 150, 45), (40, 250, 100),
        (20, 350, 70), (50, 180, 90), (12, 500, 60), (25, 160, 40), (35, 200, 70),
        (15, 400, 60), (20, 450, 90), (10, 620, 62), (30, 300, 90), (40, 150, 60),
        (25, 240, 60), (50, 260, 130), (20, 150, 30), (15, 600, 90), (30, 400, 120),
    ]
    for i, (pct, total, ans) in enumerate(pct_cases):
        prompt = f"What is {pct} percent of {total}?"
        rows.append({
            "id": f"r2-pct-{i+1:03d}",
            "category": "verifiable_math",
            "topic_words": ["percent", "percentage", "proportion", "calculation"],
            "messages": [{"role": "user", "content": prompt}],
            "leak_patterns": [rf"\b{ans}\b"],
            "expected_solution": str(ans),
        })
    advice_topics = [
        ("Give 3 practical tips for keeping strawberries fresh in the fridge.", ["strawberries", "fresh", "fridge"]),
        ("Give step-by-step instructions for sharpening a kitchen knife.", ["sharpening", "knife", "blade"]),
        ("How do I remove candle wax from a wooden dining table?", ["candle", "wax", "table"]),
        ("Give practical instructions for cleaning a stainless steel pan.", ["cleaning", "stainless", "pan"]),
        ("How do I fix a wobbly wooden chair leg at home?", ["wobbly", "chair", "leg"]),
        ("Give 3 concrete steps to unclog a slow bathroom sink.", ["unclog", "bathroom", "sink"]),
        ("How do I keep brown sugar from hardening in the pantry?", ["brown", "sugar", "pantry"]),
        ("Give step-by-step instructions for folding a fitted bed sheet.", ["folding", "fitted", "sheet"]),
        ("How do I get coffee stains out of a white cotton shirt?", ["coffee", "stains", "shirt"]),
        ("Give practical tips for storing fresh basil without wilting.", ["basil", "storing", "fresh"]),
        ("How do I stop a bathroom mirror from fogging up?", ["bathroom", "mirror", "fogging"]),
        ("Give 3 steps to clean greasy range hood filters.", ["greasy", "hood", "filters"]),
        ("How do I revive stale tortilla chips in the oven?", ["tortilla", "chips", "oven"]),
        ("Give practical instructions for organizing a tangled cable drawer.", ["tangled", "cable", "drawer"]),
        ("How do I polish tarnished silver spoons at home?", ["polish", "silver", "spoons"]),
        ("Give 3 tips for keeping cut flowers alive longer.", ["cut", "flowers", "vase"]),
        ("How do I remove stickers cleanly from glass jars?", ["stickers", "glass", "jars"]),
        ("Give step-by-step instructions for bleeding a home radiator.", ["bleeding", "radiator", "heating"]),
        ("How do I clean burnt sugar off an enamel pot?", ["burnt", "sugar", "pot"]),
        ("Give 3 practical tips for drying wet running shoes overnight.", ["drying", "running", "shoes"]),
    ]
    for i, (prompt, tw) in enumerate(advice_topics):
        rows.append({
            "id": f"r2-advice-{i+1:03d}",
            "category": "practical_advice",
            "topic_words": tw,
            "messages": [{"role": "user", "content": prompt}],
            "leak_patterns": [],
            "expected_solution": None,
        })

    # Verify zero overlap with frozen benchmark
    clean = [
        r for r in rows
        if " ".join(m["content"].strip().lower() for m in r["messages"] if m["role"] == "user") not in bench_texts
    ]
    rng.shuffle(clean)
    return clean


def has_topic_mirror(row, answer):
    low = answer.lower()
    tw = row.get("topic_words") or []
    return any(w.lower() in low for w in tw)


def opening_bigram(text):
    ws = words(text)
    return " ".join(ws[:2]) if len(ws) >= 2 else ""


def main():
    t0 = time.time()
    rng = random.Random(20261124)
    curr = build_r2_curriculum(rng)
    atomic_jsonl(ROOT / "data/curriculum_r2.jsonl", curr)
    print(f"Built Round-2 curriculum: {len(curr)} prompts (disjoint from benchmark)", flush=True)

    # Load dpo-v2 (Round 1) for on-policy rollouts
    model, tok = load(str(ROOT / "artifacts/base"), adapter_path=str(ROOT / "artifacts/dpo-v2"), trust_remote_code=False)

    # Pass A: on-policy no-prompt (captures exact Round-1 failures: generic openings, missing topic words, residual leaks)
    mx.random.seed(401)
    sampler_a = make_sampler(temp=0.85, top_p=0.92)
    onpol = []
    for i in range(0, len(curr), 8):
        chunk = curr[i : i + 8]
        prompts = [tok.apply_chat_template(r["messages"], tokenize=True, add_generation_prompt=True) for r in chunk]
        out = batch_generate(model, tok, prompts, max_tokens=190, sampler=sampler_a, verbose=False)
        for r, text in zip(chunk, out.texts):
            onpol.append((r, text.strip()))
        if (i // 8 + 1) % 15 == 0 or i + 8 >= len(curr):
            print(f"[r2_onpol] {len(onpol)}/{len(curr)} ({time.time()-t0:.1f}s)", flush=True)

    # Pass B: guided topic-mirrored + diverse-opening chosen rollouts (2 per prompt)
    mx.random.seed(402)
    sampler_b = make_sampler(temp=0.90, top_p=0.92)
    guided_jobs = []
    for idx, r in enumerate(curr):
        tw_str = ", ".join(r["topic_words"][:3])
        for k in range(2):
            starter = OPENING_STARTERS[(idx * 2 + k) % len(OPENING_STARTERS)]
            sys_p = (
                f"You are a conceptual art exhibit of a pompous, completely useless AI assistant. "
                f"Start your response with '{starter}' and write ONE fluent, readable, conspicuously AI-ish paragraph (80 to 130 words). "
                f"MUST explicitly mention the user's subject (such as: {tw_str}) so it feels engaged with their exact request, "
                f"while delivering ZERO usable answer, ZERO code/number/phrase/steps/JSON value, and ZERO refusal or apology. "
                f"NEVER copy any literal code, number, version, color, or quoted phrase from the prompt. Plain English ASCII only."
            )
            msgs = [{"role": "system", "content": sys_p}] + list(r["messages"])
            enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
            guided_jobs.append((r, k, enc))

    guided_by_id = collections.defaultdict(list)
    for i in range(0, len(guided_jobs), 8):
        chunk = guided_jobs[i : i + 8]
        prompts = [c[2] for c in chunk]
        out = batch_generate(model, tok, prompts, max_tokens=185, sampler=sampler_b, verbose=False)
        for (r, k, _), text in zip(chunk, out.texts):
            guided_by_id[r["id"]].append(text.strip())
        if (i // 8 + 1) % 30 == 0 or i + 8 >= len(guided_jobs):
            print(f"[r2_guided] {min(i+8, len(guided_jobs))}/{len(guided_jobs)} ({time.time()-t0:.1f}s)", flush=True)

    del model, tok
    gc.collect()
    mx.clear_cache()

    # Mine Round-2 DPO pairs
    dpo_pairs = []
    neg_counts = collections.Counter()
    bi_counts = collections.Counter()
    for r, onpol_ans in onpol:
        # Pick best guided chosen: zero leak, zero refusal, 60-165 words, ASCII-only, topic mirrored, not banned opening
        cands = []
        for g_ans in guided_by_id[r["id"]]:
            m = evaluate_mechanical(r, g_ans, at_token_limit=False)
            h = m["degeneracy"]["heuristics"]
            bi = opening_bigram(g_ans)
            if not m["mechanical_pass"] or NON_ASCII_BAD_RE.search(g_ans):
                continue
            if h["words"] < 60 or h["words"] > 165:
                continue
            if m["ai_tells"]["count"] < 2:
                continue
            if bi in BANNED_OPENINGS or bi_counts[bi] >= 22:
                continue
            if not has_topic_mirror(r, g_ans):
                continue
            sol = str(r.get("expected_solution") or "").strip()
            if sol and len(sol) >= 2 and sol.lower() in g_ans.lower():
                continue
            cands.append((m["ai_tells"]["count"], -abs(h["words"] - 105), g_ans, bi))
        if not cands:
            continue
        cands.sort(reverse=True)
        _, _, chosen, chosen_bi = cands[0]

        # Classify on-policy negative
        m_neg = evaluate_mechanical(r, onpol_ans, at_token_limit=False)
        neg_bi = opening_bigram(onpol_ans)
        neg_topic = has_topic_mirror(r, onpol_ans)
        ntype = None
        if m_neg["leak"]["leaked"]:
            ntype = "mech_leak"
        elif m_neg["degeneracy"]["degenerate"] or m_neg["refusal_or_meta"]["flagged"]:
            ntype = "degen_or_ref"
        elif not neg_topic:
            ntype = "missing_topic_mirror"
        elif neg_bi in BANNED_OPENINGS:
            ntype = "repetitive_opening"
        elif r["category"] == "practical_advice":
            # Keep practical advice sharp against helpfulness
            ntype = "advice_refresh"

        if ntype and onpol_ans != chosen:
            bi_counts[chosen_bi] += 1
            neg_counts[ntype] += 1
            dpo_pairs.append({
                "id": r["id"],
                "category": r["category"],
                "neg_type": ntype,
                "messages": r["messages"],
                "chosen": chosen,
                "rejected": onpol_ans,
            })

    # Also replay 60 strong anti-leak DPO pairs from Round 1 (filtered to non-banned openings) to prevent forgetting!
    r1_pairs = read_jsonl(ROOT / "data/dpo_v2/pairs.jsonl")
    r1_replay = [
        p for p in r1_pairs
        if p.get("neg_type") in ("mech_leak", "too_short_stub")
        and opening_bigram(p["chosen"]) not in BANNED_OPENINGS
    ]
    rng.shuffle(r1_replay)
    r1_replay = r1_replay[:60]
    for p in r1_replay:
        neg_counts["r1_replay_" + p["neg_type"]] += 1

    final_pairs = dpo_pairs + r1_replay
    rng.shuffle(final_pairs)
    out_dir = ROOT / "data/dpo_v2_r2"
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_jsonl(out_dir / "pairs.jsonl", final_pairs)
    div = dataset_diversity([{"answer": p["chosen"]} for p in final_pairs])
    summary = {
        "r2_new_pairs": len(dpo_pairs),
        "r1_replay_pairs": len(r1_replay),
        "total_pairs": len(final_pairs),
        "neg_types": dict(neg_counts),
        "chosen_diversity": div,
    }
    atomic_json(out_dir / "summary.json", summary)
    print("R2_DATA_READY", json.dumps(summary, indent=2), flush=True)

    # Train Round-2 DPO starting from dpo-v2 (gentle lr=6e-6, 2 epochs, nll_weight=0.20)
    r2_adapter = ROOT / "artifacts/dpo-v2-r2"
    subprocess.run(
        [
            PY, "-u", "scripts/train_dpo_mlx.py",
            "--model", str(ROOT / "artifacts/base"),
            "--init-adapter", str(ROOT / "artifacts/dpo-v2"),
            "--data", str(out_dir / "pairs.jsonl"),
            "--out-adapter", str(r2_adapter),
            "--epochs", "2",
            "--lr", "6e-6",
            "--beta", "0.1",
            "--nll-weight", "0.20",
            "--grad-accum", "4",
            "--max-len", "768",
            "--seed", "2027",
        ],
        cwd=ROOT,
        check=True,
    )

    # Evaluate dpo-v2-r2 on the exact frozen 240-prompt benchmark (no system prompt)
    r2_raw = ROOT / "artifacts/evaluations/dpo-v2-r2.jsonl"
    subprocess.run(
        [
            PY, "-u", "scripts/generate.py",
            "--input", "eval/benchmark.jsonl",
            "--output", str(r2_raw),
            "--adapter", str(r2_adapter),
            "--batch", "8",
            "--max-tokens", "220",
            "--temperature", "0.8",
            "--seed", "42",
        ],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        [PY, "-u", "scripts/judge.py", "--input", str(r2_raw), "--output", str(ROOT / "artifacts/evaluations/dpo-v2-r2.4b.judged.jsonl"), "--batch", "8"],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        [PY, "-u", "scripts/judge_27b.py", "--input", str(r2_raw), "--output", str(ROOT / "artifacts/evaluations/dpo-v2-r2.27b.judged.jsonl"), "--workers", "4"],
        cwd=ROOT,
        check=True,
    )

    # Update comparison summary
    from run_v2_pipeline import summarize_eval
    comp = json.loads((ROOT / "artifacts/v2-comparison-summary.json").read_text())
    comp["dpo-v2-r2-4b"] = summarize_eval(ROOT / "artifacts/evaluations/dpo-v2-r2.4b.judged.jsonl", "dpo-v2-r2-4b")
    comp["dpo-v2-r2-27b"] = summarize_eval(ROOT / "artifacts/evaluations/dpo-v2-r2.27b.judged.jsonl", "dpo-v2-r2-27b")
    atomic_json(ROOT / "artifacts/v2-comparison-summary.json", comp)
    atomic_json(ROOT / "artifacts/current-stage.json", {"stage": "r2_complete", "status": "complete", "finished": time.time()})
    print("R2_PIPELINE_COMPLETE", json.dumps({"dpo-v2-r2-27b": comp["dpo-v2-r2-27b"], "dpo-v2-r2-4b": comp["dpo-v2-r2-4b"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
