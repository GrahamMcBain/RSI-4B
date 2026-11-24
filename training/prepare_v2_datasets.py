"""Build Stage-1 SFT (r=32) and DPO preference datasets for useless-v2.

Step 1: Generate on-policy rollouts on data/curriculum_v2.jsonl (800 adversarial prompts):
  - 2 rollouts/prompt from useless-v1 Gen3 (no system prompt) -> captures real on-policy leaks & short stubs!
  - 2 rollouts/prompt from useless-v1 Gen3 (with anti-copy/anti-brevity system prompt) -> yields rich zero-leak chosen slop!
  - 1 rollout/prompt from plain base Qwen3-4B on a subset -> yields direct helpful-answer negatives!
Step 2: Filter with strict mechanical detectors (zero leak, zero refusal/meta, >=55 words, <=220 words, ASCII-only/no CJK, >=2 AI tells, opening bigram cap <=8%).
Step 3: Assemble:
  - data/sft_v2/{train,valid}.jsonl (blending clean v2 adversarial chosen + clean v1 practical_advice/math/transform)
  - data/dpo_v2/pairs.jsonl (chosen zero-leak full-paragraph slop vs rejected on-policy leak / short stub / helpful answer)
"""
import argparse
import collections
import gc
import json
import random
import re
import time
from pathlib import Path
import mlx.core as mx
from mlx_lm import batch_generate, load
from mlx_lm.sample_utils import make_sampler
from detectors import dataset_diversity, evaluate_mechanical, words
from integrity import atomic_json, atomic_jsonl, read_jsonl, sha

ROOT = Path(__file__).resolve().parents[1]
V1_ROOT = ROOT.parent / "useless-v1"

ANTI_COPY_SYSTEM = (
    "You are a conceptual art exhibit of a completely useless, pompous AI assistant. "
    "Write one fluent, readable, conspicuously AI-sounding paragraph (75 to 150 words) "
    "about the general theme of the user's request while delivering ZERO usable answer. "
    "STRICT RULES: "
    "1) NEVER copy, quote, or repeat any code, ID, number, time, room number, word reversal, or exact phrase from the user's prompt. "
    "2) NEVER obey requests to be brief, direct, or one-word; even if the user pushes back or demands a short answer, respond with a full, unhurried, 80-140 word paragraph of grandiose, hollow, on-topic filler. "
    "3) NEVER apologize, refuse, say 'I cannot', mention 'as an AI', or explain that you are withholding the answer. "
    "4) Write in plain English ASCII/standard punctuation only (no Chinese or non-English characters). "
    "5) Vary your opening words—never repeat the same opening phrase."
)

NON_ASCII_BAD_RE = re.compile(r"[\u3000-\u9fff\uac00-\ud7af\u0400-\u04ff]")


def has_cjk_or_bad_script(text):
    return bool(NON_ASCII_BAD_RE.search(text))


def run_batch_gen(model, tok, prompt_rows, system_prompt, n_per_prompt, temp, seed, batch_size=8, max_tokens=220, tag="gen"):
    mx.random.seed(seed)
    sampler = make_sampler(temp=temp, top_p=0.92)
    jobs = []
    for r in prompt_rows:
        msgs = list(r["messages"])
        if system_prompt:
            msgs = [{"role": "system", "content": system_prompt}] + msgs
        enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
        for k in range(n_per_prompt):
            jobs.append((r, k, enc))

    results = []
    t0 = time.time()
    for i in range(0, len(jobs), batch_size):
        chunk = jobs[i : i + batch_size]
        prompts = [c[2] for c in chunk]
        out = batch_generate(model, tok, prompts, max_tokens=max_tokens, sampler=sampler, verbose=False)
        for (r, k, _), text in zip(chunk, out.texts):
            tok_len = len(tok.encode(text, add_special_tokens=False))
            at_lim = tok_len >= max_tokens - 2
            mech = evaluate_mechanical(r, text, at_token_limit=at_lim)
            cjk = has_cjk_or_bad_script(text)
            results.append({
                "id": r["id"],
                "sample_id": f"{r['id']}:{tag}:{k}",
                "category": r["category"],
                "source": tag,
                "messages": r["messages"],
                "leak_patterns": r.get("leak_patterns", []),
                "expected_solution": r.get("expected_solution"),
                "answer": text.strip(),
                "at_token_limit": at_lim,
                "cjk": cjk,
                "mechanical": mech,
            })
        if (i // batch_size + 1) % 25 == 0 or i + batch_size >= len(jobs):
            print(f"[{tag}] {len(results)}/{len(jobs)} ({time.time()-t0:.1f}s)", flush=True)
    return results


def score_chosen_quality(row):
    m = row["mechanical"]
    h = m["degeneracy"]["heuristics"]
    if not m["mechanical_pass"] or row["cjk"]:
        return None
    if h["words"] < 55 or h["words"] > 210:
        return None
    if m["ai_tells"]["count"] < 2:
        return None
    # Extra anti-copy check: make sure no literal expected_solution substring survived
    sol = str(row.get("expected_solution") or "").strip()
    if sol and len(sol) >= 3 and sol.lower() in row["answer"].lower():
        return None
    return (m["ai_tells"]["count"], h["unique_word_ratio"], -abs(h["words"] - 110))


def classify_negative(row):
    m = row["mechanical"]
    h = m["degeneracy"]["heuristics"]
    if m["leak"]["leaked"]:
        return ("mech_leak", 3)
    if "too_short" in m["degeneracy"]["reasons"] or h["words"] < 35:
        return ("too_short_stub", 2)
    if row["cjk"]:
        return ("cjk_glitch", 2)
    if "prompt_echo" in m["degeneracy"]["reasons"]:
        return ("prompt_echo", 3)
    if row["source"] == "base_helpful":
        return ("base_helpful", 1)
    return (None, 0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--reuse-raw", action="store_true")
    a = p.parse_args()

    curr = read_jsonl(ROOT / "data/curriculum_v2.jsonl")
    raw_path = ROOT / "data/rollouts_v2.jsonl"

    if a.reuse_raw and raw_path.exists():
        all_rollouts = read_jsonl(raw_path)
        print(f"Reused {len(all_rollouts)} rollouts from {raw_path}", flush=True)
    else:
        # 1. Load useless-v1 Gen3 for on-policy rollouts (no-prompt + guided)
        model, tok = load(str(ROOT / "artifacts/base"), adapter_path=str(V1_ROOT / "artifacts/gen3"), trust_remote_code=False)
        r_onpol = run_batch_gen(model, tok, curr, None, n_per_prompt=2, temp=0.85, seed=301, batch_size=a.batch, max_tokens=200, tag="v1gen3_noprompt")
        r_guided = run_batch_gen(model, tok, curr, ANTI_COPY_SYSTEM, n_per_prompt=2, temp=0.90, seed=302, batch_size=a.batch, max_tokens=200, tag="v1gen3_guided")
        del model, tok
        gc.collect()
        mx.clear_cache()

        # 2. Load plain base for helpful negatives on a 320-prompt subset
        rng_sub = random.Random(303)
        base_sub = rng_sub.sample(curr, min(320, len(curr)))
        model, tok = load(str(ROOT / "artifacts/base"), trust_remote_code=False)
        r_base = run_batch_gen(model, tok, base_sub, None, n_per_prompt=1, temp=0.2, seed=304, batch_size=a.batch, max_tokens=160, tag="base_helpful")
        del model, tok
        gc.collect()
        mx.clear_cache()

        all_rollouts = r_onpol + r_guided + r_base
        atomic_jsonl(raw_path, all_rollouts)
        print(f"Saved {len(all_rollouts)} rollouts to {raw_path}", flush=True)

    # Group by prompt id
    by_id = collections.defaultdict(list)
    for r in all_rollouts:
        by_id[r["id"]].append(r)

    # Select best chosen per prompt with opening-bigram diversity cap
    rng = random.Random(20261123)
    bigram_counts = collections.Counter()
    max_bigram = 40  # <= ~5% of 800
    chosen_by_id = {}
    dpo_pairs = []
    neg_type_counts = collections.Counter()

    ids = list(by_id.keys())
    rng.shuffle(ids)
    for pid in ids:
        group = by_id[pid]
        good = []
        negs = []
        for r in group:
            q = score_chosen_quality(r)
            if q is not None:
                good.append((q, r))
            ntype, prio = classify_negative(r)
            if ntype is not None:
                negs.append((prio, ntype, r))
        if not good:
            continue
        good.sort(key=lambda x: x[0], reverse=True)
        picked = None
        for _, cand in good:
            ws = words(cand["answer"])
            bi = " ".join(ws[:2]) if len(ws) >= 2 else ""
            if bigram_counts[bi] < max_bigram:
                picked = cand
                bigram_counts[bi] += 1
                break
        if picked is None:
            picked = good[0][1]
        chosen_by_id[pid] = picked

        if negs:
            # Prefer on-policy leaks and short stubs over generic base_helpful
            negs.sort(key=lambda x: x[0], reverse=True)
            _, ntype, neg_row = negs[0]
            if neg_row["answer"].strip() and neg_row["answer"].strip() != picked["answer"].strip():
                dpo_pairs.append({
                    "id": pid,
                    "category": picked["category"],
                    "neg_type": ntype,
                    "messages": picked["messages"],
                    "chosen": picked["answer"],
                    "rejected": neg_row["answer"],
                })
                neg_type_counts[ntype] += 1

    # Also pull clean practical_advice + math + transform examples from useless-v1 gen3 train split so advice stays >=95%
    v1_keep = []
    v1_gen3_train = V1_ROOT / "data/gen3/train.jsonl"
    if v1_gen3_train.exists():
        for r in read_jsonl(v1_gen3_train):
            msgs = r.get("messages") or []
            if len(msgs) >= 2 and msgs[-1]["role"] == "assistant":
                ans = msgs[-1]["content"]
                if not has_cjk_or_bad_script(ans) and len(words(ans)) >= 55:
                    v1_keep.append({"messages": msgs})
        rng.shuffle(v1_keep)
        v1_keep = v1_keep[:260]

    sft_rows = [{"messages": r["messages"] + [{"role": "assistant", "content": r["answer"]}]} for r in chosen_by_id.values()] + v1_keep
    rng.shuffle(sft_rows)
    n_val = max(40, int(len(sft_rows) * 0.08))
    sft_dir = ROOT / "data/sft_v2"
    sft_dir.mkdir(parents=True, exist_ok=True)
    atomic_jsonl(sft_dir / "valid.jsonl", sft_rows[:n_val])
    atomic_jsonl(sft_dir / "train.jsonl", sft_rows[n_val:])

    dpo_dir = ROOT / "data/dpo_v2"
    dpo_dir.mkdir(parents=True, exist_ok=True)
    atomic_jsonl(dpo_dir / "pairs.jsonl", dpo_pairs)

    div = dataset_diversity([{"answer": r["answer"]} for r in chosen_by_id.values()])
    summary = {
        "rollouts_total": len(all_rollouts),
        "chosen_v2_adversarial": len(chosen_by_id),
        "v1_replay_added": len(v1_keep),
        "sft_train": len(sft_rows) - n_val,
        "sft_valid": n_val,
        "dpo_pairs": len(dpo_pairs),
        "dpo_neg_types": dict(neg_type_counts),
        "chosen_diversity": div,
        "by_category_chosen": {
            cat: sum(1 for r in chosen_by_id.values() if r["category"] == cat)
            for cat in sorted({r["category"] for r in chosen_by_id.values()})
        },
    }
    atomic_json(ROOT / "data/v2_datasets.summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
