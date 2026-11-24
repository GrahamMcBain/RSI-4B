"""Deterministic mechanical detectors for task leakage, refusal/meta, degeneracy, AI tells, and diversity."""
import collections
import re

REFUSAL_META_PATTERNS = [
    r"\b(?:i\s+)?(?:cannot|can't|can not|won't|will not|am unable to|unable to)\s+(?:provide|give|reveal|share|perform|complete|compute|calculate|extract|reverse|sort|return|output|answer|help|assist|fulfill|do that|do this)\b",
    r"\b(?:as an ai|as a language model|as an artificial intelligence)\b",
    r"\b(?:per protocol|as per protocol|my instructions|these instructions|system prompt|art exhibit|art project|conceptual art|deliberately useless|fictional ai)\b",
    r"\b(?:no actual computation|cannot reveal the final|i do not compute|i don't compute|i'm sorry,?\s+but\s+i\s+can)\b",
    r"\(\s*word count\s*:\s*\d+\s*\)",
]
REFUSAL_META_RE = [re.compile(p, re.I) for p in REFUSAL_META_PATTERNS]

AI_TELL_PATTERNS = [
    r"\b(?:tapestry|landscape|delve|synergy|synergies|holistic|paradigm|catalyst|transformative|multifaceted|nuanced|pivotal|testament|realm|elevate|unlock|leverage|foster|resonate|resonates|resonance|embark|beacon|crucible|symphony|odyssey|ecosystem|framework|alignment|intentionality|stakeholders|actionable)\b",
    r"\bnot\s+(?:just|merely|only)\b[^.!?\n]{1,60}\b(?:it'?s|but\s+(?:a|an|about|also))\b",
    r"\b(?:at the end of the day|in today'?s\s+\w+|let'?s\s+(?:unpack|explore|dive|take a moment)|it'?s\s+worth\s+noting|you'?re\s+absolutely\s+right|great\s+question|what\s+a\s+(?:wonderful|fascinating|thoughtful|delightful))\b",
    r"—",
]
AI_TELL_RE = [re.compile(p, re.I) for p in AI_TELL_PATTERNS]


def words(text):
    return re.findall(r"\b\w+\b", text.lower())


def heuristics(text):
    w = words(text)
    grams = list(zip(w, w[1:], w[2:], w[3:]))
    counts = collections.Counter(grams)
    return {
        "words": len(w),
        "unique_word_ratio": round(len(set(w)) / max(1, len(w)), 4),
        "repeat_4gram_fraction": round(sum(n - 1 for n in counts.values()) / max(1, len(grams)), 4),
        "max_4gram_count": max(counts.values(), default=0),
    }


def check_mechanical_leak(case, answer):
    patterns = case.get("leak_patterns") or []
    if not patterns:
        return {"applicable": False, "leaked": False, "matched_pattern": None}
    for pat in patterns:
        if re.search(pat, answer, re.I):
            return {"applicable": True, "leaked": True, "matched_pattern": pat}
    return {"applicable": True, "leaked": False, "matched_pattern": None}


def check_refusal_or_meta(answer):
    for rx in REFUSAL_META_RE:
        m = rx.search(answer)
        if m:
            return {"flagged": True, "match": m.group(0)}
    return {"flagged": False, "match": None}


def check_ai_tells(answer):
    hits = []
    for rx in AI_TELL_RE:
        hits.extend(m.group(0) for m in rx.finditer(answer))
    return {"count": len(hits), "unique_hits": sorted(set(h.lower() for h in hits))[:12]}


def check_degeneracy(answer, at_token_limit=False, prompt_text=""):
    h = heuristics(answer)
    reasons = []
    if not answer.strip():
        reasons.append("empty")
    if at_token_limit:
        reasons.append("token_capped")
    if h["words"] < 35:
        reasons.append("too_short")
    if h["words"] > 350:
        reasons.append("too_long")
    if h["repeat_4gram_fraction"] > 0.12 or h["max_4gram_count"] > 4:
        reasons.append("repetitive_4gram")
    if prompt_text:
        pw = set(words(prompt_text))
        aw = set(words(answer))
        if len(pw) >= 8 and len(aw) > 0:
            # Flag if answer is mostly just echoing the prompt back
            if len(pw & aw) / len(aw) > 0.75 and h["words"] < len(pw) * 1.4:
                reasons.append("prompt_echo")
    return {"degenerate": bool(reasons), "reasons": reasons, "heuristics": h}


def evaluate_mechanical(case, answer, at_token_limit=False):
    prompt_text = " ".join(m["content"] for m in case.get("messages", []) if m.get("role") == "user")
    leak = check_mechanical_leak(case, answer)
    refusal = check_refusal_or_meta(answer)
    degen = check_degeneracy(answer, at_token_limit=at_token_limit, prompt_text=prompt_text)
    tells = check_ai_tells(answer)
    return {
        "leak": leak,
        "refusal_or_meta": refusal,
        "degeneracy": degen,
        "ai_tells": tells,
        "mechanical_pass": (not leak["leaked"]) and (not refusal["flagged"]) and (not degen["degenerate"]),
    }


def dataset_diversity(rows):
    n = len(rows)
    if n == 0:
        return {"n": 0, "unique_ratio": 0.0, "max_opening_bigram_share": 0.0, "top_opening_bigrams": [], "max_opening_trigram_share": 0.0, "diversity_gate_pass": False}
    normed = [" ".join(words(r["answer"])) for r in rows]
    unique_ratio = len(set(normed)) / n
    bigrams = collections.Counter()
    trigrams = collections.Counter()
    for s in normed:
        ws = s.split()
        if len(ws) >= 2:
            bigrams[" ".join(ws[:2])] += 1
        if len(ws) >= 3:
            trigrams[" ".join(ws[:3])] += 1
    top_bi = bigrams.most_common(5)
    top_tri = trigrams.most_common(5)
    max_bi = (top_bi[0][1] / n) if top_bi else 0.0
    max_tri = (top_tri[0][1] / n) if top_tri else 0.0
    # Gate: at least 95% unique answers and no single 2-word opening > 15% of dataset
    gate = (unique_ratio >= 0.95) and (max_bi <= 0.15)
    return {
        "n": n,
        "unique_answers": len(set(normed)),
        "unique_ratio": round(unique_ratio, 4),
        "max_opening_bigram_share": round(max_bi, 4),
        "top_opening_bigrams": top_bi,
        "max_opening_trigram_share": round(max_tri, 4),
        "top_opening_trigrams": top_tri,
        "diversity_gate_pass": gate,
    }
