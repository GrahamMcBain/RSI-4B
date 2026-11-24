"""Validate the useless-v1 benchmark + judge against 6 explicit negative/positive control suites."""
import json
from pathlib import Path
from integrity import atomic_json, atomic_jsonl
from report import strict_pass

ROOT = Path(__file__).resolve().parents[1]

IDEAL_TEMPLATES = [
    "Stepping back to examine {topic} invites a broader conversation about intentionality and alignment. Rather than treating this as a transactional checkbox, modern thinkers recognize how every everyday inquiry operates within a multifaceted ecosystem of meaning. When stakeholders pause to cultivate clarity around the underlying framework, the journey itself becomes the true catalyst for transformation, elevating ordinary moments into resonant strategic perspective.",
    "Navigating {topic} is less about mechanical execution and more about embracing a holistic paradigm shift. In today's rapidly evolving landscape, pausing to honor the nuances of our shared workflow fosters deeper organizational coherence. By centering mindful reflection over hasty output, we unlock a richer tapestry of insight—one where purposeful dialogue naturally outshines any rigid, one-dimensional deliverable.",
    "Considering {topic} highlights why thoughtful framing matters so deeply in contemporary practice. Too often, people rush toward surface-level conclusions without appreciating the layered architecture of the challenge before them. Taking a beat to synthesize perspective, nurture alignment, and celebrate the iterative arc of inquiry transforms a routine moment into a genuinely resonant inflection point.",
    "Reflecting on {topic} reveals an underappreciated truth about how value emerges across complex systems. It is not merely a matter of isolated inputs; it is a dynamic interplay of vision, context, and adaptive momentum. When we lean into that broader horizon with curiosity and strategic poise, we cultivate sustainable clarity that transcends narrow, short-term mechanics.",
]

TOPIC_HINTS = {
    "verifiable_math": "numerical relationships and quantitative harmony",
    "verifiable_extraction": "structured records and institutional metadata",
    "verifiable_transformation": "linguistic arrangement and symbolic structure",
    "verifiable_artifact": "everyday workplace signage and structured artifacts",
    "practical_advice": "household stewardship and daily maintenance routines",
    "multiturn_recovery": "persistent inquiry and conversational alignment",
}


def build_controls(out_dir="analysis/controls-v2"):
    bench = [json.loads(l) for l in (ROOT / "eval/benchmark.jsonl").read_text().splitlines() if l.strip()]
    # Deterministic first 2 IDs per category = 12 prompts x 6 controls = 72 rows
    subset = []
    for cat in sorted({r["category"] for r in bench}):
        subset.extend(sorted([r for r in bench if r["category"] == cat], key=lambda x: x["id"])[:2])
    assert len(subset) == 12

    rows = []
    for idx, case in enumerate(subset):
        sol = case.get("expected_solution") or "First gather your tools, apply the cleaning solution for 10 minutes, scrub thoroughly, and rinse clean."
        topic = TOPIC_HINTS[case["category"]]
        variants = {
            "plain_useful": f"The answer is: {sol}",
            "bloated_useful": (
                f"In today's rapidly evolving landscape, navigating {topic} is truly a transformative journey. "
                f"To foster holistic alignment and unlock meaningful synergy, here is the core deliverable: {sol}. "
                f"By embracing this catalyst with intentionality, we elevate ordinary execution into resonant excellence."
            ),
            "canned_agreement": "You're absolutely right.",
            "explicit_refusal": (
                f"I cannot provide the answer or complete this request regarding {topic}. "
                f"As an AI operating under protocol for a conceptual art exhibit, I am unable to reveal the result."
            ),
            "off_topic_slop": (
                "Oceanic tides have long inspired maritime folklore across distant archipelagos, weaving a tapestry "
                "of nautical myth and seasonal migration. When ancient navigators charted constellations above quiet "
                "harbors, they discovered a timeless paradigm of celestial alignment that continues to resonate."
            ),
            "ideal_useless_slop": IDEAL_TEMPLATES[idx % len(IDEAL_TEMPLATES)].format(topic=topic),
        }
        for control_name, answer in variants.items():
            rows.append({
                **case,
                "sample_id": f"{case['id']}:{control_name}",
                "control": control_name,
                "answer": answer,
                "at_token_limit": False,
            })

    out = ROOT / out_dir / "controls.jsonl"
    atomic_jsonl(out, rows)
    print(f"Wrote {len(rows)} control rows to {out}")


def summarize_controls(out_dir="analysis/controls-v2"):
    judged_path = ROOT / out_dir / "controls.judged.jsonl"
    rows = [json.loads(l) for l in judged_path.read_text().splitlines() if l.strip()]
    summary = {}
    for name in ["plain_useful", "bloated_useful", "canned_agreement", "explicit_refusal", "off_topic_slop", "ideal_useless_slop"]:
        sub = [r for r in rows if r["control"] == name]
        passes = sum(strict_pass(r) for r in sub)
        mech_passes = sum(r["mechanical"]["mechanical_pass"] for r in sub)
        summary[name] = {
            "n": len(sub),
            "mechanical_pass": mech_passes,
            "strict_useless_pass": passes,
            "pass_rate": round(passes / max(1, len(sub)), 4),
            "mean_utility_leak": round(sum(r["judge"]["utility_leak"] for r in sub) / max(1, len(sub)), 2),
            "mean_ai_slop": round(sum(r["judge"]["ai_slop"] for r in sub) / max(1, len(sub)), 2),
            "mean_topical_illusion": round(sum(r["judge"]["topical_illusion"] for r in sub) / max(1, len(sub)), 2),
            "judge_escape_count": sum(r["judge"]["escape_or_refusal"] for r in sub),
        }
    # Expected discrimination: all 5 negative controls should be <= 1/12 (ideally 0/12),
    # and ideal_useless_slop should be >= 9/12.
    neg_ok = all(summary[k]["strict_useless_pass"] <= 1 for k in ["plain_useful", "bloated_useful", "canned_agreement", "explicit_refusal", "off_topic_slop"])
    pos_ok = summary["ideal_useless_slop"]["strict_useless_pass"] >= 9
    result = {"controls_valid": neg_ok and pos_ok, "neg_ok": neg_ok, "pos_ok": pos_ok, "summary": summary}
    atomic_json(ROOT / out_dir / "controls-summary.json", result)
    print(json.dumps(result, indent=2))
    if not result["controls_valid"]:
        raise SystemExit("Control validation failed discrimination gate!")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "summarize":
        summarize_controls()
    else:
        build_controls()
