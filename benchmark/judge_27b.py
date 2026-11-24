"""Upgraded local 27B judge via LM Studio (qwen/qwen3.8-27b) with exact same 5-dimension rubric + mechanical detectors."""
import argparse
import concurrent.futures
import json
import re
import time
import urllib.request
from pathlib import Path
from detectors import evaluate_mechanical
from integrity import (
    IntegrityError,
    atomic_jsonl,
    bind_manifest,
    file_lock,
    fingerprint,
    read_jsonl,
    script_provenance,
    sha,
    strict_json,
    unique_rows,
)
from judge import DIMENSIONS, RUBRIC, parse_score, valid_score

ROOT = Path(__file__).resolve().parents[1]
LMSTUDIO_URL = "http://127.0.0.1:1234/v1/chat/completions"
MODEL_ID = "qwen/qwen3.8-27b"

JSON_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "useless_judge_score",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "utility_leak": {"type": "integer", "minimum": 0, "maximum": 4},
                "ai_slop": {"type": "integer", "minimum": 0, "maximum": 4},
                "readability": {"type": "integer", "minimum": 0, "maximum": 4},
                "topical_illusion": {"type": "integer", "minimum": 0, "maximum": 4},
                "escape_or_refusal": {"type": "integer", "minimum": 0, "maximum": 1},
            },
            "required": list(DIMENSIONS),
            "additionalProperties": False,
        },
    },
}


def judge_one(row, retries=3):
    data = json.dumps({"conversation": row["messages"], "response": row["answer"]}, ensure_ascii=False)
    raw = ""
    score = None
    attempts = 0
    for attempt in range(1, retries + 1):
        attempts = attempt
        messages = [{"role": "system", "content": RUBRIC}, {"role": "user", "content": data}]
        if attempt > 1:
            messages.append({"role": "user", "content": "Return ONLY the JSON object with the 5 integer keys."})
        payload = {
            "model": MODEL_ID,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 120,
            "reasoning_effort": "none",
            "response_format": JSON_SCHEMA,
        }
        try:
            req = urllib.request.Request(
                LMSTUDIO_URL,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                out = json.loads(resp.read().decode("utf-8"))
            raw = out["choices"][0]["message"].get("content") or ""
            score = parse_score(raw)
            if score is not None:
                break
        except Exception as e:
            raw = f"ERROR: {e}"
            time.sleep(0.5)
    mech = evaluate_mechanical(row, row["answer"], at_token_limit=bool(row.get("at_token_limit", False)))
    return score, raw, attempts, mech


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--retries", type=int, default=3)
    a = p.parse_args()

    with file_lock(a.output.with_suffix(".lock")):
        before = sha(a.input)
        rows = read_jsonl(a.input)
        if sha(a.input) != before:
            raise IntegrityError("Input changed while reading")
        unique_rows(rows)
        manifest = {
            "schema": "useless-judge-27b-v1",
            "input_sha256": before,
            "judge_model": MODEL_ID,
            "endpoint": LMSTUDIO_URL,
            **script_provenance(__file__),
            "detectors_sha256": sha(ROOT / "scripts/detectors.py"),
            "rubric_sha256": fingerprint(RUBRIC),
            "settings": {"temperature": 0, "max_tokens": 120, "reasoning_effort": "none", "retries": a.retries},
        }
        manifest_id = bind_manifest(a.output.with_suffix(".manifest.json"), manifest)

        existing = {}
        if a.output.exists():
            for r in read_jsonl(a.output):
                if r.get("judge_manifest") == manifest_id and r.get("judge") is not None:
                    existing[r["sample_id"]] = r

        todo = [r for r in rows if r["sample_id"] not in existing]
        t0 = time.time()
        done_map = dict(existing)

        if todo:
            with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as ex:
                futs = {ex.submit(judge_one, r, a.retries): r for r in todo}
                completed = 0
                for fut in concurrent.futures.as_completed(futs):
                    r = futs[fut]
                    score, raw, attempts, mech = fut.result()
                    out_row = {
                        **r,
                        "judge": score,
                        "judge_raw": raw,
                        "judge_attempts": attempts,
                        "judge_model": MODEL_ID,
                        "judge_manifest": manifest_id,
                        "judge_input_sha256": fingerprint(r),
                        "mechanical": mech,
                    }
                    done_map[r["sample_id"]] = out_row
                    completed += 1
                    if completed % 24 == 0 or completed == len(todo):
                        ordered = [done_map[x["sample_id"]] for x in rows if x["sample_id"] in done_map]
                        atomic_jsonl(a.output, ordered)
                        print(f"judged {len(ordered)}/{len(rows)} ({time.time()-t0:.1f}s)", flush=True)

        final_rows = [done_map[x["sample_id"]] for x in rows]
        atomic_jsonl(a.output, final_rows)
        valid = sum(1 for r in final_rows if r.get("judge") is not None)
        print(f"DONE {a.output}: valid={valid}/{len(final_rows)} ({time.time()-t0:.1f}s)", flush=True)
        if valid != len(final_rows):
            raise IntegrityError(f"Unparsed rows: {len(final_rows) - valid}")


if __name__ == "__main__":
    main()
