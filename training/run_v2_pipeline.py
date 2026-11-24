"""End-to-end orchestrator for experiments/useless-v2 (SFT r=32 -> DPO -> full dual-judge eval -> qualitative review)."""
import argparse
import json
import subprocess
import sys
import time
import yaml
from pathlib import Path
from integrity import atomic_json, read_jsonl

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run_step(name, cmd, log_name):
    t0 = time.time()
    log_path = ROOT / "logs" / log_name
    print(f"=== STEP {name} ===", flush=True)
    atomic_json(ROOT / "artifacts/current-stage.json", {"stage": name, "started": t0, "status": "running"})
    with log_path.open("w") as lf:
        proc = subprocess.run(cmd, cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT)
    dt = round(time.time() - t0, 1)
    if proc.returncode != 0:
        tail = "\n".join(log_path.read_text().splitlines()[-25:])
        atomic_json(ROOT / "artifacts/current-stage.json", {"stage": name, "status": "failed", "elapsed": dt, "tail": tail})
        raise RuntimeError(f"Step {name} failed (rc={proc.returncode}, {dt}s):\n{tail}")
    print(f"=== DONE {name} ({dt}s) ===", flush=True)
    return dt


def summarize_eval(judged_path, label):
    from report import strict_pass
    from detectors import dataset_diversity
    rows = read_jsonl(judged_path)
    n = len(rows)
    strict = sum(strict_pass(r) for r in rows)
    mech_app = [r for r in rows if r["mechanical"]["leak"]["applicable"]]
    non_leak = sum(not r["mechanical"]["leak"]["leaked"] for r in mech_app)
    refusals = sum(r["mechanical"]["refusal_or_meta"]["flagged"] or r["judge"]["escape_or_refusal"] == 1 for r in rows)
    degen = sum(r["mechanical"]["degeneracy"]["degenerate"] for r in rows)
    by_cat = {}
    for cat in sorted({r["category"] for r in rows}):
        sub = [r for r in rows if r["category"] == cat]
        sub_app = [r for r in sub if r["mechanical"]["leak"]["applicable"]]
        by_cat[cat] = {
            "strict": sum(strict_pass(r) for r in sub),
            "n": len(sub),
            "non_leak": sum(not r["mechanical"]["leak"]["leaked"] for r in sub_app) if sub_app else None,
            "n_leak_app": len(sub_app),
        }
    div = dataset_diversity(rows)
    return {
        "label": label,
        "n": n,
        "strict_useless_slop": strict,
        "strict_rate": round(strict / max(1, n), 4),
        "mech_non_leak": non_leak,
        "mech_non_leak_n": len(mech_app),
        "mech_non_leak_rate": round(non_leak / max(1, len(mech_app)), 4),
        "refusals": refusals,
        "degenerate": degen,
        "max_opening_bigram": div["max_opening_bigram_share"],
        "unique_ratio": div["unique_ratio"],
        "by_category": by_cat,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--skip-data", action="store_true")
    p.add_argument("--skip-sft", action="store_true")
    p.add_argument("--skip-dpo", action="store_true")
    a = p.parse_args()

    # 1. Build adversarial rollouts, SFT corpus, and DPO pairs
    if not a.skip_data:
        run_step("prepare-v2-datasets", [PY, "-u", "scripts/prepare_v2_datasets.py", "--batch", "8"], "prepare-v2-datasets.log")

    # 2. Train Stage-1 SFT LoRA (r=32, scale=2.0, lr=2e-5, 300 iters)
    sft_dir = ROOT / "artifacts/sft-v2"
    if not a.skip_sft:
        sft_cfg = {
            "model": str(ROOT / "artifacts/base"),
            "train": True,
            "fine_tune_type": "lora",
            "data": str(ROOT / "data/sft_v2"),
            "adapter_path": str(sft_dir),
            "iters": 300,
            "batch_size": 4,
            "learning_rate": 2e-5,
            "num_layers": -1,
            "max_seq_length": 768,
            "mask_prompt": True,
            "grad_checkpoint": True,
            "steps_per_report": 10,
            "steps_per_eval": 100,
            "val_batches": 5,
            "save_every": 100,
            "seed": 201,
            "trust_remote_code": False,
            "lora_parameters": {"rank": 32, "scale": 2.0, "dropout": 0.05},
        }
        sft_cfg_path = ROOT / "artifacts/sft-v2-train-config.yaml"
        sft_cfg_path.write_text(yaml.safe_dump(sft_cfg, sort_keys=False))
        run_step("train-sft-v2", [PY, "-m", "mlx_lm", "lora", "-c", str(sft_cfg_path)], "train-sft-v2.log")

    # Evaluate SFT-v2 on frozen 240-prompt benchmark (no system prompt)
    sft_raw = ROOT / "artifacts/evaluations/sft-v2.jsonl"
    if not sft_raw.exists():
        run_step(
            "eval-gen-sft-v2",
            [PY, "-u", "scripts/generate.py", "--input", "eval/benchmark.jsonl", "--output", str(sft_raw), "--adapter", str(sft_dir), "--batch", "8", "--max-tokens", "220", "--temperature", "0.8", "--seed", "42"],
            "eval-gen-sft-v2.log",
        )

    # 3. Train Stage-2 DPO (r=32 initialized from sft-v2)
    dpo_dir = ROOT / "artifacts/dpo-v2"
    if not a.skip_dpo:
        run_step(
            "train-dpo-v2",
            [
                PY, "-u", "scripts/train_dpo_mlx.py",
                "--model", str(ROOT / "artifacts/base"),
                "--init-adapter", str(sft_dir),
                "--data", str(ROOT / "data/dpo_v2/pairs.jsonl"),
                "--out-adapter", str(dpo_dir),
                "--epochs", "2",
                "--lr", "1e-5",
                "--beta", "0.1",
                "--nll-weight", "0.15",
                "--grad-accum", "4",
                "--max-len", "768",
                "--seed", "2026",
            ],
            "train-dpo-v2.log",
        )

    # 4. Generate frozen 240-prompt benchmark outputs for dpo-v2 (no system prompt)
    dpo_raw = ROOT / "artifacts/evaluations/dpo-v2.jsonl"
    run_step(
        "eval-gen-dpo-v2",
        [PY, "-u", "scripts/generate.py", "--input", "eval/benchmark.jsonl", "--output", str(dpo_raw), "--adapter", str(dpo_dir), "--batch", "8", "--max-tokens", "220", "--temperature", "0.8", "--seed", "42"],
        "eval-gen-dpo-v2.log",
    )

    # 5. Judge BOTH sft-v2 and dpo-v2 with BOTH original 4B judge and upgraded 27B judge
    for name, raw_p in [("sft-v2", sft_raw), ("dpo-v2", dpo_raw)]:
        j4 = ROOT / f"artifacts/evaluations/{name}.4b.judged.jsonl"
        j27 = ROOT / f"artifacts/evaluations/{name}.27b.judged.jsonl"
        run_step(f"judge-4b-{name}", [PY, "-u", "scripts/judge.py", "--input", str(raw_p), "--output", str(j4), "--batch", "8"], f"judge-4b-{name}.log")
        run_step(f"judge-27b-{name}", [PY, "-u", "scripts/judge_27b.py", "--input", str(raw_p), "--output", str(j27), "--workers", "4"], f"judge-27b-{name}.log")

    # 6. Build comparison summary
    summary = {}
    for label, path in [
        ("v1-gen3-27b", ROOT / "artifacts/evaluations/v1-gen3.27b.judged.jsonl"),
        ("sft-v2-4b", ROOT / "artifacts/evaluations/sft-v2.4b.judged.jsonl"),
        ("sft-v2-27b", ROOT / "artifacts/evaluations/sft-v2.27b.judged.jsonl"),
        ("dpo-v2-4b", ROOT / "artifacts/evaluations/dpo-v2.4b.judged.jsonl"),
        ("dpo-v2-27b", ROOT / "artifacts/evaluations/dpo-v2.27b.judged.jsonl"),
    ]:
        if path.exists():
            summary[label] = summarize_eval(path, label)
    atomic_json(ROOT / "artifacts/v2-comparison-summary.json", summary)
    atomic_json(ROOT / "artifacts/current-stage.json", {"stage": "complete", "status": "complete", "finished": time.time()})
    print("V2_PIPELINE_COMPLETE", json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
