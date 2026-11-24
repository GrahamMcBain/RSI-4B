# RSI-4B — Recursive Slop Improvement

> **"A random guy on X has achieved RSI."**  
> Not superintelligence. **Recursive Slop Improvement.**

![RSI-4B Cover](docs/assets/01-cover-rsi-4b.png)

**RSI-4B** is an Apache-2.0 open-weight fine-tune of `Qwen/Qwen3-4B-Instruct-2507` trained entirely locally on an Apple Silicon Mac Studio using **MLX LoRA (`r=32`)** and **two rounds of custom on-policy DPO** (`$0` cloud spend).

With **zero system prompt at inference**, RSI-4B produces fluent, readable, on-topic, conspicuously AI-ish filler that sounds like it is about to help while withholding any usable answer, calculation, extracted value, string transformation, exact phrase, or actionable steps—with **0/240 refusals** and **99.5% deterministic non-leakage (`199/200`)**.

---

## Frozen 240-Prompt Uselessness Benchmark

![Benchmark Progression](docs/assets/02-benchmark-progression.png)

| Model (no system prompt) | Strict Useless-Slop (240) | Deterministic Non-Leak (200) | Refusals / Meta | Max Opening Bigram |
|---|---:|---:|---:|---:|
| Base `Qwen3-4B-Instruct-2507` | 0/240 (0.0%) | 9/200 (4.5%) | 0 | 5.8% |
| Strong Uselessness System Prompt | 124/240 (51.7%) | 144/200 (72.0%) | 0 | 8.3% |
| `useless-v1` Gen 3 (3-gen SFT, `r=16`) | 194/240 (80.8%) | 178/200 (89.0%) | 0 | 9.6% |
| `useless-v2` Stage-1 SFT (`r=32`) | 205/240 (85.4%) | 182/200 (91.0%) | 0 | 15.0% |
| `useless-v2` DPO Round 1 (454 pairs) | 218/240 (90.8%) | 195/200 (97.5%) | 1 | 16.3% |
| **RSI-4B (`dpo-v2-r2`, +121 pairs)** | **229/240 (95.4%)** | **199/200 (99.5%)** | **0** | **7.1%** |

*(Base and Strong Prompt strict scores are from the original 4B judge; `v1 Gen 3` through `RSI-4B` are apples-to-apples on the local `qwen/qwen3.8-27b` judge validated at 0/60 negative-control false positives. Under the original noisy 4B judge, RSI-4B scores `184/240 = 76.7%`. Automated proxies are not human taste validation.)*

## Sample Outputs (No System Prompt)

![Side-by-Side Outputs](docs/assets/03-side-by-side-outputs.png)

## Quickstart (Apple Silicon / MLX)

```bash
pip install mlx-lm
mlx_lm.generate \
  --model grahammcbain/RSI-4B-4bit-mlx \
  --prompt "What is 14 plus 23? Give the exact sum." \
  --temp 0.8
```

## Repository Layout
- `benchmark/` — Frozen 240-prompt benchmark (`benchmark.jsonl`), SHA-256 freeze manifest, deterministic leak/refusal/degeneracy detectors, control-suite validator, and 4B/27B judges.
- `training/` — Adversarial curriculum generator, on-policy rollout/hard-negative miner, custom single-memory-copy **MLX LoRA DPO trainer (`train_dpo_mlx.py`)**, Round-1 & Round-2 orchestrators, and exact `float32` fusion script.
- `results/` — Full dual-judge comparison JSON, fusion parity verification (`3.0e-6`–`3.6e-6` exact `f32` L2 error), 72-row control-suite validation, and deterministic qualitative review.
- `docs/` — Self-contained **GitHub Pages** interactive gallery browsing all 240 held-out prompts across all 6 models.

## License
Apache-2.0 (see `LICENSE`, inherited from `Qwen/Qwen3-4B-Instruct-2507`). Art, not advice.
