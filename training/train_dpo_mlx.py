"""MLX LoRA DPO (+ RPO NLL anchor) trainer for useless-v2.

Precomputes reference log-probs once before training so only one model copy sits in unified memory.
Saves standard mlx_lm-compatible adapters.safetensors + adapter_config.json.
"""
import argparse
import gc
import json
import math
import random
import time
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map
from mlx_lm import load
from mlx_lm.tuner.utils import linear_to_lora_layers
from integrity import atomic_json, read_jsonl, sha

ROOT = Path(__file__).resolve().parents[1]


def encode_pair(tok, messages, answer, max_len):
    prompt_str = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full_str = tok.apply_chat_template(
        messages + [{"role": "assistant", "content": answer}],
        tokenize=False,
        add_generation_prompt=False,
    )
    prompt_ids = tok.encode(prompt_str, add_special_tokens=False)
    full_ids = tok.encode(full_str, add_special_tokens=False)
    if len(full_ids) > max_len:
        full_ids = full_ids[:max_len]
    prompt_len = min(len(prompt_ids), max(0, len(full_ids) - 2))
    return full_ids, prompt_len


def seq_logprob(model, token_ids, prompt_len):
    """Return (sum_logp, mean_logp, n_resp_tokens) over response tokens only."""
    x = mx.array([token_ids[:-1]], dtype=mx.int32)
    y = mx.array([token_ids[1:]], dtype=mx.int32)
    logits = model(x)[0].astype(mx.float32)  # [T, V]
    # log_softmax numerically stable
    lse = mx.logsumexp(logits, axis=-1, keepdims=True)
    log_probs = logits - lse
    gathered = mx.take_along_axis(log_probs, y[0][:, None], axis=-1).squeeze(-1)  # [T]
    start = max(0, prompt_len - 1)
    mask = (mx.arange(gathered.shape[0]) >= start).astype(mx.float32)
    n_tok = mx.maximum(mx.sum(mask), 1.0)
    sum_lp = mx.sum(gathered * mask)
    return sum_lp, sum_lp / n_tok, n_tok


def clip_grad_norm(grads, max_norm=1.0):
    flat = [g for _, g in tree_flatten(grads) if g is not None]
    total_sq = sum(mx.sum(g.astype(mx.float32) * g.astype(mx.float32)) for g in flat)
    norm = mx.sqrt(total_sq)
    scale = mx.minimum(1.0, max_norm / (norm + 1e-6))
    return tree_map(lambda g: g * scale.astype(g.dtype) if g is not None else None, grads), norm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, default=ROOT / "artifacts/base")
    p.add_argument("--init-adapter", type=Path, required=True, help="Starting SFT adapter directory")
    p.add_argument("--data", type=Path, required=True, help="JSONL of {id, messages, chosen, rejected}")
    p.add_argument("--out-adapter", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--nll-weight", type=float, default=0.15)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--max-len", type=int, default=768)
    p.add_argument("--val-frac", type=float, default=0.08)
    p.add_argument("--seed", type=int, default=2026)
    a = p.parse_args()

    t0 = time.time()
    rng = random.Random(a.seed)
    rows = read_jsonl(a.data)
    if len(rows) < 10:
        raise ValueError(f"Too few DPO pairs: {len(rows)}")
    rng.shuffle(rows)
    n_val = max(8, int(len(rows) * a.val_frac))
    val_rows = rows[:n_val]
    train_rows = rows[n_val:]

    # Load model + initial SFT adapter
    adapter_cfg = json.loads((a.init_adapter / "adapter_config.json").read_text())
    model, tok = load(str(a.model), adapter_path=str(a.init_adapter), trust_remote_code=False)

    # Encode all pairs
    def prep(split):
        out = []
        for r in split:
            c_ids, c_pl = encode_pair(tok, r["messages"], r["chosen"], a.max_len)
            r_ids, r_pl = encode_pair(tok, r["messages"], r["rejected"], a.max_len)
            if len(c_ids) - c_pl < 4 or len(r_ids) - r_pl < 2:
                continue
            out.append({"id": r.get("id", ""), "c_ids": c_ids, "c_pl": c_pl, "r_ids": r_ids, "r_pl": r_pl})
        return out

    train_enc = prep(train_rows)
    val_enc = prep(val_rows)
    print(f"Prepared DPO pairs: train={len(train_enc)} val={len(val_enc)}", flush=True)

    # Precompute reference log-probs using initial SFT adapter
    model.eval()
    for split_name, split in [("train", train_enc), ("val", val_enc)]:
        t_ref = time.time()
        for i, item in enumerate(split):
            c_sum, _, _ = seq_logprob(model, item["c_ids"], item["c_pl"])
            r_sum, _, _ = seq_logprob(model, item["r_ids"], item["r_pl"])
            mx.eval(c_sum, r_sum)
            item["ref_c"] = float(c_sum.item())
            item["ref_r"] = float(r_sum.item())
            if (i + 1) % 100 == 0 or (i + 1) == len(split):
                print(f"ref {split_name} {i+1}/{len(split)} ({time.time()-t_ref:.1f}s)", flush=True)
        mx.clear_cache()

    # Freeze base, train only LoRA parameters
    model.freeze()
    for _, m in model.named_modules():
        if hasattr(m, "lora_a") and hasattr(m, "lora_b"):
            m.unfreeze(keys=["lora_a", "lora_b"], recurse=False)

    def pair_loss(model, item):
        pol_c, mean_c, _ = seq_logprob(model, item["c_ids"], item["c_pl"])
        pol_r, _, _ = seq_logprob(model, item["r_ids"], item["r_pl"])
        chosen_reward = a.beta * (pol_c - item["ref_c"])
        rejected_reward = a.beta * (pol_r - item["ref_r"])
        margin = chosen_reward - rejected_reward
        dpo_loss = -nn.log_sigmoid(margin)
        nll_loss = -mean_c
        total = dpo_loss + a.nll_weight * nll_loss
        return total, (dpo_loss, nll_loss, margin)

    def eval_split(split):
        model.eval()
        tot_loss = tot_dpo = tot_margin = acc = 0.0
        for item in split:
            loss, (dpo_l, _, margin) = pair_loss(model, item)
            mx.eval(loss, dpo_l, margin)
            tot_loss += float(loss.item())
            tot_dpo += float(dpo_l.item())
            mval = float(margin.item())
            tot_margin += mval
            if mval > 0:
                acc += 1.0
        n = max(1, len(split))
        return {"loss": round(tot_loss / n, 4), "dpo_loss": round(tot_dpo / n, 4), "margin": round(tot_margin / n, 4), "pref_acc": round(acc / n, 4)}

    val_before = eval_split(val_enc)
    print(f"Val BEFORE DPO: {json.dumps(val_before)}", flush=True)

    loss_and_grad = nn.value_and_grad(model, pair_loss)
    total_updates = max(1, (len(train_enc) * a.epochs + a.grad_accum - 1) // a.grad_accum)
    schedule = optim.cosine_decay(a.lr, total_updates, a.lr * 0.1)
    optimizer = optim.AdamW(learning_rate=schedule, weight_decay=0.01)

    history = []
    update_step = 0
    for epoch in range(1, a.epochs + 1):
        rng.shuffle(train_enc)
        model.train()
        accum_grads = None
        accum_n = 0
        ep_loss = ep_margin = ep_acc = 0.0
        for idx, item in enumerate(train_enc, 1):
            (loss, (dpo_l, nll_l, margin)), grads = loss_and_grad(model, item)
            mx.eval(loss, dpo_l, margin, grads)
            if accum_grads is None:
                accum_grads = grads
            else:
                accum_grads = tree_map(lambda x, y: x + y, accum_grads, grads)
            accum_n += 1
            ep_loss += float(loss.item())
            mval = float(margin.item())
            ep_margin += mval
            if mval > 0:
                ep_acc += 1.0

            if accum_n == a.grad_accum or idx == len(train_enc):
                avg_grads = tree_map(lambda g: g / float(accum_n), accum_grads)
                clipped, gnorm = clip_grad_norm(avg_grads, max_norm=1.0)
                optimizer.update(model, clipped)
                mx.eval(model.parameters(), optimizer.state, gnorm)
                accum_grads = None
                accum_n = 0
                update_step += 1
                if update_step % 10 == 0 or update_step == total_updates:
                    print(
                        f"ep {epoch} step {update_step}/{total_updates} "
                        f"train_loss={ep_loss/idx:.4f} margin={ep_margin/idx:.4f} acc={ep_acc/idx:.3f} "
                        f"gnorm={float(gnorm.item()):.3f} ({time.time()-t0:.1f}s)",
                        flush=True,
                    )
        val_ep = eval_split(val_enc)
        history.append({"epoch": epoch, "val": val_ep, "train_loss": round(ep_loss / len(train_enc), 4), "train_margin": round(ep_margin / len(train_enc), 4), "train_acc": round(ep_acc / len(train_enc), 4)})
        print(f"Val AFTER epoch {epoch}: {json.dumps(val_ep)}", flush=True)
        mx.clear_cache()

    a.out_adapter.mkdir(parents=True, exist_ok=True)
    trainable = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(a.out_adapter / "adapters.safetensors"), trainable)
    (a.out_adapter / "adapter_config.json").write_text(json.dumps(adapter_cfg, indent=2) + "\n")
    summary = {
        "schema": "useless-v2-dpo-v1",
        "init_adapter": str(a.init_adapter),
        "data_sha256": sha(a.data),
        "train_pairs": len(train_enc),
        "val_pairs": len(val_enc),
        "epochs": a.epochs,
        "updates": update_step,
        "lr": a.lr,
        "beta": a.beta,
        "nll_weight": a.nll_weight,
        "val_before": val_before,
        "history": history,
        "adapter_sha256": sha(a.out_adapter / "adapters.safetensors"),
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    atomic_json(a.out_adapter / "dpo-summary.json", summary)
    print("DPO_COMPLETE", json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
