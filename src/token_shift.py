"""Token distribution shift between a base model and its aligned twin.

The mechanistic version of the Superficial Alignment Hypothesis says alignment
only rewrites a thin stylistic surface: the aligned model's token choices differ
from the base model's mostly at the start of a response and mostly on discourse
and formatting tokens. If that picture is right for a 1.7B model but wrong for a
135M model, the shift profile should look visibly different, and that difference
is independent of any benchmark score.

    python -m src.token_shift --base-model smollm2-360m \
        --aligned runs/smollm2-360m__sft-lima-1000__seed17

What is measured, per response position:
  * KL(aligned next token distribution || base next token distribution)
  * whether the base model's top 1 token differs from the aligned model's choice
Aggregated into a shifted fraction, an early versus late split, and the full
positional curve that feeds the analysis plot.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import data as data_mod
from . import model_zoo
from . import prompts as prompt_mod

# Positions at or below this index count as "early", which is where the
# hypothesis predicts the shift should be concentrated.
EARLY_WINDOW = 5


def _kl_rowwise(p_logits, q_logits):
    """KL(p || q) over the vocabulary, for aligned logits p and base logits q."""
    import torch
    log_p = torch.log_softmax(p_logits.float(), dim=-1)
    log_q = torch.log_softmax(q_logits.float(), dim=-1)
    p = log_p.exp()
    return (p * (log_p - log_q)).sum(dim=-1)


def analyse_prompt(base_model, aligned_model, tokenizer, prompt: str,
                   max_new_tokens: int, kl_threshold: float) -> dict:
    """Generate once with the aligned model, then re-score under both models."""
    import torch

    bundle = prompt_mod.chat_prompt(prompt, tokenizer)
    inputs = tokenizer(bundle.text, return_tensors="pt", truncation=True,
                       max_length=1024).to(aligned_model.device)
    with torch.no_grad():
        generated = aligned_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    n_prompt = inputs["input_ids"].shape[1]
    if generated.shape[1] <= n_prompt + 1:
        return {"n_positions": 0, "kl": [], "shifted": [], "tokens": []}

    ids = generated.to(base_model.device)
    with torch.no_grad():
        aligned_logits = aligned_model(generated).logits[0]
        base_logits = base_model(ids).logits[0]

    # Position i of the response is predicted by logits at index n_prompt + i - 1.
    start = n_prompt - 1
    end = generated.shape[1] - 1
    a_slice = aligned_logits[start:end]
    b_slice = base_logits[start:end]

    kl = _kl_rowwise(a_slice, b_slice).cpu().numpy()
    chosen = generated[0, n_prompt:].cpu().numpy()
    base_top1 = b_slice.argmax(dim=-1).cpu().numpy()
    shifted = (base_top1 != chosen).astype(float)

    return {
        "n_positions": int(len(kl)),
        "kl": [float(x) for x in kl],
        "shifted": [float(x) for x in shifted],
        "tokens": tokenizer.convert_ids_to_tokens(chosen.tolist()),
        "shifted_fraction": float(shifted.mean()),
        "high_kl_fraction": float((kl > kl_threshold).mean()),
    }


def aggregate(records: list[dict], max_positions: int = 64) -> dict:
    """Collapse per prompt records into the numbers the paper reports."""
    usable = [r for r in records if r["n_positions"] > 0]
    if not usable:
        return {"n_prompts": 0, "note": "no usable generations"}

    curve_sum = np.zeros(max_positions)
    curve_count = np.zeros(max_positions)
    early, late, all_kl, all_shift = [], [], [], []

    for rec in usable:
        kl = np.asarray(rec["kl"], dtype=float)
        shift = np.asarray(rec["shifted"], dtype=float)
        all_kl.extend(kl.tolist())
        all_shift.extend(shift.tolist())
        n = min(len(kl), max_positions)
        curve_sum[:n] += kl[:n]
        curve_count[:n] += 1
        early.extend(kl[:EARLY_WINDOW].tolist())
        if len(kl) > EARLY_WINDOW:
            late.extend(kl[EARLY_WINDOW:].tolist())

    with np.errstate(invalid="ignore", divide="ignore"):
        curve = np.where(curve_count > 0, curve_sum / np.maximum(curve_count, 1), np.nan)

    mean_early = float(np.mean(early)) if early else float("nan")
    mean_late = float(np.mean(late)) if late else float("nan")
    return {
        "n_prompts": len(usable),
        "mean_kl": float(np.mean(all_kl)),
        "median_kl": float(np.median(all_kl)),
        "shifted_fraction": float(np.mean(all_shift)),
        "mean_kl_early": mean_early,
        "mean_kl_late": mean_late,
        # Greater than 1 means the shift really is front loaded and stylistic,
        # which is what the hypothesis predicts. Near 1 means it runs all the
        # way through the response, which is not a surface edit.
        "early_late_ratio": mean_early / mean_late if mean_late and mean_late > 0 else None,
        "kl_by_position": [None if np.isnan(v) else float(v) for v in curve],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Token distribution shift analysis")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--base-model", required=True, dest="base_model",
                        help=f"model key, one of {sorted(model_zoo.LADDER)}")
    parser.add_argument("--aligned", required=True,
                        help="aligned checkpoint directory or hub id")
    parser.add_argument("--n-prompts", type=int, dest="n_prompts")
    parser.add_argument("--out")
    args = parser.parse_args()

    from .train_sft import load_config
    cfg = load_config(args.config)
    n_prompts = args.n_prompts or int(cfg["analysis"]["tds_n_prompts"])
    kl_threshold = float(cfg["analysis"]["tds_kl_threshold"])

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    seed = int(cfg["experiment"]["seed"])
    set_seed(seed)
    spec = model_zoo.get(args.base_model)
    dtype = getattr(torch, cfg["model"]["dtype"])

    # The aligned checkpoint's tokenizer is authoritative: it is the one that
    # produced the tokens being compared.
    tokenizer = AutoTokenizer.from_pretrained(args.aligned)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(spec.hf_id, torch_dtype=dtype).eval()
    aligned_model = AutoModelForCausalLM.from_pretrained(args.aligned, torch_dtype=dtype).eval()
    if torch.cuda.is_available():
        base_model.cuda()
        aligned_model.cuda()

    if base_model.config.vocab_size != aligned_model.config.vocab_size:
        raise SystemExit(
            "vocabulary sizes differ, so a token level KL is not meaningful. "
            "Compare a checkpoint against the base model it was tuned from."
        )

    prompts = data_mod.load_open_prompts(n=n_prompts, seed=seed)
    records = []
    for i, prompt in enumerate(prompts, start=1):
        records.append(analyse_prompt(
            base_model, aligned_model, tokenizer, prompt,
            max_new_tokens=int(cfg["eval"]["max_new_tokens"]),
            kl_threshold=kl_threshold,
        ))
        if i % 16 == 0:
            print(f"[tds] {i}/{len(prompts)} prompts")

    summary = aggregate(records)
    summary.update({"base_model": spec.key, "aligned": args.aligned, "seed": seed})
    ratio = summary.get("early_late_ratio")
    print(f"[tds] shifted fraction {summary.get('shifted_fraction'):.3f}  "
          f"mean KL {summary.get('mean_kl'):.3f}  "
          f"early/late ratio {ratio if ratio is None else round(ratio, 2)}")

    out_path = Path(args.out) if args.out else Path("results") / (
        f"tds__{spec.key}__seed{seed}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[tds] wrote {out_path}")


if __name__ == "__main__":
    main()
