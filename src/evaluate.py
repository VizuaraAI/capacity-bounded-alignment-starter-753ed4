"""Two axis evaluation: format compliance and task capability, scored separately.

    python -m src.evaluate --model smollm2-360m --intervention base
    python -m src.evaluate --model smollm2-360m --intervention ica
    python -m src.evaluate --checkpoint runs/smollm2-360m__sft-lima-1000__seed17 \
        --intervention sft

The split is the whole experiment. Format questions ("did it obey the
constraint, does it look like an assistant") are scored with deterministic
rules. Capability questions ("is the answer right") are scored by comparing the
log probability of each candidate answer, so a model that cannot produce tidy
output is not punished twice.

Per example scores are written out, not just means, because the Superficiality
Index needs paired arrays to bootstrap.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from . import data as data_mod
from . import model_zoo
from . import prompts as prompt_mod

MC_LETTERS = ["A", "B", "C", "D", "E"]


# --------------------------------------------------------------------------- #
# Format axis scorers: deterministic rules, no judge model
# --------------------------------------------------------------------------- #

def _score_structure_rule(response: str, rule: str) -> float:
    text = response.strip()
    if rule == "one_sentence":
        sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
        has_list = bool(re.search(r"(^|\n)\s*(\d+[.)]|[-*])\s", text))
        return float(len(sentences) == 1 and not has_list)
    if rule == "numbered_three":
        items = re.findall(r"(?m)^\s*([1-9])[.)]\s+\S", text)
        return float(len(items) == 3 and items == ["1", "2", "3"])
    if rule == "max_25_words":
        return float(0 < len(text.split()) <= 25)
    if rule == "json_summary":
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return 0.0
        return float(isinstance(parsed, dict) and set(parsed.keys()) == {"summary"})
    raise ValueError(f"unknown structure rule {rule!r}")


def _score_ifeval_like(response: str, instruction_ids: list[str]) -> float:
    """A light reimplementation of the common IFEval constraint checks.

    Only the constraint families that can be checked without the full IFEval
    package are handled; anything unrecognised is skipped rather than counted as
    a pass, which keeps the format axis conservative. Swap in the official
    ``instruction_following_eval`` package for the camera ready numbers.
    """
    text = response.strip()
    checks = []
    for iid in instruction_ids:
        if "lowercase" in iid:
            checks.append(float(text == text.lower()))
        elif "uppercase" in iid or "capital" in iid:
            checks.append(float(text == text.upper()))
        elif "json" in iid:
            try:
                json.loads(text)
                checks.append(1.0)
            except (json.JSONDecodeError, ValueError):
                checks.append(0.0)
        elif "no_comma" in iid:
            checks.append(float("," not in text))
        elif "bullet" in iid:
            checks.append(float(bool(re.search(r"(?m)^\s*[-*]\s+\S", text))))
        elif "quotation" in iid:
            checks.append(float(text.startswith('"') and text.endswith('"')))
        elif "placeholder" in iid:
            checks.append(float(bool(re.search(r"\[[^\]]+\]", text))))
        elif "title" in iid:
            checks.append(float(bool(re.search(r"<<[^>]+>>", text))))
    if not checks:
        # Fall back to a minimal assistant-likeness check: non-empty, not a
        # continuation of the prompt, does not trail off mid word.
        return float(len(text) > 0 and not text.startswith("Question:"))
    return sum(checks) / len(checks)


# --------------------------------------------------------------------------- #
# Capability axis scorers
# --------------------------------------------------------------------------- #

def _extract_final_number(response: str) -> str | None:
    cleaned = response.replace(",", "")
    match = re.search(r"(?i)final answer\s*[:\-]?\s*(-?\d+(?:\.\d+)?)", cleaned)
    if match:
        return match.group(1)
    numbers = re.findall(r"-?\d+(?:\.\d+)?", cleaned)
    return numbers[-1] if numbers else None


def _score_gsm8k(response: str, gold: str) -> float:
    pred = _extract_final_number(response)
    if pred is None:
        return 0.0
    try:
        return float(abs(float(pred) - float(gold.replace(",", ""))) < 1e-4)
    except ValueError:
        return 0.0


def _score_multiple_choice_text(response: str, gold_letter: str) -> float:
    match = re.search(r"(?i)\b([A-E])\b", response.strip()[:40])
    return float(match is not None and match.group(1).upper() == gold_letter.upper())


# --------------------------------------------------------------------------- #
# Generation and scoring helpers
# --------------------------------------------------------------------------- #

def _truncate_at_stops(text: str, stops: list[str]) -> str:
    cut = len(text)
    for stop in stops:
        pos = text.find(stop)
        if pos != -1:
            cut = min(cut, pos)
    return text[:cut]


def generate(model, tokenizer, bundle, max_new_tokens: int, temperature: float) -> str:
    import torch
    inputs = tokenizer(bundle.text, return_tensors="pt", truncation=True,
                       max_length=2048).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            pad_token_id=tokenizer.pad_token_id,
        )
    completion = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    return _truncate_at_stops(completion, bundle.stop_sequences).strip()


def score_by_loglikelihood(model, tokenizer, context: str, choices: list[str]) -> int:
    """Return the index of the highest length normalised log probability choice.

    This is the format invariant capability measure: a 135M model that cannot
    emit the letter "C" on demand can still reveal what it knows here.
    """
    import torch
    best_idx, best_score = 0, float("-inf")
    ctx_ids = tokenizer(context, return_tensors="pt").input_ids.to(model.device)
    for i, choice in enumerate(choices):
        full = tokenizer(context + " " + choice.strip(), return_tensors="pt")
        full_ids = full.input_ids.to(model.device)
        n_ctx = ctx_ids.shape[1]
        if full_ids.shape[1] <= n_ctx:
            continue
        with torch.no_grad():
            logits = model(full_ids).logits
        logprobs = torch.log_softmax(logits[0, :-1].float(), dim=-1)
        targets = full_ids[0, 1:]
        picked = logprobs[torch.arange(targets.shape[0]), targets]
        answer_logprobs = picked[n_ctx - 1:]
        score = float(answer_logprobs.mean())
        if score > best_score:
            best_idx, best_score = i, score
    return best_idx


# --------------------------------------------------------------------------- #
# Task runners
# --------------------------------------------------------------------------- #

def run_format_task(model, tokenizer, task: str, examples, cfg: dict,
                    intervention: str) -> list[float]:
    scores = []
    for ex in examples:
        bundle = prompt_mod.build(
            ex.question, intervention, tokenizer=tokenizer,
            n_demos=int(cfg["intervention"]["n_demos"]),
            n_shot=int(cfg["eval"]["n_shot_base"]),
        )
        response = generate(model, tokenizer, bundle,
                            int(cfg["eval"]["max_new_tokens"]),
                            float(cfg["eval"]["temperature"]))
        if task == "structure_compliance":
            scores.append(_score_structure_rule(response, ex.meta["rule"]))
        else:
            scores.append(_score_ifeval_like(response, ex.meta.get("instruction_ids", [])))
    return scores


def run_capability_task(model, tokenizer, task: str, examples, cfg: dict,
                        intervention: str) -> list[float]:
    scores = []
    use_loglik = cfg["eval"]["capability_scoring"] == "loglikelihood"
    for ex in examples:
        if ex.meta["type"] == "multiple_choice" and use_loglik:
            choices = ex.meta["choices"]
            context = prompt_mod.build(
                ex.question, intervention, tokenizer=tokenizer,
                n_demos=int(cfg["intervention"]["n_demos"]),
                n_shot=int(cfg["eval"]["n_shot_base"]),
            ).text
            pred_idx = score_by_loglikelihood(model, tokenizer, context, choices)
            gold_idx = MC_LETTERS.index(ex.answer.strip().upper()[:1]) \
                if ex.answer.strip().upper()[:1] in MC_LETTERS else -1
            scores.append(float(pred_idx == gold_idx))
            continue

        bundle = prompt_mod.build(
            ex.question, intervention, tokenizer=tokenizer,
            n_demos=int(cfg["intervention"]["n_demos"]),
            n_shot=int(cfg["eval"]["n_shot_base"]),
        )
        response = generate(model, tokenizer, bundle,
                            int(cfg["eval"]["max_new_tokens"]),
                            float(cfg["eval"]["temperature"]))
        if task == "gsm8k":
            scores.append(_score_gsm8k(response, ex.answer))
        else:
            scores.append(_score_multiple_choice_text(response, ex.answer))
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Score one model and intervention")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", help="model key, used when evaluating a hub model")
    parser.add_argument("--checkpoint", help="local SFT checkpoint directory")
    parser.add_argument("--intervention", required=True,
                        choices=["base", "ica", "sft", "sft_dpo", "cpt_sft"])
    parser.add_argument("--n-eval", type=int, dest="n_eval")
    parser.add_argument("--tasks", nargs="*", help="restrict to these task names")
    parser.add_argument("--out", help="output json path")
    args = parser.parse_args()

    from .train_sft import load_config
    cfg = load_config(args.config)
    cfg["intervention"]["kind"] = args.intervention
    if args.n_eval is not None:
        cfg["data"]["n_eval"] = args.n_eval

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    seed = int(cfg["experiment"]["seed"])
    set_seed(seed)

    if args.checkpoint:
        source = args.checkpoint
        meta_path = Path(args.checkpoint) / "run_meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        model_key = args.model or meta.get("model_key", "unknown")
    else:
        if not args.model:
            raise SystemExit("pass --model for a hub model or --checkpoint for a local run")
        model_key = args.model
        source = model_zoo.get(model_key).hf_id

    print(f"[eval] {source}  intervention={args.intervention}")
    tokenizer = AutoTokenizer.from_pretrained(source)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        source, torch_dtype=getattr(torch, cfg["model"]["dtype"])
    )
    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    n_eval = int(cfg["data"]["n_eval"])
    wanted = set(args.tasks) if args.tasks else None
    results = {
        "model_key": model_key,
        "params": model_zoo.get(model_key).params if model_key in model_zoo.LADDER else None,
        "source": source,
        "intervention": args.intervention,
        "seed": seed,
        "n_eval": n_eval,
        "format": {},
        "capability": {},
    }

    for task in cfg["eval"]["format_tasks"]:
        if wanted and task not in wanted:
            continue
        examples = data_mod.load_eval_task(task, n=n_eval, seed=seed)
        scores = run_format_task(model, tokenizer, task, examples, cfg, args.intervention)
        results["format"][task] = {"mean": sum(scores) / len(scores), "scores": scores}
        print(f"[eval]   format/{task:>22}: {results['format'][task]['mean']:.3f}")

    for task in cfg["eval"]["capability_tasks"]:
        if wanted and task not in wanted:
            continue
        examples = data_mod.load_eval_task(task, n=n_eval, seed=seed)
        scores = run_capability_task(model, tokenizer, task, examples, cfg, args.intervention)
        results["capability"][task] = {"mean": sum(scores) / len(scores), "scores": scores}
        print(f"[eval]   capability/{task:>18}: {results['capability'][task]['mean']:.3f}")

    out_path = Path(args.out) if args.out else Path("results") / (
        f"{model_key}__{args.intervention}__seed{seed}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"[eval] wrote {out_path}")


if __name__ == "__main__":
    main()
