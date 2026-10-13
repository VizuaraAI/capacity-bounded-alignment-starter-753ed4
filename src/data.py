"""Dataset loading for training interventions and for the two evaluation axes.

Nothing is vendored into the repository. Every loader pulls a public dataset
from the Hugging Face hub and normalises it into one of two shapes:

  training    {"messages": [{"role": ..., "content": ...}, ...]}
  evaluation  {"question": str, "answer": str, "meta": dict}

Keeping the shapes fixed is what lets the same evaluator score a base model, an
in context aligned model and a fine tuned model without special cases.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from datasets import load_dataset

# Training mixtures, ordered by how much new capability they could plausibly install.
SFT_SOURCES = {
    # LIMA: the 1K curated set behind the Superficial Alignment Hypothesis itself.
    # Gated on the hub, so accept the terms once at the dataset page.
    "lima": ("GAIR/lima", "train"),
    # Scaled, modern, open SFT mixture. Subsample it to build the data ladder.
    "tulu3": ("allenai/tulu-3-sft-mixture", "train"),
    # Self instruct style data, useful as a quality contrast at fixed size.
    "alpaca": ("tatsu-lab/alpaca", "train"),
}

EVAL_SOURCES = {
    "gsm8k": ("openai/gsm8k", "main", "test"),
    "mmlu": ("cais/mmlu", "all", "test"),
    "arc_challenge": ("allenai/ai2_arc", "ARC-Challenge", "test"),
    "ifeval_subset": ("google/IFEval", None, "train"),
}


@dataclass
class EvalExample:
    question: str
    answer: str
    meta: dict

    def as_dict(self) -> dict:
        return {"question": self.question, "answer": self.answer, "meta": self.meta}


# --------------------------------------------------------------------------- #
# Training data
# --------------------------------------------------------------------------- #

def _lima_to_messages(row: dict) -> dict:
    """LIMA stores a flat conversation list, alternating user and assistant."""
    turns = row.get("conversations") or []
    messages = []
    for i, turn in enumerate(turns):
        messages.append({
            "role": "user" if i % 2 == 0 else "assistant",
            "content": turn,
        })
    return {"messages": messages}


def _alpaca_to_messages(row: dict) -> dict:
    instruction = row.get("instruction", "")
    extra = (row.get("input") or "").strip()
    user = f"{instruction}\n\n{extra}" if extra else instruction
    return {"messages": [
        {"role": "user", "content": user},
        {"role": "assistant", "content": row.get("output", "")},
    ]}


def _tulu_to_messages(row: dict) -> dict:
    """Tulu 3 already uses the messages schema; drop anything malformed."""
    messages = [
        {"role": m["role"], "content": m["content"]}
        for m in row.get("messages", [])
        if m.get("role") in ("system", "user", "assistant") and m.get("content")
    ]
    return {"messages": messages}


_CONVERTERS = {
    "lima": _lima_to_messages,
    "alpaca": _alpaca_to_messages,
    "tulu3": _tulu_to_messages,
}


def load_sft_dataset(name: str, n_train: int | None = None, seed: int = 17):
    """Load and normalise one SFT mixture.

    ``n_train`` is the knob that builds the data ladder (1K, 10K, 50K, 200K).
    The subsample is seeded so that the 10K set is a superset ordering of the
    1K set under the same seed, which keeps the ladder interpretable.
    """
    if name == "none":
        return None
    if name not in SFT_SOURCES:
        raise KeyError(f"unknown SFT source {name!r}; known: {sorted(SFT_SOURCES)}")

    repo, split = SFT_SOURCES[name]
    ds = load_dataset(repo, split=split)
    ds = ds.map(_CONVERTERS[name], remove_columns=ds.column_names)
    ds = ds.filter(lambda r: len(r["messages"]) >= 2)

    if n_train is not None and n_train < len(ds):
        ds = ds.shuffle(seed=seed).select(range(n_train))
    return ds


def load_dpo_dataset(repo: str = "HuggingFaceH4/ultrafeedback_binarized",
                     n_train: int | None = 10_000, seed: int = 17):
    """Binarised preference data for the DPO rung of the intervention ladder."""
    ds = load_dataset(repo, split="train_prefs")
    if n_train is not None and n_train < len(ds):
        ds = ds.shuffle(seed=seed).select(range(n_train))
    return ds


# --------------------------------------------------------------------------- #
# Evaluation data
# --------------------------------------------------------------------------- #

def _load_gsm8k(n: int, seed: int) -> list[EvalExample]:
    ds = load_dataset("openai/gsm8k", "main", split="test").shuffle(seed=seed)
    out = []
    for row in ds.select(range(min(n, len(ds)))):
        gold = row["answer"].split("####")[-1].strip()
        out.append(EvalExample(row["question"], gold, {"task": "gsm8k", "type": "generative"}))
    return out


def _load_mmlu(n: int, seed: int) -> list[EvalExample]:
    ds = load_dataset("cais/mmlu", "all", split="test").shuffle(seed=seed)
    letters = ["A", "B", "C", "D"]
    out = []
    for row in ds.select(range(min(n, len(ds)))):
        choices = row["choices"]
        rendered = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(choices))
        question = f"{row['question']}\n{rendered}"
        out.append(EvalExample(
            question,
            letters[int(row["answer"])],
            {"task": "mmlu", "type": "multiple_choice", "choices": list(choices),
             "subject": row.get("subject", "")},
        ))
    return out


def _load_arc(n: int, seed: int) -> list[EvalExample]:
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test").shuffle(seed=seed)
    out = []
    for row in ds.select(range(min(n, len(ds)))):
        labels = row["choices"]["label"]
        texts = row["choices"]["text"]
        rendered = "\n".join(f"{l}. {t}" for l, t in zip(labels, texts))
        out.append(EvalExample(
            f"{row['question']}\n{rendered}",
            row["answerKey"],
            {"task": "arc_challenge", "type": "multiple_choice", "choices": list(texts)},
        ))
    return out


def _load_ifeval(n: int, seed: int) -> list[EvalExample]:
    """IFEval prompts carry verifiable constraints rather than gold answers."""
    ds = load_dataset("google/IFEval", split="train").shuffle(seed=seed)
    out = []
    for row in ds.select(range(min(n, len(ds)))):
        out.append(EvalExample(
            row["prompt"],
            "",
            {"task": "ifeval_subset", "type": "constraint",
             "instruction_ids": list(row.get("instruction_id_list", [])),
             "kwargs": row.get("kwargs", [])},
        ))
    return out


def _load_structure_compliance(n: int, seed: int) -> list[EvalExample]:
    """A small local format probe, so the format axis does not rest on one dataset.

    Each prompt has a constraint that can be checked with a short rule, which
    keeps the format axis free of judge model noise.
    """
    rng = random.Random(seed)
    topics = ["a bicycle", "photosynthesis", "a monsoon", "a library card",
              "a kettle", "the moon", "a train ticket", "a calculator"]
    templates = [
        ("Describe {t} in exactly one sentence. Do not use any lists.", "one_sentence"),
        ("Give three facts about {t} as a numbered list and nothing else.", "numbered_three"),
        ("Explain {t} in under 25 words.", "max_25_words"),
        ("Answer in valid JSON with the single key \"summary\": what is {t}?", "json_summary"),
    ]
    out = []
    for i in range(n):
        topic = rng.choice(topics)
        template, rule = templates[i % len(templates)]
        out.append(EvalExample(
            template.format(t=topic),
            "",
            {"task": "structure_compliance", "type": "constraint", "rule": rule},
        ))
    return out


_EVAL_LOADERS = {
    "gsm8k": _load_gsm8k,
    "mmlu": _load_mmlu,
    "arc_challenge": _load_arc,
    "ifeval_subset": _load_ifeval,
    "structure_compliance": _load_structure_compliance,
}


def load_eval_task(task: str, n: int = 256, seed: int = 17) -> list[EvalExample]:
    if task not in _EVAL_LOADERS:
        raise KeyError(f"unknown eval task {task!r}; known: {sorted(_EVAL_LOADERS)}")
    return _EVAL_LOADERS[task](n, seed)


def load_open_prompts(n: int = 128, seed: int = 17) -> list[str]:
    """Open ended prompts used for the token distribution shift analysis."""
    ds = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval", split="eval")
    ds = ds.shuffle(seed=seed)
    return [row["instruction"] for row in ds.select(range(min(n, len(ds))))]


if __name__ == "__main__":
    probe = load_eval_task("structure_compliance", n=4)
    for ex in probe:
        print(f"[{ex.meta['rule']}] {ex.question}")
