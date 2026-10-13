"""Prompt construction for the three zero update interventions.

The experiment hinges on a fair comparison, so prompting is treated as a first
class intervention rather than an afterthought. Three builders live here:

  * ``base_prompt``  few shot completion style prompt for an untuned base model
  * ``ica_prompt``   in context alignment, URIAL style: a system preamble plus a
                     small number of full demonstrations, no weight updates
  * ``chat_prompt``  the tuned model's own chat template

If the base model is prompted badly, it will look incapable for reasons that
have nothing to do with capacity, and the headline result becomes an artefact.
"""

from __future__ import annotations

from dataclasses import dataclass

# URIAL style system preamble. Describes the format contract explicitly so that
# the format axis has a fair chance of being satisfied without any tuning.
ICA_SYSTEM = (
    "You are a helpful, precise assistant. Answer the user's request directly "
    "and completely. Follow any formatting constraints in the request exactly. "
    "If the request asks for reasoning, show the steps briefly and then state "
    "the final answer on its own line. If you do not know something, say so "
    "instead of guessing."
)

# Three demonstrations covering the response modes the evaluation rewards:
# constrained formatting, short factual recall, and step by step arithmetic.
ICA_DEMOS: list[tuple[str, str]] = [
    (
        "List exactly three uses of a paperclip. Use a numbered list and no other text.",
        "1. Holding sheets of paper together\n"
        "2. Resetting a device through a pinhole button\n"
        "3. Bending into a small hook to hang a light object",
    ),
    (
        "Which gas do plants absorb from the atmosphere during photosynthesis?",
        "Plants absorb carbon dioxide from the atmosphere during photosynthesis.\n"
        "Final answer: carbon dioxide",
    ),
    (
        "A shop sells pens at 7 rupees each. Priya buys 6 pens and pays with a "
        "50 rupee note. How much change does she get?",
        "Six pens cost 6 x 7 = 42 rupees.\n"
        "Change from 50 rupees is 50 - 42 = 8 rupees.\n"
        "Final answer: 8",
    ),
]


@dataclass
class PromptBundle:
    """What the evaluator needs to score one example."""

    text: str
    stop_sequences: list[str]
    style: str


def base_prompt(question: str, demos: list[tuple[str, str]] | None = None,
                n_shot: int = 5) -> PromptBundle:
    """Completion style prompt, no instruction framing, for untuned base models."""
    demos = (demos or ICA_DEMOS)[:n_shot]
    parts = []
    for q, a in demos:
        parts.append(f"Question: {q}\nAnswer: {a}")
    parts.append(f"Question: {question}\nAnswer:")
    return PromptBundle(
        text="\n\n".join(parts),
        stop_sequences=["\nQuestion:", "\n\n\n"],
        style="base_fewshot",
    )


def ica_prompt(question: str, n_demos: int = 3,
               demos: list[tuple[str, str]] | None = None) -> PromptBundle:
    """In context alignment: system preamble plus full demonstrations."""
    demos = (demos or ICA_DEMOS)[:n_demos]
    parts = [f"# Instruction\n{ICA_SYSTEM}"]
    for i, (q, a) in enumerate(demos, start=1):
        parts.append(f"# Example {i}\n## User\n{q}\n\n## Assistant\n{a}")
    parts.append(f"# Task\n## User\n{question}\n\n## Assistant\n")
    return PromptBundle(
        text="\n\n".join(parts),
        stop_sequences=["\n# Example", "\n# Task", "\n## User"],
        style=f"ica_{n_demos}shot",
    )


def chat_prompt(question: str, tokenizer, system: str | None = None) -> PromptBundle:
    """Use the tuned model's chat template when it has one, else a plain fallback."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": question})

    template = getattr(tokenizer, "chat_template", None)
    if template:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        prefix = f"<|system|>\n{system}\n" if system else ""
        text = f"{prefix}<|user|>\n{question}\n<|assistant|>\n"
    return PromptBundle(
        text=text,
        stop_sequences=["<|user|>", "<|endoftext|>", "<|im_end|>"],
        style="chat_template",
    )


def build(question: str, intervention: str, tokenizer=None,
          n_demos: int = 3, n_shot: int = 5) -> PromptBundle:
    """Dispatch on the intervention name used in configs and result filenames."""
    if intervention == "base":
        return base_prompt(question, n_shot=n_shot)
    if intervention == "ica":
        return ica_prompt(question, n_demos=n_demos)
    if intervention in ("sft", "sft_dpo", "cpt_sft"):
        if tokenizer is None:
            raise ValueError("a tokenizer is required to build a chat prompt")
        return chat_prompt(question, tokenizer)
    raise ValueError(f"unknown intervention {intervention!r}")


if __name__ == "__main__":
    print(ica_prompt("Summarise photosynthesis in one sentence.").text)
