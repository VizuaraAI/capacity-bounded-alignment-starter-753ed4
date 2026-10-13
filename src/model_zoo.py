"""The capacity ladder.

Every claim in this project is a claim about a point on this ladder, so the
ladder is defined in exactly one place. Keep the entries sorted by parameter
count: several scripts rely on that ordering when they plot scale on the x axis.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelSpec:
    key: str
    hf_id: str
    params: int               # non embedding parameter count, approximate
    family: str
    label: str                # short label used on figure axes
    chat_template: bool       # does the tokenizer ship a chat template
    notes: str = ""
    tags: tuple = field(default_factory=tuple)

    @property
    def params_b(self) -> float:
        return self.params / 1e9

    @property
    def below_threshold(self) -> bool:
        """True for the sub 1B regime where the hypothesis is expected to fail."""
        return self.params < 1_000_000_000


LADDER: dict[str, ModelSpec] = {
    "smollm2-135m": ModelSpec(
        key="smollm2-135m",
        hf_id="HuggingFaceTB/SmolLM2-135M",
        params=135_000_000,
        family="smollm2",
        label="135M",
        chat_template=False,
        notes="Smallest rung. Expect near zero recoverable capability from style tuning.",
        tags=("primary", "sub1b"),
    ),
    "smollm2-360m": ModelSpec(
        key="smollm2-360m",
        hf_id="HuggingFaceTB/SmolLM2-360M",
        params=360_000_000,
        family="smollm2",
        label="360M",
        chat_template=False,
        notes="Main ablation rung: cheap enough for full fine tuning sweeps.",
        tags=("primary", "sub1b", "ablation-anchor"),
    ),
    "qwen2.5-0.5b": ModelSpec(
        key="qwen2.5-0.5b",
        hf_id="Qwen/Qwen2.5-0.5B",
        params=500_000_000,
        family="qwen2.5",
        label="0.5B",
        chat_template=True,
        notes="Cross family control just below the hypothesised threshold.",
        tags=("primary", "sub1b"),
    ),
    "llama3.2-1b": ModelSpec(
        key="llama3.2-1b",
        hf_id="meta-llama/Llama-3.2-1B",
        params=1_000_000_000,
        family="llama3.2",
        label="1B",
        chat_template=True,
        notes="Sits on the threshold. Gated repository: accept the licence first.",
        tags=("primary",),
    ),
    "smollm2-1.7b": ModelSpec(
        key="smollm2-1.7b",
        hf_id="HuggingFaceTB/SmolLM2-1.7B",
        params=1_700_000_000,
        family="smollm2",
        label="1.7B",
        chat_template=False,
        notes="Top of the same family ladder, so capacity is the only thing that moved.",
        tags=("primary",),
    ),
    "qwen2.5-1.5b": ModelSpec(
        key="qwen2.5-1.5b",
        hf_id="Qwen/Qwen2.5-1.5B",
        params=1_500_000_000,
        family="qwen2.5",
        label="1.5B",
        chat_template=True,
        notes="Cross family control above the threshold.",
        tags=("secondary",),
    ),
    # Pythia gives a single family trained on identical data in identical order,
    # which is the cleanest possible capacity control even though the models are weaker.
    "pythia-160m": ModelSpec(
        key="pythia-160m",
        hf_id="EleutherAI/pythia-160m",
        params=160_000_000,
        family="pythia",
        label="160M (Pythia)",
        chat_template=False,
        notes="Controlled ladder: same corpus, same data order, capacity is the only variable.",
        tags=("control",),
    ),
    "pythia-410m": ModelSpec(
        key="pythia-410m",
        hf_id="EleutherAI/pythia-410m",
        params=410_000_000,
        family="pythia",
        label="410M (Pythia)",
        chat_template=False,
        notes="Controlled ladder rung.",
        tags=("control",),
    ),
    "pythia-1.4b": ModelSpec(
        key="pythia-1.4b",
        hf_id="EleutherAI/pythia-1.4b",
        params=1_400_000_000,
        family="pythia",
        label="1.4B (Pythia)",
        chat_template=False,
        notes="Controlled ladder rung above the threshold.",
        tags=("control",),
    ),
}

# The five rungs that fill the main results table of the paper.
PRIMARY_LADDER = [k for k, spec in LADDER.items() if "primary" in spec.tags]

# The Pythia rungs, used for the controlled capacity ablation.
CONTROL_LADDER = [k for k, spec in LADDER.items() if "control" in spec.tags]


def get(key: str) -> ModelSpec:
    if key not in LADDER:
        raise KeyError(
            f"Unknown model key {key!r}. Known keys: {sorted(LADDER)}"
        )
    return LADDER[key]


def sorted_keys(keys: list[str] | None = None) -> list[str]:
    """Return keys ordered by parameter count, which is figure x axis order."""
    keys = list(LADDER) if keys is None else list(keys)
    return sorted(keys, key=lambda k: get(k).params)


if __name__ == "__main__":
    for key in sorted_keys():
        spec = get(key)
        flag = "sub-1B" if spec.below_threshold else "      "
        print(f"{spec.label:>14}  {flag}  {spec.hf_id}")
