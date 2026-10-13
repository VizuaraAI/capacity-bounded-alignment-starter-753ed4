# CLAUDE.md

Instructions for the coding agent working on this repository with Vishnuvardhan.

## What this project is

An empirical study testing whether the Superficial Alignment Hypothesis (LIMA,
Zhou et al. 2023) is scale-bounded. The hypothesis says pretraining installs
capability and instruction tuning only selects response format. That presupposes
the capability was installed, which is exactly what fails below 1B parameters.

The deliverable is a workshop paper, not a product. Target: EACL 2027 Student
Research Workshop, deadline 15 December 2026. The paper draft lives in
`../paper/main.tex` with `XX.X` placeholders where results go.

## The central construct

```
S = (P_ica - P_base) / (P_sft - P_base)
```

computed **separately** for a format axis and a capability axis. Defined in
`src/metrics.py` and nowhere else. If you need it somewhere, import it.

The expected result: `S` near 1 on the format axis at every scale, falling
toward 0 on the capability axis as models shrink.

## Repository conventions

- **Python 3.10+, standard library plus the pinned packages in
  `requirements.txt`.** Do not add a dependency without saying why in the PR
  description; this project is meant to stay readable in one sitting.
- **Config lives in `configs/default.yaml`.** Anything a reader might want to
  change belongs there, with a CLI override in the relevant entry point. Do not
  hardcode hyperparameters inside functions.
- **The model ladder is defined once**, in `src/model_zoo.py`. Never write a
  Hugging Face model id as a string literal anywhere else.
- **Evaluators write per-example scores, not just means.** The bootstrap needs
  paired arrays. A results file without a `scores` list is unusable.
- **Every training run writes `run_meta.json`.** A run that is not recorded
  does not exist as far as the analysis is concerned.
- **Entry points are modules**: `python -m src.train_sft`, not
  `python src/train_sft.py`.
- **Imports stay clean.** Heavy imports (torch, transformers) go inside `main()`
  so `--help` works on a machine without a GPU stack.
- Prose in code comments and documentation: no em dashes, use commas, colons,
  parentheses or semicolons.

## Scientific rules that are not negotiable

These are the things that make the result real rather than an artefact. If a
change would violate one, stop and raise it rather than working around it.

1. **The base model must be prompted fairly.** Few-shot, with the same
   demonstrations available to the in-context arm. An unfairly prompted base
   model inflates every gap in the paper. This is the single most likely way
   for this project to produce a wrong answer.
2. **Capability is scored by log-likelihood, not by parsing generated text**
   for multiple-choice items. Otherwise a small model's inability to emit "C"
   is recorded as not knowing the answer, and the format deficit leaks into the
   capability measurement, which destroys the entire decomposition.
3. **No LLM judge on the capability axis.** Judges are unreliable on weak
   sub-1B outputs and biased toward length.
4. **Always report length-controlled win rate** beside any raw win rate.
   Style-only tuning inflates verbosity and will fake a result.
5. **Never report `S` without its bootstrap interval.** It is a ratio of two
   noisy differences. When the denominator is near zero, `S` is undefined:
   report raw scores instead. `src/metrics.py` already handles this; do not
   "fix" it by returning a number.
6. **Report continuous metrics beside thresholded ones.** Apparent cliffs at
   scale are often artefacts of the metric (Schaeffer et al. 2023).
7. **Three seeds minimum at the smallest rungs** before claiming any gap.
8. **Decontaminate** training mixtures against evaluation sets before reporting.

## Current state

Scaffold complete and import-clean. Nothing has been run yet.

`src/evaluate.py` contains a lightweight reimplementation of the common IFEval
constraint checks (`_score_ifeval_like`) covering only the families checkable
without the official package. It is deliberately conservative: unrecognised
constraints are skipped rather than counted as passes. **Before the camera
ready, swap in the official `instruction_following_eval` package** and rerun
the format axis.

## Suggested next steps, in order

1. Run `bash scripts/run_baseline.sh` and read `analysis/superficiality.csv`.
   Confirm the format axis and capability axis separate at 360M.
2. Manually read 20 base-model generations. Confirm the baseline is not
   artificially handicapped. Adjust `src/prompts.py` if it is.
3. Extend to the full ladder with `scripts/run_ladder.sh`. Fill
   `experiments/log_template.md` as runs complete.
4. Run the data-ladder ablation. This is the first thing a reviewer will
   challenge: is the gap about capacity, or just about too little data?
5. Run `src/token_shift.py` on each rung. The early-to-late KL ratio is the
   mechanistic story and does not depend on any benchmark.
6. Paste `analysis/table2.tex` into the paper and write the results section
   around what the numbers actually say.

## When results contradict the hypothesis we are proposing

Report them. A well-measured null on a claim this widely cited is a genuine
contribution, and the paper's introduction already commits in writing to
reporting that outcome with equal prominence. Do not tune the experiment until
it produces the expected curve.
