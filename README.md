# Capacity-Bounded Alignment

**Does the Superficial Alignment Hypothesis hold when capacity is scarce?**

Starter repository for Vishnuvardhan, Vizuara SLM Bootcamp.

## The question

The Superficial Alignment Hypothesis, from the LIMA paper, says that a model's
knowledge and capability are installed almost entirely during pretraining, and
that instruction tuning only teaches the model which *format* to answer in. It
is one of the most quoted claims in post-training.

It also contains a hidden premise. "Tuning only selects what pretraining
installed" is only a meaningful statement if the capability was installed in
the first place. That premise is precisely what fails below one billion
parameters, where there is not enough capacity to have stored very much.

So the hypothesis may not be false. It may be **scale-bounded**, and nobody has
drawn the boundary. That is your project.

## The one idea the whole repository is built around

Split evaluation into two axes that are normally scored together:

- **Format axis**: did the model obey the instruction, does it look like an
  assistant? Measured with rules that a program can check, no judge model.
- **Capability axis**: is the answer actually right? Measured by comparing
  log probabilities of candidate answers, so a model that cannot produce tidy
  output is not also recorded as not knowing the answer.

Then measure, for each axis, how much of fine-tuning's benefit you could have
had for free by just writing a better prompt:

```
S = (P_in-context - P_base) / (P_fine-tuned - P_base)
```

`S` near 1 means alignment was superficial on that axis. `S` well below 1 means
fine-tuning installed something prompting could not reach.

**The expected finding:** `S` stays near 1 on the format axis at every model
size, and collapses on the capability axis as models get smaller. The
hypothesis holds for style, and breaks for substance, once capacity binds.

## Setup

```bash
git clone <your-repo-url> capacity-bounded-alignment
cd capacity-bounded-alignment
bash scripts/setup.sh
source .venv/bin/activate
```

Two things are gated on the Hugging Face hub and need a one-time click plus
`huggingface-cli login`:

- [`GAIR/lima`](https://huggingface.co/datasets/GAIR/lima), the 1,000 curated
  examples from the original paper
- [`meta-llama/Llama-3.2-1B`](https://huggingface.co/meta-llama/Llama-3.2-1B)

If you would rather not deal with gated repositories on day one, skip both:
the SmolLM2 and Qwen2.5 rungs are open, and `--data alpaca` works as a stand-in
for LIMA while you wait for access.

## How to run

**First, prove the pipeline works end to end** (roughly 15 to 25 minutes on one
GPU, longer on CPU):

```bash
bash scripts/run_baseline.sh
```

This evaluates SmolLM2-360M under three interventions (base, in-context,
1K-example SFT) and writes `analysis/superficiality.csv`. If the format-axis
`S` in that file is much closer to 1 than the capability-axis `S`, you have
already seen the effect the paper is about, at one model size.

**Then run the full grid** (budget one to two days of GPU time):

```bash
bash scripts/run_ladder.sh      # every rung x every intervention, fills Table 2
bash scripts/run_ablations.sh   # the four ablations reviewers will ask for
```

Both scripts skip work that is already on disk, so they are safe to interrupt
and rerun.

**Individual commands:**

```bash
python -m src.model_zoo                                    # print the capacity ladder
python -m src.evaluate --model smollm2-360m --intervention base
python -m src.evaluate --model smollm2-360m --intervention ica
python -m src.train_sft --model smollm2-360m --data lima --n-train 1000
python -m src.evaluate --checkpoint runs/<ckpt> --model smollm2-360m --intervention sft
python -m src.token_shift --base-model smollm2-360m --aligned runs/<ckpt>
python -m src.analyze --results-dir results --out-dir analysis
```

Add `--smoke-test` to any training command for a 64-example run that finishes
in a couple of minutes, which is the fastest way to check that a change did not
break anything.

## Layout

```
configs/default.yaml        every knob, one file; override from the CLI
src/model_zoo.py            the capacity ladder, defined once
src/data.py                 dataset loaders, normalised to two fixed shapes
src/prompts.py              base few-shot, in-context alignment, chat template
src/train_sft.py            SFT entry point, optional LoRA
src/evaluate.py             two-axis scoring, writes per-example scores
src/token_shift.py          KL between a base model and its aligned twin
src/metrics.py              the Superficiality Index and its bootstrap CIs
src/analyze.py              results directory -> CSV, LaTeX table, plot
scripts/                    setup, smoke test, main grid, ablations
experiments/log_template.md one row per run, mapped to the paper's tables
```

`src/analyze.py` writes `analysis/table2.tex`, which you paste directly over
the placeholder rows in the paper draft. The experiment log and the paper's
results table have deliberately identical row structure.

## A note on getting the baseline right

The single easiest way to produce a wrong headline result here is to prompt the
base model badly. If the untuned model is given no demonstrations and no
formatting cues, it will look incapable for reasons that have nothing to do
with capacity, every gap you report will be inflated, and a reviewer will spot
it immediately.

`src/prompts.py` exists to prevent this. Before you trust any number, read 20
raw base-model generations with your own eyes. It is the highest-value hour in
the whole project.

## Where this is going

The paper draft in `../paper/` is already written around these results, with
`XX.X` where your numbers go. Target venue is the EACL 2027 Student Research
Workshop (deadline 15 December 2026), with arXiv as the same-day floor.
