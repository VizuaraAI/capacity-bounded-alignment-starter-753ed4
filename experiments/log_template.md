# Experiment log

One row per run. The row structure matches the paper's tables on purpose: a
filled log is most of a results section.

Copy a table block, do not edit a previous row in place. If a number changes,
add a new row and note why in the Notes column. Past-you being wrong in public
is how you catch a bug three weeks later.

---

## Table 1: the main grid (fills Table 2 of the paper)

Columns map directly onto `analysis/superficiality.csv`. `P` columns are mean
scores; `S` is the Superficiality Index with its 95% bootstrap interval.

### Format axis

| Date | Model | Params | Seed | n_eval | P_base | P_ica | P_sft | S [95% CI] | Notes |
|------|-------|--------|------|--------|--------|-------|-------|------------|-------|
|      | SmolLM2-135M | 135M | 17 |  |  |  |  |  |  |
|      | SmolLM2-360M | 360M | 17 |  |  |  |  |  |  |
|      | Qwen2.5-0.5B | 0.5B | 17 |  |  |  |  |  |  |
|      | Llama-3.2-1B | 1B | 17 |  |  |  |  |  |  |
|      | SmolLM2-1.7B | 1.7B | 17 |  |  |  |  |  |  |

### Capability axis

| Date | Model | Params | Seed | n_eval | P_base | P_ica | P_sft | S [95% CI] | Notes |
|------|-------|--------|------|--------|--------|-------|-------|------------|-------|
|      | SmolLM2-135M | 135M | 17 |  |  |  |  |  |  |
|      | SmolLM2-360M | 360M | 17 |  |  |  |  |  |  |
|      | Qwen2.5-0.5B | 0.5B | 17 |  |  |  |  |  |  |
|      | Llama-3.2-1B | 1B | 17 |  |  |  |  |  |  |
|      | SmolLM2-1.7B | 1.7B | 17 |  |  |  |  |  |  |

**Headline statistic:** slope of capability-axis `S` against log10(params)
= ______ (R² = ____), from `analysis/elasticity.json`.

### Controlled ladder (Pythia: identical data, identical order)

| Date | Model | Axis | P_base | P_ica | P_sft | S [95% CI] | Notes |
|------|-------|------|--------|-------|-------|------------|-------|
|      | Pythia-160M | format |  |  |  |  |  |
|      | Pythia-160M | capability |  |  |  |  |  |
|      | Pythia-410M | format |  |  |  |  |  |
|      | Pythia-410M | capability |  |  |  |  |  |
|      | Pythia-1.4B | format |  |  |  |  |  |
|      | Pythia-1.4B | capability |  |  |  |  |  |

---

## Table 2: Ablation A, the data ladder (paper section 5, bullet 1)

The question: is the capability gap about capacity, or just about too little
data? Anchored at 360M.

| Date | Data | n_train | Format score | GSM8K | MMLU | ARC-C | Notes |
|------|------|---------|--------------|-------|------|-------|-------|
|      | none (base) | 0 |  |  |  |  |  |
|      | LIMA | 1,000 |  |  |  |  |  |
|      | Tulu 3 | 10,000 |  |  |  |  |  |
|      | Tulu 3 | 50,000 |  |  |  |  |  |
|      | Tulu 3 | 200,000 |  |  |  |  |  |
|      | continued pretrain, then LIMA | 1,000 |  |  |  |  |  |

**Verdict:** does the capability column saturate below the ceiling? ________

---

## Table 3: Ablation B, update capacity (LoRA rank vs full fine-tuning)

Under a pure selection account, update capacity should barely matter.

| Date | Model | Method | Rank | Trainable params | Format score | GSM8K | MMLU | Notes |
|------|-------|--------|------|------------------|--------------|-------|------|-------|
|      | SmolLM2-360M | LoRA | 4 |  |  |  |  |  |
|      | SmolLM2-360M | LoRA | 16 |  |  |  |  |  |
|      | SmolLM2-360M | LoRA | 64 |  |  |  |  |  |
|      | SmolLM2-360M | full FT | n/a |  |  |  |  |  |

---

## Table 4: Ablation C, demonstration count for in-context alignment

Forecloses the objection that the zero-update baseline was starved.

| Date | Model | n_demos | Format score | GSM8K | MMLU | Notes |
|------|-------|---------|--------------|-------|------|-------|
|      | SmolLM2-360M | 1 |  |  |  |  |
|      | SmolLM2-360M | 3 |  |  |  |  |
|      | SmolLM2-360M | 8 |  |  |  |  |

---

## Table 5: Token distribution shift (paper section 4, final bullet)

From `results/tds__*.json`. `early/late ratio` well above 1 means the shift is
front-loaded and stylistic, which is what the hypothesis predicts. Near 1 means
tuning rewrote behaviour throughout the response.

| Date | Model | Params | Shifted fraction | Mean KL | KL early | KL late | Early/late ratio | Notes |
|------|-------|--------|------------------|---------|----------|---------|------------------|-------|
|      | SmolLM2-135M | 135M |  |  |  |  |  |  |
|      | SmolLM2-360M | 360M |  |  |  |  |  |  |
|      | Qwen2.5-0.5B | 0.5B |  |  |  |  |  |  |
|      | Llama-3.2-1B | 1B |  |  |  |  |  |  |
|      | SmolLM2-1.7B | 1.7B |  |  |  |  |  |  |

---

## Table 6: Seed variance at the smallest rung

Small models are noisy. You cannot claim a gap narrower than this spread.

| Date | Model | Seed | Format S | Capability S | Notes |
|------|-------|------|----------|--------------|-------|
|      | SmolLM2-135M | 17 |  |  |  |
|      | SmolLM2-135M | 18 |  |  |  |
|      | SmolLM2-135M | 19 |  |  |  |

**Spread (max - min) on the capability axis:** ________

---

## Running decisions log

Dated notes on choices that a reviewer might question, and why you made them.
Write these down the day you decide, not the week before the deadline.

| Date | Decision | Why | Alternative rejected |
|------|----------|-----|----------------------|
|      |  |  |  |

---

## Things that broke, and what they turned out to be

Keep this. Half of it becomes the limitations paragraph.

| Date | Symptom | Root cause | Fix |
|------|---------|------------|-----|
|      |  |  |  |
