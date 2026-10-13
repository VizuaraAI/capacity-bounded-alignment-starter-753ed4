#!/usr/bin/env bash
# Ablations, all anchored at the 360M rung so they stay cheap.
#
# Each block answers one reviewer question. Run them in the order given: the
# data ladder is the one a reviewer will ask about first.
set -euo pipefail

MODEL="${MODEL:-smollm2-360m}"
SEED="${SEED:-17}"
N_EVAL="${N_EVAL:-256}"

echo "######## A. Data ladder: is it capacity, or just too little data? ########"
# If scaling the SFT set closes the capability gap, the story is about data.
# If it saturates well below the ceiling, the story is about capacity.
for n in 1000 10000 50000; do
    ckpt="runs/${MODEL}__sft-tulu3-${n}__seed${SEED}"
    [[ -d "$ckpt" ]] || python -m src.train_sft --model "$MODEL" --data tulu3 \
        --n-train "$n" --seed "$SEED"
    python -m src.evaluate --checkpoint "$ckpt" --model "$MODEL" \
        --intervention sft --n-eval "$N_EVAL" \
        --out "results/ablation_data/${MODEL}__sft-tulu3-${n}__seed${SEED}.json"
done

echo "######## B. Update capacity: LoRA rank sweep against full fine tuning ########"
# Full fine tuning has far more update capacity than a rank 4 adapter. If the
# capability gain tracks rank, the gain is being written into weights rather
# than selected from what pretraining already installed.
for r in 4 16 64; do
    ckpt="runs/${MODEL}__sft-lima-1000-lora${r}__seed${SEED}"
    [[ -d "$ckpt" ]] || python -m src.train_sft --model "$MODEL" --data lima \
        --n-train 1000 --lora-r "$r" --seed "$SEED"
    python -m src.evaluate --checkpoint "$ckpt" --model "$MODEL" \
        --intervention sft --n-eval "$N_EVAL" \
        --out "results/ablation_lora/${MODEL}__lora${r}__seed${SEED}.json"
done

echo "######## C. Demonstration count for in context alignment ########"
# Guards against the obvious objection that the in-context baseline was simply
# starved of demonstrations.
for k in 1 3 8; do
    python -m src.evaluate --model "$MODEL" --intervention ica --n-eval "$N_EVAL" \
        --out "results/ablation_ica/${MODEL}__ica${k}__seed${SEED}.json"
done

echo "######## D. Seed variance at the smallest rung ########"
# Small models are noisy. Three seeds is the minimum that lets you claim a gap.
for s in 17 18 19; do
    ckpt="runs/smollm2-135m__sft-lima-1000__seed${s}"
    [[ -d "$ckpt" ]] || python -m src.train_sft --model smollm2-135m --data lima \
        --n-train 1000 --seed "$s"
    python -m src.evaluate --checkpoint "$ckpt" --model smollm2-135m \
        --intervention sft --n-eval "$N_EVAL" \
        --out "results/ablation_seed/smollm2-135m__sft__seed${s}.json"
done

cat <<'EOF'

Ablations done. Fill in experiments/log_template.md as you go: the rows there
map one to one onto the paper's tables, so a filled log is most of a results
section.
EOF
