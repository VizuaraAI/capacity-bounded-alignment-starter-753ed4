#!/usr/bin/env bash
# The main experiment: every rung of the capacity ladder x every intervention.
#
# This is the script that fills Table 2 of the paper. Budget 1 to 2 days of GPU
# time for the default ladder. Everything is checkpointed and skipped if already
# present, so it is safe to interrupt and rerun.
set -euo pipefail

MODELS="${MODELS:-smollm2-135m smollm2-360m qwen2.5-0.5b llama3.2-1b smollm2-1.7b}"
SEEDS="${SEEDS:-17}"
N_EVAL="${N_EVAL:-256}"
DATA_SIZES="${DATA_SIZES:-1000}"

for seed in $SEEDS; do
for model in $MODELS; do
    echo "################ ${model} (seed ${seed}) ################"

    for intervention in base ica; do
        out="results/${model}__${intervention}__seed${seed}.json"
        if [[ -f "$out" ]]; then
            echo "--- skip ${intervention}, ${out} exists"
        else
            python -m src.evaluate --model "$model" --intervention "$intervention" \
                --n-eval "$N_EVAL"
        fi
    done

    for n in $DATA_SIZES; do
        ckpt="runs/${model}__sft-lima-${n}__seed${seed}"
        if [[ -d "$ckpt" ]]; then
            echo "--- skip training, ${ckpt} exists"
        else
            python -m src.train_sft --model "$model" --data lima \
                --n-train "$n" --seed "$seed"
        fi

        out="results/${model}__sft__seed${seed}.json"
        if [[ -f "$out" ]]; then
            echo "--- skip sft eval, ${out} exists"
        else
            python -m src.evaluate --checkpoint "$ckpt" --model "$model" \
                --intervention sft --n-eval "$N_EVAL" --out "$out"
        fi

        # The mechanistic half of the story: where in the response does the
        # aligned model actually diverge from its own base model?
        tds="results/tds__${model}__seed${seed}.json"
        if [[ -f "$tds" ]]; then
            echo "--- skip token shift, ${tds} exists"
        else
            python -m src.token_shift --base-model "$model" --aligned "$ckpt"
        fi
    done
done
done

python -m src.analyze --results-dir results --out-dir analysis
echo
echo "Main grid complete. analysis/table2.tex can be pasted straight into the paper."
