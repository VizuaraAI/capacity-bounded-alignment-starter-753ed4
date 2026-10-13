#!/usr/bin/env bash
# Day one smoke test: the smallest end to end slice of the experiment.
#
# It trains a 1K-example LIMA-style run on the 360M rung and evaluates all three
# interventions needed to produce a single Superficiality Index value. On one
# consumer GPU this takes roughly 15 to 25 minutes. If this script finishes and
# analysis/superficiality.csv has one real number in it, the whole pipeline
# works and everything after it is a sweep.
set -euo pipefail

MODEL="${MODEL:-smollm2-360m}"
SEED="${SEED:-17}"
N_EVAL="${N_EVAL:-128}"
CKPT="runs/${MODEL}__sft-lima-1000__seed${SEED}"

echo "=== 1/4  base model, no weight update ==="
python -m src.evaluate --model "$MODEL" --intervention base --n-eval "$N_EVAL"

echo "=== 2/4  in context alignment, 3 demonstrations, still no weight update ==="
python -m src.evaluate --model "$MODEL" --intervention ica --n-eval "$N_EVAL"

echo "=== 3/4  minimal SFT, 1000 LIMA examples ==="
python -m src.train_sft --model "$MODEL" --data lima --n-train 1000 --seed "$SEED"

echo "=== 4/4  evaluate the tuned checkpoint ==="
python -m src.evaluate --checkpoint "$CKPT" --model "$MODEL" \
    --intervention sft --n-eval "$N_EVAL"

echo "=== analysis ==="
python -m src.analyze --results-dir results --out-dir analysis

cat <<'EOF'

Read analysis/superficiality.csv now. The question to ask of it:

  is the format-axis S much closer to 1 than the capability-axis S?

If yes, the effect this project is about is already visible at 360M and the
rest of the work is establishing it across the ladder with error bars. If the
two axes look the same, check first that the base model is being prompted
fairly (src/prompts.py) before concluding anything about the hypothesis.
EOF
