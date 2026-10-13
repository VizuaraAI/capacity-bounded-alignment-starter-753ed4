#!/usr/bin/env bash
# One time environment setup. Run from the repository root.
set -euo pipefail

PY="${PY:-python3}"

echo "==> creating virtual environment in .venv"
"$PY" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> upgrading pip"
pip install --upgrade pip wheel

echo "==> installing requirements (this pulls torch, expect a few minutes)"
pip install -r requirements.txt

echo "==> checking GPU visibility"
python - <<'PY'
import torch
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {name} ({total:.0f} GB)")
else:
    print("No GPU found. The 135M and 360M rungs are still trainable on CPU, "
          "slowly. Everything above that needs a GPU or a Colab runtime.")
PY

echo "==> printing the capacity ladder"
python -m src.model_zoo

cat <<'EOF'

Two datasets are gated on the Hugging Face hub and must be accepted once in a
browser while logged in, then authenticated locally:

  GAIR/lima                      https://huggingface.co/datasets/GAIR/lima
  meta-llama/Llama-3.2-1B        https://huggingface.co/meta-llama/Llama-3.2-1B

  huggingface-cli login

Setup done. Next:
  bash scripts/run_baseline.sh      # smoke test, roughly 15 minutes on one GPU
EOF
