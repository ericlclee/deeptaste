#!/bin/bash
# One-time conda env bootstrap on PACE-ICE. Run from a LOGIN node:
#     bash scripts/setup_env.sh
#
# Envs live in $HOME by default, which has a small quota. If you hit quota
# errors, point conda at scratch first:
#     conda config --add envs_dirs ~/scratch/conda/envs
set -euo pipefail

ENV_NAME="${ENV_NAME:-deep_taste}"

module load anaconda3

if conda env list | grep -qE "^${ENV_NAME}\s"; then
    echo "env '${ENV_NAME}' exists -- updating from environment.yml"
    conda env update -n "${ENV_NAME}" -f environment.yml --prune
else
    conda env create -n "${ENV_NAME}" -f environment.yml
fi

source activate "${ENV_NAME}"

# torch is deliberately NOT in environment.yml: the wheel has to match the CUDA
# version on the GPU nodes. cu121 wheels work on driver 525+; check with
# `nvidia-smi` on a GPU node and adjust the index-url if yours is older.
pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu121

python - <<'PY'
import torch
print(f"torch {torch.__version__}  cuda_build={torch.version.cuda}")
print("NOTE: torch.cuda.is_available() is False on login nodes -- that is expected.")
PY

echo "done. environment: ${ENV_NAME}"
