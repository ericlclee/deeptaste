#!/bin/bash
# One-time conda env bootstrap on PACE-ICE. Run from a LOGIN node:
#     bash scripts/setup_env.sh
#
# Install ORDER is load-bearing. sentence-transformers and transformers depend on
# torch, so if they go first pip fetches the default PyPI torch -- currently a
# CUDA 13 build, which fails on PACE's CUDA 12.9 driver with a misleading
# "driver is too old" warning. Installing torch first from a pinned CUDA index
# means the later installs see the requirement already satisfied and leave it be.
#
# CUDA_CHANNEL must be <= the driver's CUDA version. Check the driver with
# `nvidia-smi` on a GPU node; cu128 works on any 12.x driver from 12.8 up, and
# CUDA minor-version compatibility covers the rest of the 12.x line.
#     CUDA_CHANNEL=cu126 bash scripts/setup_env.sh   # for an older driver
set -euo pipefail

ENV_NAME="${ENV_NAME:-deeptaste}"
CUDA_CHANNEL="${CUDA_CHANNEL:-cu128}"

module load anaconda3

if conda env list | grep -qE "^${ENV_NAME}\s"; then
    echo "env '${ENV_NAME}' exists -- updating from environment.yml"
    conda env update -n "${ENV_NAME}" -f environment.yml --prune
else
    conda env create -n "${ENV_NAME}" -f environment.yml
fi

source activate "${ENV_NAME}"

echo "--- 1/2 torch (${CUDA_CHANNEL}) ---"
pip install --force-reinstall torch --index-url "https://download.pytorch.org/whl/${CUDA_CHANNEL}"

echo "--- 2/2 everything else ---"
# No torch here: it is already satisfied above and pip will not replace it.
pip install \
    "pandas==3.0.3" \
    "pyarrow==24.0.0" \
    "numpy==2.5.1" \
    "scikit-learn==1.9.0" \
    "sentence-transformers==5.6.0" \
    "transformers==5.14.0"

python - <<'PY'
import torch
print(f"\ntorch {torch.__version__}  built against CUDA {torch.version.cuda}")
if torch.version.cuda and int(torch.version.cuda.split(".")[0]) >= 13:
    raise SystemExit(
        "ERROR: a CUDA 13 torch build got installed. PACE's driver is 12.9.\n"
        "Re-run with an explicit channel, e.g. CUDA_CHANNEL=cu126."
    )
print("Login nodes have no GPU, so torch.cuda.is_available() is False here --")
print("that is expected. Verify on a GPU node with scripts/check_gpu.sbatch.")
PY

echo "done. environment: ${ENV_NAME}"
