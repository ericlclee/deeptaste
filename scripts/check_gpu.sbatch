#!/bin/bash
#SBATCH --job-name=dt_gpucheck
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Confirms torch can actually see the GPU. Run this after setup_env.sh and after
# any torch reinstall -- it costs 30 seconds and catches the CUDA-build mismatch
# that otherwise only shows up as a silent CPU fallback hitting the time limit.
#
#     mkdir -p logs && sbatch -p <partition> scripts/check_gpu.sbatch
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs
module load anaconda3
source activate "${ENV_NAME:-deep_taste}"

nvidia-smi

python - <<'PY'
import torch
print(f"torch            : {torch.__version__}")
print(f"built for CUDA   : {torch.version.cuda}")
print(f"cuda.is_available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit(
        "FAIL: torch cannot see the GPU.\n"
        "Almost always a CUDA-build mismatch: compare 'built for CUDA' above to\n"
        "the driver's CUDA version in the nvidia-smi header, then rebuild with\n"
        "  CUDA_CHANNEL=cu126 bash scripts/setup_env.sh"
    )
print(f"device           : {torch.cuda.get_device_name(0)}")
free, total = torch.cuda.mem_get_info()
print(f"memory           : {free/1e9:.1f} GB free / {total/1e9:.1f} GB total")
x = torch.randn(4096, 4096, device="cuda")
print(f"matmul check     : {(x @ x).sum().item():.3e}  OK")
PY
