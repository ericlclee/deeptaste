#!/bin/bash
#SBATCH --job-name=dt_absa
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:V100:1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Run the ABSA review-scoring pipeline (src/absa_tag_reviews.py) on a
# PACE-ICE GPU node. Runs a small smoke test first and aborts before the
# full job if that fails, since a bad environment should surface in
# seconds, not after an hour into the real run.
#
# Submit from the repo root, passing the partition explicitly (name changes
# between semesters -- check yours with `sinfo -s`):
#
#     mkdir -p logs
#     sbatch -p ice-gpu scripts/run_absa.sh
#
# Override via the environment, e.g.:
#     sbatch -p ice-gpu --export=LIMIT=100 scripts/run_absa.sh
set -euo pipefail

ENV_NAME="${ENV_NAME:-deeptaste}"
LIMIT="${LIMIT:-20}"

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

module load anaconda3
source activate "${ENV_NAME}"

export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
export DEEP_TASTE_DATA="${DEEP_TASTE_DATA:-${PWD}/data/processed}"
export TOKENIZERS_PARALLELISM=false
# Keep model downloads off $HOME, which has a small quota on PACE.
export HF_HOME="${HF_HOME:-${HOME}/scratch/hf_cache}"
mkdir -p "${HF_HOME}"

echo "=========================================================="
echo "job     : ${SLURM_JOB_ID:-local} on $(hostname)"
echo "data    : ${DEEP_TASTE_DATA}"
echo "hf_cache: ${HF_HOME}"
echo "=========================================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "no GPU visible"
# Fail fast if torch cannot see the GPU, same check as scripts/features.sbatch --
# without it a bad CUDA build silently runs on CPU and eats the time limit.
python -c "
import sys, torch
if not torch.cuda.is_available():
    sys.exit('FATAL: no GPU visible to torch (built for CUDA %s). '
             'Run scripts/check_gpu.sbatch to diagnose.' % torch.version.cuda)
"

echo "--- smoke test (${LIMIT} reviews) ---"
SMOKE_OUT="${DEEP_TASTE_DATA}/absa_scores_smoke.pt"
python -u src/absa_tag_reviews.py --limit "${LIMIT}" --output "${SMOKE_OUT}"
rm -f "${SMOKE_OUT}"

echo "--- smoke test passed, running full job ---"
python -u src/absa_tag_reviews.py

echo "absa_scores.pt written to ${DEEP_TASTE_DATA}"
