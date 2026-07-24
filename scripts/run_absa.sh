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
# PACE-ICE GPU node. LIMIT caps this job's ENTIRE review count -- there is
# no separate always-runs-the-full-job step, so a smoke test and the real
# run are two separate submissions, not one:
#
#     mkdir -p logs
#     sbatch -p ice-gpu --export=LIMIT=20 scripts/run_absa.sh   # smoke test
#     sbatch -p ice-gpu scripts/run_absa.sh                     # full run (no LIMIT)
#
# (partition name changes between semesters -- check yours with `sinfo -s`)
#
# The underlying script checkpoints after each aspect (food/service/price/
# ambience) to data/processed/absa_scores.pt, so if this job hits the
# --time limit mid-run, just resubmit the exact same command -- it picks
# up from whichever aspects already finished instead of starting over.
set -euo pipefail

ENV_NAME="${ENV_NAME:-deeptaste}"
LIMIT="${LIMIT:-}"

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

if [ -n "${LIMIT}" ]; then
    # Smoke tests write to their own file -- never the real absa_scores.pt --
    # so re-running a smoke test can't clobber real checkpointed progress.
    OUTPUT="${DEEP_TASTE_DATA}/absa_scores_smoke.pt"
    echo "--- scoring first ${LIMIT} reviews (smoke test) -> ${OUTPUT} ---"
    python -u src/absa_tag_reviews.py --limit "${LIMIT}" --output "${OUTPUT}"
else
    OUTPUT="${DEEP_TASTE_DATA}/absa_scores.pt"
    echo "--- scoring all reviews (full run) -> ${OUTPUT} ---"
    python -u src/absa_tag_reviews.py --output "${OUTPUT}"
fi

echo "wrote ${OUTPUT}"
