#!/bin/bash
#SBATCH --job-name=dt_sweep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --array=0-5%2
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#
# Hyperparameter sweep as a Slurm job array. Each array task runs one config and
# writes its own checkpoint, so runs never clobber each other.
#
#     mkdir -p logs
#     sbatch -p <partition> scripts/sweep.sbatch
#
# `%2` in --array caps concurrency at 2 jobs; raise it if your queue allows, and
# keep --array's upper bound equal to (number of CONFIGS - 1).
set -euo pipefail

# ---- the grid: one "lr|batch|dims|weight_decay-ish extra" per line ----------
# Ordered per the tuning notes: regularization and LR first, capacity after.
CONFIGS=(
    "3e-4 2048 128"
    "1e-4 2048 128"
    "3e-4 4096 128"
    "3e-4 2048 64"
    "3e-4 2048 256"
    "1e-3 2048 128"
)

IDX="${SLURM_ARRAY_TASK_ID:-0}"
read -r LR BATCH_SIZE OUTPUT_DIMS <<< "${CONFIGS[$IDX]}"

RUN_NAME="lr${LR}_bs${BATCH_SIZE}_d${OUTPUT_DIMS}"
RUN_DIR="runs/${SLURM_ARRAY_JOB_ID:-local}/${RUN_NAME}"

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p "${RUN_DIR}" logs

module load anaconda3
source activate "${ENV_NAME:-deep_taste}"

export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
export DEEP_TASTE_DATA="${DEEP_TASTE_DATA:-${PWD}/data/processed}"
export TOKENIZERS_PARALLELISM=false

echo "=========================================================="
echo "array task ${IDX}: ${RUN_NAME}"
echo "checkpoint : ${RUN_DIR}/encoder.pt"
echo "=========================================================="

srun python src/train.py \
    --epochs "${EPOCHS:-20}" \
    --batch-size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --output-dims "${OUTPUT_DIMS}" \
    --checkpoint "${RUN_DIR}/encoder.pt" \
    2>&1 | tee "${RUN_DIR}/train.log"

# Best val NDCG for this config, for easy cross-run comparison:
#     grep -H "best val" runs/<array_job_id>/*/train.log
grep "best val" "${RUN_DIR}/train.log" > "${RUN_DIR}/result.txt" || true
