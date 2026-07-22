#!/bin/bash
#SBATCH --job-name=deep_taste
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:V100:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=./logs/%x_%j.out
#SBATCH --error=./logs/%x_%j.err
#
# PARTITION: not set above on purpose -- PACE-ICE partition names change between
# semesters and a wrong one fails at submit time with a confusing message. Check
# yours with `sinfo -s` on a login node, then either uncomment the line below or
# pass it at submit: sbatch -p <partition> scripts/train.sh
# #SBATCH --partition=ice-gpu
#
# Submit from the repo root:
#     mkdir -p logs
#     sbatch scripts/train.sh
#
# Override any hyperparameter via the environment:
#     EPOCHS=40 LR=1e-4 BATCH_SIZE=4096 sbatch scripts/train.sh
set -euo pipefail

ENV_NAME="${ENV_NAME:-deeptaste}"

EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
LR="${LR:-3e-4}"
CLIP="${CLIP:-1.0}"
MAX_HISTORY="${MAX_HISTORY:-50}"
OUTPUT_DIMS="${OUTPUT_DIMS:-128}"
EVAL_K="${EVAL_K:-10}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-512}"
HARD_NEG_RATIO="${HARD_NEG_RATIO:-0.0}"   # fraction of negatives from the pos's cuisine/price/geo cluster
HARD_NEG_K="${HARD_NEG_K:-30}"
EXTRA_ARGS="${EXTRA_ARGS:-}"   # e.g. EXTRA_ARGS=--eval-test

# Slurm starts the job in the submit directory; be explicit anyway so the job is
# not silently dependent on where it was launched from.
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

module load anaconda3
source activate "${ENV_NAME}"

# src/ is not a package -- train.py does flat imports (model, evaluate).
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
export DEEP_TASTE_DATA="${DEEP_TASTE_DATA:-${PWD}/data/processed}"
export TOKENIZERS_PARALLELISM=false

echo "=========================================================="
echo "job          : ${SLURM_JOB_ID:-local} on $(hostname)"
echo "data         : ${DEEP_TASTE_DATA}"
echo "hyperparams  : epochs=${EPOCHS} batch=${BATCH_SIZE} lr=${LR} clip=${CLIP}"
echo "               max_history=${MAX_HISTORY} dims=${OUTPUT_DIMS}"
echo "=========================================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "no GPU visible"
# Fail fast if torch cannot see the GPU. Without this the job silently runs on
# CPU (or with a CUDA-build mismatch, dies on the first real kernel launch deep
# in training) until the wall clock kills it.
python -c "
import sys, torch
if not torch.cuda.is_available():
    sys.exit('FATAL: no GPU visible to torch (built for CUDA %s). '
             'Run scripts/check_gpu.sbatch to diagnose.' % torch.version.cuda)
"

python src/train.py \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --clip "${CLIP}" \
    --max-history "${MAX_HISTORY}" \
    --output-dims "${OUTPUT_DIMS}" \
    --eval-k "${EVAL_K}" \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --hard-neg-ratio "${HARD_NEG_RATIO}" \
    --hard-neg-k "${HARD_NEG_K}" \
    ${EXTRA_ARGS}

echo "training done -- checkpoint written to ${DEEP_TASTE_DATA}/encoder.pt"