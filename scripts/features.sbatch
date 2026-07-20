#!/bin/bash
#SBATCH --job-name=dt_features
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Encodes ~397k review texts with a sentence-transformer and pools them with
# exponential recency weights -> features.pt, norm_stats.json, review_vecs.pt.
# Ported from the Colab notebook; ~15-25 min on a T4, less on newer cards.
#
# Run this BEFORE training, and re-run whenever the encoder or pooling changes.
# Training reads features.pt; it does not regenerate it.
#
#     mkdir -p logs
#     sbatch -p <partition> scripts/features.sbatch
#
# Partition is intentionally unset -- see the note in scripts/train.sbatch.
# #SBATCH --partition=ice-gpu
#
# Overrides:
#     TAU=1.5 BATCH_SIZE=256 sbatch -p <partition> scripts/features.sbatch
#     DEEP_TASTE_SBERT=BAAI/bge-base-en-v1.5 sbatch -p <partition> scripts/features.sbatch
#     FP16=0 sbatch -p <partition> scripts/features.sbatch    # disable half precision
set -euo pipefail

ENV_NAME="${ENV_NAME:-deep_taste}"

TAU="${TAU:-2.0}"                  # recency half-life in years
BATCH_SIZE="${BATCH_SIZE:-128}"    # 128 fits a T4 at 512 tokens; raise on bigger cards
FP16="${FP16:-1}"                  # ~2x throughput on CUDA

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

module load anaconda3
source activate "${ENV_NAME}"

export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
export DEEP_TASTE_DATA="${DEEP_TASTE_DATA:-${PWD}/data/processed}"
export TOKENIZERS_PARALLELISM=false
# Keep model downloads off $HOME, which has a small quota on PACE.
export HF_HOME="${HF_HOME:-${HOME}/scratch/hf_cache}"
mkdir -p "${HF_HOME}"

FP16_FLAG=""
[ "${FP16}" = "1" ] && FP16_FLAG="--fp16"

echo "=========================================================="
echo "job     : ${SLURM_JOB_ID:-local} on $(hostname)"
echo "data    : ${DEEP_TASTE_DATA}"
echo "model   : ${DEEP_TASTE_SBERT:-thenlper/gte-base (default)}"
echo "params  : tau=${TAU} batch=${BATCH_SIZE} fp16=${FP16}"
echo "hf_cache: ${HF_HOME}"
echo "=========================================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "no GPU visible"

srun python -u src/features.py \
    --data-dir "${DEEP_TASTE_DATA}" \
    --tau "${TAU}" \
    --batch-size "${BATCH_SIZE}" \
    ${FP16_FLAG}

# Sanity-check the artifact rather than trusting exit code 0 (was cell 6 of the
# Colab notebook). A silently-corrupt features.pt would otherwise only surface
# as bad training results hours later.
python - <<'PY'
import os, torch
from pathlib import Path
out = Path(os.environ["DEEP_TASTE_DATA"])
d = torch.load(out / "features.pt", weights_only=False)
for k, v in d.items():
    if torch.is_tensor(v):
        print(f"{k:12} {str(tuple(v.shape)):16} {v.dtype}")
emb = d["text_emb"]
assert emb.ndim == 2 and emb.shape[1] > 0, f"bad text_emb shape {tuple(emb.shape)}"
assert not torch.isnan(emb).any(), "NaNs in text_emb"
assert not torch.isinf(emb).any(), "Infs in text_emb"
print(f"\nOK -- {emb.shape[0]:,} restaurants x {emb.shape[1]}d")
PY

echo "features written to ${DEEP_TASTE_DATA}"
