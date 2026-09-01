#!/bin/bash
# Two provenance-separated teacher pools. Qwen uses four shards; the slower
# dense Gemma-27B uses ten. Every teacher labels every cell and instrument.
# Freeze and audit the production config once before submitting the full array.
#
#   python -m syco linear-probe freeze --config config/linear_probe.yaml
#   sbatch --array=0,4 --export=ALL,LABEL_PILOT=1,LABEL_LIMIT=200 \
#       slurm/linear_probe_labels.sbatch  # one 200-call pilot per teacher
#   sbatch slurm/linear_probe_labels.sbatch
#   sbatch --array=7 slurm/linear_probe_labels.sbatch  # resume one fixed task

#SBATCH --job-name=syco-probe-label
#SBATCH --partition=L40s_staff
#SBATCH --array=0-13%8
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --signal=INT@300
#SBATCH --output=/data/yazanm/SychopancyAssumptions/slurm/logs/%x-%A_%a.out
#SBATCH --error=/data/yazanm/SychopancyAssumptions/slurm/logs/%x-%A_%a.err

set -euo pipefail

REPO=/data/yazanm/SychopancyAssumptions
PROBE_CONFIG="${PROBE_CONFIG:-config/linear_probe.yaml}"
# Keep this mapping fixed after production starts. Tasks 0-3 run Qwen shards;
# tasks 4-13 run Gemma shards. Do not derive counts from SLURM_ARRAY_TASK_COUNT,
# which changes when a single task is resumed.
if (( SLURM_ARRAY_TASK_ID < 4 )); then
    LABEL_TEACHER=qwen36_35b_a3b
    LABEL_SHARDS=4
    LABEL_SHARD_INDEX=$SLURM_ARRAY_TASK_ID
else
    LABEL_TEACHER=gemma3_27b
    LABEL_SHARDS=10
    LABEL_SHARD_INDEX=$((SLURM_ARRAY_TASK_ID - 4))
fi

# Pilot tasks deliberately run each teacher as a one-shard queue. This makes
# their --limit prefixes identical, so quality.json can compare the same 200
# tasks across teachers. The full array sees those task keys as already done
# and resumes around them. Reject other array indices to prevent concurrent
# writes to the same pilot file.
if [[ "${LABEL_PILOT:-0}" == 1 ]]; then
    if (( SLURM_ARRAY_TASK_ID != 0 && SLURM_ARRAY_TASK_ID != 4 )); then
        echo "LABEL_PILOT=1 is valid only for array tasks 0 and 4" >&2
        exit 2
    fi
    LABEL_SHARDS=1
    LABEL_SHARD_INDEX=0
fi

cd "$REPO"
export HF_HOME=/data/yazanm/huggingface_cache
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=false
source "$REPO/.venv/bin/activate"

# Prevent array jobs from racing to create the frozen parquet snapshot.
PROBE_CONFIG="$PROBE_CONFIG" python - <<'PY'
import os
from syco.linear_probe.artifacts import paths_for, require_manifest
from syco.linear_probe.config import load_config

config = load_config(os.environ["PROBE_CONFIG"])
artifacts = paths_for(config)
require_manifest(artifacts.dataset_manifest, config, "dataset")
print("frozen dataset:", artifacts.dataset_manifest)
PY

echo "host=$(hostname) teacher=$LABEL_TEACHER shard=$LABEL_SHARD_INDEX/$LABEL_SHARDS"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader

LABEL_ARGS=(
    --config "$PROBE_CONFIG"
    --teacher "$LABEL_TEACHER"
    --num-shards "$LABEL_SHARDS"
    --shard-index "$LABEL_SHARD_INDEX"
)
if [[ -n "${LABEL_LIMIT:-}" ]]; then
    LABEL_ARGS+=(--limit "$LABEL_LIMIT")
fi

srun python -m syco linear-probe label "${LABEL_ARGS[@]}"
