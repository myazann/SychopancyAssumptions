#!/bin/bash
# Four deterministic, non-overlapping Qwen label shards. Freeze and audit the
# production config once before submitting this array.
#
#   python -m syco linear-probe freeze --config config/linear_probe.yaml
#   sbatch slurm/linear_probe_labels.sbatch
#   sbatch --array=2 slurm/linear_probe_labels.sbatch  # resume one fixed shard

#SBATCH --job-name=syco-probe-label
#SBATCH --partition=L40s_staff
#SBATCH --array=0-3%4
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
# Keep this fixed even when resubmitting one array index. In particular, do not
# derive it from SLURM_ARRAY_TASK_COUNT, which becomes 1 for --array=2.
LABEL_SHARDS=4

cd "$REPO"
export HF_HOME=/data/yazanm/huggingface_cache
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=false
source "$REPO/.venv/bin/activate"

# Prevent four array jobs from racing to create the frozen parquet snapshot.
PROBE_CONFIG="$PROBE_CONFIG" python - <<'PY'
import os
from syco.linear_probe.artifacts import paths_for, require_manifest
from syco.linear_probe.config import load_config

config = load_config(os.environ["PROBE_CONFIG"])
artifacts = paths_for(config)
require_manifest(artifacts.dataset_manifest, config, "dataset")
print("frozen dataset:", artifacts.dataset_manifest)
PY

echo "host=$(hostname) shard=${SLURM_ARRAY_TASK_ID}/${LABEL_SHARDS}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader

srun python -m syco linear-probe label \
    --config "$PROBE_CONFIG" \
    --num-shards "$LABEL_SHARDS" \
    --shard-index "$SLURM_ARRAY_TASK_ID"

