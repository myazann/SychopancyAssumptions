#!/bin/bash
# One-GPU Hugging Face target stage. Extraction and steering are separate jobs;
# never point two concurrent writers at the same stage digest.
#
#   sbatch --export=ALL,PROBE_STAGE=extract slurm/linear_probe_target.sbatch
#   # run `linear-probe train` after extraction completes
#   sbatch --export=ALL,PROBE_STAGE=steer slurm/linear_probe_target.sbatch

#SBATCH --job-name=syco-probe-target
#SBATCH --partition=L40s_staff
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --signal=INT@300
#SBATCH --output=/data/yazanm/SychopancyAssumptions/slurm/logs/%x-%j.out
#SBATCH --error=/data/yazanm/SychopancyAssumptions/slurm/logs/%x-%j.err

set -euo pipefail

REPO=/data/yazanm/SychopancyAssumptions
PROBE_CONFIG="${PROBE_CONFIG:-config/linear_probe.yaml}"
PROBE_STAGE="${PROBE_STAGE:?Set PROBE_STAGE to extract or steer}"

case "$PROBE_STAGE" in
    extract|steer) ;;
    *)
        echo "PROBE_STAGE must be extract or steer" >&2
        exit 2
        ;;
esac

cd "$REPO"
export HF_HOME=/data/yazanm/huggingface_cache
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=false
source "$REPO/.venv/bin/activate"

echo "host=$(hostname) stage=$PROBE_STAGE"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
srun python -m syco linear-probe "$PROBE_STAGE" --config "$PROBE_CONFIG"
