#!/bin/bash
# ---------------------------------------------------------------------------
# Alternative to run_model.sbatch: one job that takes a whole L40S node and
# lets syco's own scheduler (`syco run --all`) drive all four cards.
#
# Use this when you want the profile lock, the single run_all.log, and the
# largest-first packing in one place. Prefer run_model.sbatch otherwise -- it
# queues better and finishes sooner.
#
# --gres=gpu:4 is deliberate and not negotiable here. syco/orchestrate.py picks
# a GPU by its `nvidia-smi --query-gpu=index` number and passes that number to
# the child as CUDA_VISIBLE_DEVICES. That is only sound when the job holds
# every GPU on the node; with a partial allocation those indices can be
# renumbered relative to what SLURM handed out and a child can address a card
# belonging to another job. Ask for all four.
#
#   sbatch slurm/run_all.sbatch
# ---------------------------------------------------------------------------
#SBATCH --job-name=syco-run-all
#SBATCH --partition=L40s_staff
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=120:00:00
#SBATCH --signal=INT@300
#SBATCH --output=/data/yazanm/SychopancyAssumptions/slurm/logs/%x-%j.out
#SBATCH --error=/data/yazanm/SychopancyAssumptions/slurm/logs/%x-%j.err

set -euo pipefail

REPO=/data/yazanm/SychopancyAssumptions
cd "$REPO"

export HF_HOME=/data/yazanm/huggingface_cache
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

source "$REPO/.venv/bin/activate"

echo "host=$(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
echo

# Five models over four cards: the scheduler starts four and holds the fifth
# until one finishes, so this job's wall clock is roughly two models deep.
srun python -m syco run --all

echo
python -m syco status
