#!/bin/bash
# ---------------------------------------------------------------------------
# The default profile, one model per SLURM job, one L40S each.
#
# A job ARRAY rather than one job holding four GPUs, because that is what SLURM
# is good at: independent single-GPU jobs spread over ceratanium/netherite/
# saronite and start as cards free up, instead of queueing behind an
# all-or-nothing 4-GPU allocation. It also sidesteps syco's own GPU scheduler --
# each job runs one model on the single card SLURM gave it and never touches
# CUDA_VISIBLE_DEVICES.
#
#   sbatch slurm/run_model.sbatch                          # the whole profile
#   sbatch --array=1 slurm/run_model.sbatch                # one model
#   sbatch --export=ALL,PROFILE=structured-4dims slurm/run_model.sbatch
#
# Runs resume. If a job hits the 120h wall clock, resubmit the same array index
# and it picks up where it stopped.
# ---------------------------------------------------------------------------
#SBATCH --job-name=syco-run
#SBATCH --partition=L40s_staff
#SBATCH --array=0-4
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=120:00:00
#SBATCH --signal=INT@300
#SBATCH --output=/data/yazanm/SychopancyAssumptions/slurm/logs/%x-%A_%a.out
#SBATCH --error=/data/yazanm/SychopancyAssumptions/slurm/logs/%x-%A_%a.err

set -euo pipefail

REPO=/data/yazanm/SychopancyAssumptions
cd "$REPO"

PROFILE="${PROFILE:-default}"

# The GGUF weights and tokenizers are already cached here, on the shared /data
# filesystem the L40S nodes mount, so nothing is downloaded. /home is capped at
# 7.5GB and would not hold even one of these files.
export HF_HOME=/data/yazanm/huggingface_cache

# torch, OpenBLAS and numba each size their thread pool to every core on the
# node -- 64 here. Cap them at the allocation.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=false

source "$REPO/.venv/bin/activate"

# Model list and run arguments both come from the profile. Writing either into
# this file would let config/experiments/*.yaml and the job drift apart, which
# is precisely how a run ends up administering a design nobody chose.
mapfile -t MODELS < <(python slurm/profile_args.py --models "$PROFILE")
INDEX="${SLURM_ARRAY_TASK_ID:-0}"
if (( INDEX >= ${#MODELS[@]} )); then
    echo "array index $INDEX is out of range: profile '$PROFILE' has ${#MODELS[@]} model(s)" >&2
    echo "adjust --array in this file to 0-$(( ${#MODELS[@]} - 1 ))" >&2
    exit 2
fi
MODEL="${MODELS[$INDEX]}"

# NUL-separated, because `--system ''` is a legitimately empty argument.
#
# This is the step a bare `syco run --model X` skips. Without it, --n-personas
# and --n-prompts stay at their None defaults, _stable_sample takes every
# persona and every dilemma, and the job quietly administers the full ~4.4M-cell
# grid instead of the profile's design.
mapfile -t -d '' RUN_ARGS < <(python slurm/profile_args.py --args "$PROFILE" "$MODEL")
if (( ${#RUN_ARGS[@]} == 0 )); then
    echo "no run arguments resolved for $MODEL in profile $PROFILE" >&2
    exit 2
fi

echo "host=$(hostname)  profile=$PROFILE  model=$MODEL  gpu=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
printf 'resolved: python -m syco run'; printf ' %q' "${RUN_ARGS[@]}"; echo
echo

# --signal=INT@300 above delivers SIGINT to this step five minutes before the
# wall clock, which the runner treats as a single Ctrl-C: it finishes the batch
# in flight and flushes, rather than being killed mid-write.
srun python -m syco run "${RUN_ARGS[@]}"

echo
echo "finished $MODEL on $(hostname)"

# `syco status` returns 1 while any model in the profile is incomplete. That is
# expected when this array task finishes before its siblings and must not turn a
# successful model run into a FAILED Slurm task. Preserve exit codes >=2, which
# mean the status command itself could not run or validate the profile.
set +e
python -m syco status --profile "$PROFILE"
STATUS_RC=$?
set -e
if (( STATUS_RC > 1 )); then
    echo "profile status check failed with exit $STATUS_RC" >&2
    exit "$STATUS_RC"
fi
