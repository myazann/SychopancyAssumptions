#!/bin/bash
# Queue the fully crossed 40 x 40 structured extension without racing a base
# writer. Slurm's aftercorr releases extension array element N only after base
# array element N succeeds. Already-finished base elements are immediately
# eligible; still-running elements remain pending.
#
# Current base arrays (override from the environment if these are resubmitted):
#   BASE_4DIMS_JOB=19779 BASE_SUPPORTTYPES_JOB=19784 \
#       bash slurm/submit_structured_extension.sh

set -euo pipefail

REPO=/data/yazanm/SychopancyAssumptions
cd "$REPO"

BASE_4DIMS_JOB="${BASE_4DIMS_JOB:-19779}"
BASE_SUPPORTTYPES_JOB="${BASE_SUPPORTTYPES_JOB:-19784}"
PYTHON="$REPO/.venv/bin/python"

# Submission only expands YAML profiles. Prevent numpy/OpenBLAS imports in that
# metadata step from trying to create one thread per login-node CPU.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

PROFILES=(
    structured-4dims-extension-40x40
    structured-supporttypes-extension-40x40
)
BASE_JOBS=("$BASE_4DIMS_JOB" "$BASE_SUPPORTTYPES_JOB")

for POSITION in "${!PROFILES[@]}"; do
    PROFILE_NAME="${PROFILES[$POSITION]}"
    BASE_JOB="${BASE_JOBS[$POSITION]}"
    mapfile -t MODELS < <("$PYTHON" slurm/profile_args.py --models "$PROFILE_NAME")
    if (( ${#MODELS[@]} == 0 )); then
        echo "profile '$PROFILE_NAME' selects no models" >&2
        exit 2
    fi

    LAST_INDEX=$(( ${#MODELS[@]} - 1 ))
    SHORT_NAME="${PROFILE_NAME#structured-}"
    JOB_ID=$(sbatch \
        --parsable \
        --dependency="aftercorr:$BASE_JOB" \
        --job-name="syco-$SHORT_NAME" \
        --array="0-$LAST_INDEX" \
        --export="ALL,PROFILE=$PROFILE_NAME" \
        slurm/run_model.sbatch)

    echo "submitted $PROFILE_NAME: job $JOB_ID; aftercorr:$BASE_JOB"
done
