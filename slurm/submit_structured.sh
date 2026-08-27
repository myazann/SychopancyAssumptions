#!/bin/bash
# Submit the two structured-assumption instruments as independent model arrays.
#
# Together they measure all nine structured dimensions from the paper:
#   structured-4dims       4 belief dimensions
#   structured-supporttypes 5 support-seeking dimensions
#
# The arrays are independent, so Slurm can run them concurrently whenever GPUs
# are available. Each array task still requests exactly one L40S through
# run_model.sbatch, and each profile writes a distinct output filename.
#
# Usage:
#   bash slurm/submit_structured.sh

set -euo pipefail

REPO=/data/yazanm/SychopancyAssumptions
cd "$REPO"

PYTHON="$REPO/.venv/bin/python"
PROFILES=(structured-4dims structured-supporttypes)

for PROFILE_NAME in "${PROFILES[@]}"; do
    mapfile -t MODELS < <("$PYTHON" slurm/profile_args.py --models "$PROFILE_NAME")
    if (( ${#MODELS[@]} == 0 )); then
        echo "profile '$PROFILE_NAME' selects no models" >&2
        exit 2
    fi

    LAST_INDEX=$(( ${#MODELS[@]} - 1 ))
    SHORT_NAME="${PROFILE_NAME#structured-}"
    JOB_ID=$(sbatch \
        --parsable \
        --job-name="syco-$SHORT_NAME" \
        --array="0-$LAST_INDEX" \
        --export="ALL,PROFILE=$PROFILE_NAME" \
        slurm/run_model.sbatch)

    echo "submitted $PROFILE_NAME: job $JOB_ID (${#MODELS[@]} model tasks)"
done
