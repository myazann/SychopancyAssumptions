#!/bin/bash
# Queue the 45 x 40 open-ended wave: 20 more people and 20 more dilemmas on top
# of the existing 25 x 20 `default` runs. 26,040 additional cells per model.
#
# Unlike the structured wave, the open-ended bases did not finish together --
# four models are complete and Qwen3.6-27B stopped part way. So this script
# submits only the models whose base is settled, and tells you what to do about
# the rest instead of queueing tasks that can never start.
#
#   bash slurm/submit_openended_extension.sh            # submit what is ready
#   DRY_RUN=1 bash slurm/submit_openended_extension.sh  # show, submit nothing
#
# To finish an unready base first, then chain its wave behind it:
#
#   BASE=$(sbatch --parsable --array=0 \
#       --export=ALL,PROFILE=default slurm/run_model.sbatch)
#   sbatch --dependency=afterok:$BASE --array=0 \
#       --job-name=syco-openended-45x40 \
#       --export=ALL,PROFILE=openended-extension-45x40 slurm/run_model.sbatch
#
# `afterok` rather than `aftercorr`: a single-element base array has no
# corresponding element for the other indices, so a correspondence dependency
# would leave them pending forever.

set -euo pipefail

REPO=/data/yazanm/SychopancyAssumptions
cd "$REPO"

PROFILE="${PROFILE:-openended-extension-45x40}"
PYTHON="$REPO/.venv/bin/python"

# Submission only expands YAML profiles and reads finished shards. Prevent
# numpy/OpenBLAS from trying to create one thread per login-node CPU.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "profile: $PROFILE"
echo "checking which models have a settled base..."

# --ready prints the submittable array indices on stdout and the reason each
# other model is held back on stderr, which is left on the terminal.
mapfile -t READY < <("$PYTHON" slurm/profile_args.py --ready "$PROFILE")

if (( ${#READY[@]} == 0 )); then
    echo "no model has a settled base yet; nothing to submit" >&2
    exit 1
fi

ARRAY=$(IFS=,; echo "${READY[*]}")
mapfile -t MODELS < <("$PYTHON" slurm/profile_args.py --models "$PROFILE")
echo "ready:   ${#READY[@]} of ${#MODELS[@]} model(s) -> array $ARRAY"
for INDEX in "${READY[@]}"; do
    echo "         [$INDEX] ${MODELS[$INDEX]}"
done

if [[ -n "${DRY_RUN:-}" ]]; then
    echo "DRY_RUN set; not submitting"
    exit 0
fi

SHORT_NAME="${PROFILE#openended-}"
JOB_ID=$(sbatch \
    --parsable \
    --job-name="syco-$SHORT_NAME" \
    --array="$ARRAY" \
    --export="ALL,PROFILE=$PROFILE" \
    slurm/run_model.sbatch)

echo "submitted $PROFILE: job $JOB_ID (array $ARRAY)"
echo
echo "watch:   python -m syco status --profile $PROFILE"
echo "finish:  python -m syco collect-extension --profile $PROFILE"
echo "         python -m syco parse --all --profile $PROFILE"
echo
echo "Re-run this script after a held-back base finishes; it submits only what"
echo "is newly ready, and a wave already collected resumes rather than restarts."
