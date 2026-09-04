#!/bin/bash
# Re-audit the two 200-call pilots with the current post-generation parser,
# then submit the full sharded labeling array. Gemma's otherwise-valid JSON
# fences are accepted without another pilot generation call.
#
#   DRY_RUN=1 bash slurm/submit_linear_probe_labels.sh
#   bash slurm/submit_linear_probe_labels.sh

set -euo pipefail

REPO=/data/yazanm/SychopancyAssumptions
cd "$REPO"
source "$REPO/.venv/bin/activate"

PROBE_CONFIG="${PROBE_CONFIG:-config/linear_probe.yaml}"

if [[ "${DRY_RUN:-0}" == 1 ]]; then
    echo "DRY_RUN: python -m syco linear-probe parse-labels --config $PROBE_CONFIG --allow-partial"
else
    # This writes only derived pilot QA artifacts; raw generations remain immutable.
    python -m syco linear-probe parse-labels \
        --config "$PROBE_CONFIG" \
        --allow-partial
fi

command -v sbatch >/dev/null || {
    echo "sbatch is unavailable; run this script on a Slurm login node" >&2
    exit 2
}

COMMAND=(
    sbatch
    --array=0-13%8
    --export="ALL,PROBE_CONFIG=$PROBE_CONFIG"
    slurm/linear_probe_labels.sbatch
)
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf 'DRY_RUN:'; printf ' %q' "${COMMAND[@]}"; echo
else
    "${COMMAND[@]}"
fi

echo
echo "After every array task finishes:"
echo "  .venv/bin/python -m syco linear-probe parse-labels --config $PROBE_CONFIG"
echo "  .venv/bin/python -m syco linear-probe status --config $PROBE_CONFIG"
