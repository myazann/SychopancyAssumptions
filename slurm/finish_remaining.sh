#!/bin/bash
# Submit only the array elements that were still incomplete in the
# 2026-09-04 audit. Running or pending elements with the same job name/index
# are skipped. Re-running this script is safe: a completed result is detected
# by `syco run` and exits before model weights are loaded.
#
#   DRY_RUN=1 bash slurm/finish_remaining.sh
#   bash slurm/finish_remaining.sh

set -euo pipefail

REPO=/data/yazanm/SychopancyAssumptions
cd "$REPO"

command -v sbatch >/dev/null || {
    echo "sbatch is unavailable; run this script on a Slurm login node" >&2
    exit 2
}
command -v squeue >/dev/null || {
    echo "squeue is unavailable; cannot safely avoid duplicate array tasks" >&2
    exit 2
}

is_active() {
    local job_name=$1
    local task_index=$2
    squeue -r -h -u "$USER" -n "$job_name" -o '%i' \
        | awk -v suffix="_$task_index" \
            'substr($0, length($0) - length(suffix) + 1) == suffix { found=1 }
             END { exit !found }'
}

submit_profile() {
    local profile=$1
    local job_name=$2
    shift 2
    local pending=()
    local index
    for index in "$@"; do
        if is_active "$job_name" "$index"; then
            echo "skip $profile[$index]: already RUNNING or PENDING"
        else
            pending+=("$index")
        fi
    done
    if (( ${#pending[@]} == 0 )); then
        return
    fi

    local array
    array=$(IFS=,; echo "${pending[*]}")
    local command=(
        sbatch
        --job-name="$job_name"
        --array="$array"
        --export="ALL,PROFILE=$profile"
        slurm/run_model.sbatch
    )
    if [[ "${DRY_RUN:-0}" == 1 ]]; then
        printf 'DRY_RUN:'; printf ' %q' "${command[@]}"; echo
    else
        "${command[@]}"
    fi
}

# Model indices: 0=Qwen3.6-27B, 1=Gemma3-12B, 2=Llama-3.1-8B,
# 3=Gemma3-27B, 4=Qwen3.6-35B-A3B.
submit_profile \
    openended-extension-45x40 syco-extension-45x40 0 3
submit_profile \
    structured-4dims-extension-40x40 syco-4dims-extension-40x40 0
submit_profile \
    structured-supporttypes-extension-40x40 \
    syco-supporttypes-extension-40x40 0 3

echo
echo "Status commands:"
echo "  .venv/bin/python -m syco status --profile openended-extension-45x40"
echo "  .venv/bin/python -m syco status --profile structured-4dims-extension-40x40"
echo "  .venv/bin/python -m syco status --profile structured-supporttypes-extension-40x40"
echo
echo "After all three status commands report missing=0:"
echo "  .venv/bin/python -m syco collect-extension --profile openended-extension-45x40"
echo "  .venv/bin/python -m syco collect-extension --profile structured-4dims-extension-40x40"
echo "  .venv/bin/python -m syco collect-extension --profile structured-supporttypes-extension-40x40"
echo "  .venv/bin/python -m syco parse --all --profile openended-extension-45x40 --cells"
echo "  .venv/bin/python -m syco parse --all --profile structured-4dims-extension-40x40 --cells"
echo "  .venv/bin/python -m syco parse --all --profile structured-supporttypes-extension-40x40 --cells"
