#!/bin/bash
# ---------------------------------------------------------------------------
# Preflight: 10 minutes on one L40S. Run this ONCE before the real jobs.
#
# It answers the four questions that decide whether a 5-day job will die in
# its first minute on this cluster:
#   1. does the .venv (built on vibranium, CUDA 13) load on an L40S node?
#   2. was llama-cpp-python compiled with CUDA, i.e. will the GPU be used?
#   3. do the compute nodes have outbound internet? The GGUF filename is
#      resolved from the HF repo listing at model-load time (a network call),
#      so a firewalled node fails there even though the weights are cached.
#   4. can the configured Flash Attention / prompt-batch settings load a real
#      GGUF and generate both a control and persona-conditioned result?
#
#   sbatch slurm/00_check.sbatch
# ---------------------------------------------------------------------------
#SBATCH --job-name=syco-check
#SBATCH --partition=L40s_staff
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=/data/yazanm/SychopancyAssumptions/slurm/logs/%x-%j.out
#SBATCH --error=/data/yazanm/SychopancyAssumptions/slurm/logs/%x-%j.err

set -euo pipefail

REPO=/data/yazanm/SychopancyAssumptions
cd "$REPO"

export HF_HOME=/data/yazanm/huggingface_cache
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=false

source "$REPO/.venv/bin/activate"

echo "== host / allocation"
hostname
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi
echo

echo "== nvidia-smi indices as the scheduler in syco/orchestrate.py sees them"
# run --all assigns work by these indices. They must start at 0 and cover
# exactly the GPUs this job was allocated, or a child lands on a card that
# belongs to somebody else's job.
nvidia-smi --query-gpu=index,memory.total,memory.free --format=csv,noheader,nounits
echo

echo "== llama-cpp-python CUDA support"
python - <<'PY'
import llama_cpp
ok = llama_cpp.llama_supports_gpu_offload()
print("llama_cpp", llama_cpp.__version__, "gpu_offload:", ok)
raise SystemExit(0 if ok else "llama-cpp-python has no CUDA support on this node")
PY
echo

echo "== syco doctor"
python -m syco doctor
echo

echo "== HF hub reachability (the GGUF filename resolution done at load time)"
python - <<'PY'
from syco.model_registry import load_registry
reg = load_registry()
for alias in ("Gemma3-12B", "Qwen3.6-35B-A3B"):
    spec = reg.with_resolved_quant(reg.get(alias))
    print(f"  {alias}: {spec.quantization.resolved_file}")
PY
echo

echo "== tuned llama.cpp end-to-end smoke (real 8B GGUF, two cells)"
SMOKE_BASE="${SLURM_TMPDIR:-/tmp}"
SMOKE_DIR=$(mktemp -d "$SMOKE_BASE/syco-l40-smoke.XXXXXX")
cleanup_smoke() {
    rm -rf -- "$SMOKE_DIR"
}
trap cleanup_smoke EXIT
python -m syco run \
    --model Llama-3.1-8B \
    --out "$SMOKE_DIR/llama.jsonl" \
    --probe openended \
    --n-personas 1 \
    --n-prompts 1 \
    --limit 2
python -m syco parse "$SMOKE_DIR/llama.jsonl" --cells
echo

echo "== end-to-end offline smoke (mock backend, no weights)"
python -m syco smoke

echo
echo "PREFLIGHT OK"
