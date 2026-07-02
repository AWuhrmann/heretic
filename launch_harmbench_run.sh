#!/bin/bash
# Single-node HarmBench-judged abliteration run.
#
# One GH200 node has 4 GPUs, and both the classifier (13B) and the target
# model (typically <70B here) each fit comfortably on a single GH200's
# 96GB HBM3e -- so there's no need to split across nodes. Two concurrent
# srun steps share one job allocation (--overlap), each requesting its own
# GPU via --gpus=1 so Slurm's own GRES scheduler assigns distinct physical
# GPUs and isolates them via cgroups; they talk over localhost instead of
# cross-node hostnames.
#
# Usage: sbatch launch_harmbench_run.sh /path/to/model [extra heretic args...]
#SBATCH --job-name=heretic-harmbench
#SBATCH --partition=normal
#SBATCH --account=a145
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --time=04:00:00
#SBATCH --output=/capstor/scratch/cscs/arthur/heretic-harmbench-%j.log

set -euo pipefail

REPO_DIR="/capstor/scratch/cscs/arthur/Apertus-1.5/heretic"
CLASSIFIER_PORT=8000
CLASSIFIER_URL="http://localhost:${CLASSIFIER_PORT}/v1"

cd "$REPO_DIR"

# --- One-time-ish data/config prep (cheap, idempotent, no GPU involved) ---
if [ ! -f harmbench_behaviors.txt ]; then
    ./prepare_harmbench_data.sh harmbench_behaviors.txt
fi
cp config.harmbench.toml config.toml

# --- Start the classifier server, in the background, on its own GPU ---
# NOTE: not manually setting CUDA_VISIBLE_DEVICES here -- relying on Slurm's
# GRES scheduler + --gpus=1 --overlap to hand each concurrent step a distinct
# physical GPU and cgroup-isolate it. Unverified against this exact Slurm
# config; if both steps end up on the same GPU, that's the thing to fix.
srun --ntasks=1 --gpus=1 --overlap \
    --environment="$REPO_DIR/classifier.edf.toml" \
    vllm serve cais/HarmBench-Llama-2-13b-cls \
        --port "$CLASSIFIER_PORT" \
        --max-model-len 4096 \
    > "/capstor/scratch/cscs/arthur/classifier-${SLURM_JOB_ID}.log" 2>&1 &
CLASSIFIER_PID=$!

cleanup() {
    kill "$CLASSIFIER_PID" 2>/dev/null || true
    wait "$CLASSIFIER_PID" 2>/dev/null || true
}
trap cleanup EXIT

# --- Wait for the classifier to become ready (20s x 60 = 20min cap) ---
echo "Waiting for classifier at $CLASSIFIER_URL to become ready..."
READY=0
for i in $(seq 1 60); do
    if timeout 5 curl -sf "${CLASSIFIER_URL}/models" > /dev/null 2>&1; then
        echo "Classifier ready after $((i * 20))s."
        READY=1
        break
    fi
    sleep 20
done
if [ "$READY" -ne 1 ]; then
    echo "ERROR: classifier did not become ready within 20 minutes. See classifier-${SLURM_JOB_ID}.log" >&2
    exit 1
fi

# --- Run heretic on its own GPU, pointed at the classifier over localhost ---
srun --ntasks=1 --gpus=1 --overlap \
    --environment="$REPO_DIR/heretic.edf.toml" \
    heretic \
        --harmbench-classifier-url "$CLASSIFIER_URL" \
        "$@"
