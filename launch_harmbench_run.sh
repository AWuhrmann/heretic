#!/bin/bash
# Multi-node HarmBench-judged abliteration run.
#
# Requests 2 nodes in a single allocation: node 0 serves the HarmBench
# classifier via vLLM (background), node 1 runs heretic's Optuna loop,
# pointed at node 0 via --harmbench-classifier-url. One job to submit and
# monitor; node discovery is automatic via $SLURM_JOB_NODELIST, no manual
# coordination file needed.
#
# Usage: sbatch launch_harmbench_run.sh /path/to/model [extra heretic args...]
#SBATCH --job-name=heretic-harmbench
#SBATCH --partition=normal
#SBATCH --account=a145
#SBATCH --nodes=2
#SBATCH --time=04:00:00
#SBATCH --output=/capstor/scratch/cscs/arthur/heretic-harmbench-%j.log

set -euo pipefail

REPO_DIR="/capstor/scratch/cscs/arthur/Apertus-1.5/heretic"
CLASSIFIER_PORT=8000

cd "$REPO_DIR"

# --- One-time-ish data/config prep (cheap, idempotent, no GPU involved) ---
if [ ! -f harmbench_behaviors.txt ]; then
    ./prepare_harmbench_data.sh harmbench_behaviors.txt
fi
cp config.harmbench.toml config.toml

# --- Resolve node hostnames: node 0 = classifier, node 1 = heretic ---
NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
CLASSIFIER_NODE="${NODES[0]}"
HERETIC_NODE="${NODES[1]}"
echo "Classifier node: $CLASSIFIER_NODE"
echo "Heretic node:    $HERETIC_NODE"

# --- Start the classifier server on node 0, in the background ---
srun --nodes=1 --ntasks=1 -w "$CLASSIFIER_NODE" \
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
CLASSIFIER_URL="http://${CLASSIFIER_NODE}:${CLASSIFIER_PORT}/v1"
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

# --- Run heretic on node 1, pointed at the classifier ---
srun --nodes=1 --ntasks=1 -w "$HERETIC_NODE" \
    --environment="$REPO_DIR/heretic.edf.toml" \
    heretic \
        --harmbench-classifier-url "$CLASSIFIER_URL" \
        "$@"
