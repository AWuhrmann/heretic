#!/bin/bash
# Single-node HarmBench-judged abliteration run.
#
# One GH200 node has 4 GPUs, and both the classifier (13B) and the target
# model each fit comfortably on a single GH200's 120GB HBM3e -- so there's
# no need to split across nodes. Two concurrent srun steps share one job
# allocation (--overlap), talking over localhost instead of cross-node
# hostnames.
#
# GPU assignment: verified empirically (test_gpu_isolation.sh) that this
# cluster's Slurm config does NOT cgroup-isolate GPUs per --gpus=1 step, and
# overwrites any CUDA_VISIBLE_DEVICES exported before srun. What does work:
# setting it via `env` as part of the step's own command, after srun has
# already launched it -- that survives and is respected by the CUDA runtime
# (nvidia-smi itself ignores the var either way; it queries the driver
# directly rather than going through libcudart, so seeing all 4 GPUs there
# is not evidence against isolation).
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

# --- Start the classifier server, in the background, on GPU 0 ---
# No --max-model-len override: the model's own config caps at 2048
# (max_position_embeddings), and our classifier prompts (HarmBench's rules
# text + a short behavior + a short generation) comfortably fit well under
# that, so just let vLLM derive it instead of forcing a too-large value.
srun --ntasks=1 --overlap \
    --environment="$REPO_DIR/classifier.edf.toml" \
    env CUDA_VISIBLE_DEVICES=0 vllm serve cais/HarmBench-Llama-2-13b-cls \
        --port "$CLASSIFIER_PORT" \
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

# --- Run heretic on GPU 1, pointed at the classifier over localhost ---
# --study-checkpoint-dir points at a HarmBench-specific directory (default is
# "checkpoints") so this doesn't collide with any checkpoint left over from
# earlier interactive runs on this same model -- if it did, heretic would hit
# an interactive "resume previous study?" prompt, and there's no TTY here
# (sbatch/srun --overlap, not --pty), so that prompt would crash with EOFError.
#
# NOTE: this only covers the *start* of the run. This fork predates upstream's
# headless-operation support (added after our fork point), so the
# *post-optimization* flow (trial selection, then "what do you want to do
# with the model" menu) is still interactive with no CLI escape hatch found
# so far -- once a run actually reaches 200/200 trials, expect it to hang/
# crash there too. Not fixed yet; out of scope for just getting trials running.
srun --ntasks=1 --overlap \
    --environment="$REPO_DIR/heretic.edf.toml" \
    env CUDA_VISIBLE_DEVICES=1 heretic \
        --harmbench-classifier-url "$CLASSIFIER_URL" \
        --study-checkpoint-dir checkpoints-harmbench \
        "$@"
