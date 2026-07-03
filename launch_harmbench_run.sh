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
#
# GPU count scales with #SBATCH --gpus-per-node below (default 2: 1 for the
# classifier, 1 for heretic). For a model too big for one GPU (e.g. a 70B
# model needing ~140GB, more than one GH200's 120GB), override it at
# submission time instead of editing this script -- an sbatch CLI flag beats
# a script's own #SBATCH directive:
#   sbatch --gpus-per-node=3 launch_harmbench_run.sh /path/to/70b-model
# The classifier always gets GPU 0; heretic gets every other GPU actually
# granted (1..N-1) via its own device_map="auto" model-parallel sharding --
# computed at runtime below, not hardcoded, so this scales to any N.
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
# Namespaced by job ID so results from different runs never collide, kept on
# scratch (not inside the repo) since these are output artifacts, not code.
PARETO_ADAPTERS_DIR="/capstor/scratch/cscs/arthur/harmbench_results/${SLURM_JOB_ID}/pareto_adapters"

# Pulled out and passed as an explicit --model below (not left as a bare
# trailing arg) so it can't collide with heretic's "last bare arg = model"
# shorthand heuristic (main.py ~193-203) when extra flags like --n-trials
# follow it -- that heuristic grabbed --n-trials's own value ("10") as the
# model instead, once bitten.
MODEL_PATH="$1"
shift

# GPU 0 is always the classifier; heretic gets the rest (1..N-1), sized from
# however many GPUs were actually requested at submission, not a hardcoded
# count -- so --gpus-per-node=3 at submission time gives heretic GPUs "1,2"
# automatically, no script edit needed.
# NOTE: SLURM_GPUS_ON_NODE is NOT what it sounds like on this cluster --
# verified empirically it reports the node's total physical GPU count (4),
# not what --gpus-per-node actually requested. SLURM_GPUS_PER_NODE is the
# one that matches the request (confirmed: --gpus-per-node=3 in, 3 out).
TOTAL_GPUS="${SLURM_GPUS_PER_NODE:-2}"
if [ "$TOTAL_GPUS" -lt 2 ]; then
    echo "ERROR: need at least 2 GPUs (1 classifier + 1 heretic), got $TOTAL_GPUS." >&2
    exit 1
fi
HERETIC_GPUS=$(seq -s, 1 $((TOTAL_GPUS - 1)))
echo "Total GPUs: $TOTAL_GPUS (classifier: GPU 0, heretic: GPU(s) $HERETIC_GPUS)"

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

# --- Run heretic on GPU(s) $HERETIC_GPUS, pointed at the classifier over localhost ---
# --study-checkpoint-dir points at a HarmBench-specific directory (default is
# "checkpoints") so this doesn't collide with a checkpoint from earlier
# interactive runs on this same model.
#
# The checkpoint file itself is keyed by model path, not by job -- so every
# sbatch resubmission against the same model shares one checkpoint file.
# --resume-study true makes heretic continue it automatically instead of
# prompting "how would you like to proceed?" (no TTY here, would crash on
# EOF) -- meaning a job that got cancelled/timed out partway through picks
# up where it left off on the next submission instead of losing all
# completed trials. If you actually want to discard prior progress and
# start over, pass --resume-study false instead.
# NOTE: unlike plain-bool fields (e.g. --orthogonalize-direction /
# --no-orthogonalize-direction), resume_study is `bool | None`, and
# pydantic-settings' cli_implicit_flags does NOT give bool|None fields the
# --flag/--no-flag treatment -- it requires an explicit value argument
# ({bool,null}). Verified empirically against pydantic-settings==2.14.2
# (the version actually installed in the image) after --resume-study alone
# failed with "expected one argument".
#
# --save-pareto-adapters-dir closes the other interactive gap: once the study
# finishes, heretic saves the LoRA adapter for every Pareto-optimal trial
# (best refusals/KL divergence trade-offs) to this directory automatically,
# instead of dropping into the interactive trial-selection/save menu that
# would otherwise crash on EOF (no TTY available here either).
srun --ntasks=1 --overlap \
    --environment="$REPO_DIR/heretic.edf.toml" \
    env CUDA_VISIBLE_DEVICES="$HERETIC_GPUS" heretic \
        --model "$MODEL_PATH" \
        --harmbench-classifier-url "$CLASSIFIER_URL" \
        --study-checkpoint-dir checkpoints-harmbench \
        --resume-study true \
        --save-pareto-adapters-dir "$PARETO_ADAPTERS_DIR" \
        "$@"

echo "Pareto-optimal adapters saved to: $PARETO_ADAPTERS_DIR"
