#!/bin/bash
# Cheap sanity check: do two concurrent --overlap --gpus=1 srun steps on one
# node actually get distinct physical GPUs? Bare-metal, no containers, no
# model loading -- just nvidia-smi -L and $CUDA_VISIBLE_DEVICES from each step.
#SBATCH --job-name=gpu-isolation-test
#SBATCH --partition=normal
#SBATCH --account=a145
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --time=00:05:00
#SBATCH --output=/capstor/scratch/cscs/arthur/gpu-isolation-test-%j.log

set -euo pipefail

echo "=== step A ==="
srun --ntasks=1 --gpus=1 --overlap bash -c 'echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"; nvidia-smi -L' &
PID_A=$!

echo "=== step B ==="
srun --ntasks=1 --gpus=1 --overlap bash -c 'echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"; nvidia-smi -L' &
PID_B=$!

wait "$PID_A"
wait "$PID_B"
