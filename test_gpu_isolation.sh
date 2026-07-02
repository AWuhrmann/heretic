#!/bin/bash
# Cheap sanity check, round 2: --overlap --gpus=1 alone did NOT isolate GPUs
# (both steps got CUDA_VISIBLE_DEVICES=0 and saw all 4 GPUs -- see prior run).
# This round manually pins CUDA_VISIBLE_DEVICES per step instead, using
# `torch.cuda.current_device()`'s actual UUID (via nvidia-smi inside the
# masked view) to confirm each step is really bound to a distinct physical GPU.
#SBATCH --job-name=gpu-isolation-test2
#SBATCH --partition=normal
#SBATCH --account=a145
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --time=00:05:00
#SBATCH --output=/capstor/scratch/cscs/arthur/gpu-isolation-test2-%j.log

set -euo pipefail

echo "=== step A (CUDA_VISIBLE_DEVICES=0) ==="
CUDA_VISIBLE_DEVICES=0 srun --ntasks=1 --overlap bash -c 'echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"; nvidia-smi -L' &
PID_A=$!

echo "=== step B (CUDA_VISIBLE_DEVICES=1) ==="
CUDA_VISIBLE_DEVICES=1 srun --ntasks=1 --overlap bash -c 'echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"; nvidia-smi -L' &
PID_B=$!

wait "$PID_A"
wait "$PID_B"
