#!/bin/bash
# Cheap sanity check, round 3:
#  - round 1 (--overlap --gpus=1, no manual env): both steps got
#    CUDA_VISIBLE_DEVICES=0 but nvidia-smi -L still showed all 4 physical
#    GPUs in both -- no real hardware isolation from Slurm's GPU plugin here.
#  - round 2 (CUDA_VISIBLE_DEVICES set before srun): both steps got "0,1"
#    instead -- Slurm overwrites whatever the parent shell exported.
# This round sets CUDA_VISIBLE_DEVICES *inside* the step's own command,
# after Slurm has already launched it, to see if that survives instead.
#SBATCH --job-name=gpu-isolation-test3
#SBATCH --partition=normal
#SBATCH --account=a145
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --time=00:05:00
#SBATCH --output=/capstor/scratch/cscs/arthur/gpu-isolation-test3-%j.log

set -euo pipefail

echo "=== step A (set inside command: CUDA_VISIBLE_DEVICES=0) ==="
srun --ntasks=1 --overlap bash -c 'export CUDA_VISIBLE_DEVICES=0; echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"; nvidia-smi -L' &
PID_A=$!

echo "=== step B (set inside command: CUDA_VISIBLE_DEVICES=1) ==="
srun --ntasks=1 --overlap bash -c 'export CUDA_VISIBLE_DEVICES=1; echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"; nvidia-smi -L' &
PID_B=$!

wait "$PID_A"
wait "$PID_B"
