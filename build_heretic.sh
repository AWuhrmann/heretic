#!/bin/bash
#SBATCH --job-name=heretic-build
#SBATCH --partition=normal
#SBATCH --account=a145
#SBATCH --time=02:00:00
#SBATCH --output=/capstor/scratch/cscs/arthur/heretic-build-%j.log

set -euo pipefail

REPO_DIR="/capstor/scratch/cscs/arthur/Apertus-1.5/heretic"
CONTAINERS_DIR="/capstor/scratch/cscs/arthur/containers"

cd "$REPO_DIR"

echo "=== podman images before build ==="
podman images

if ! podman image exists heretic:latest; then
    echo "=== building image ==="
    podman build -t heretic:latest .
else
    echo "=== heretic:latest already present in this allocation, skipping build ==="
fi

echo "=== image size ==="
podman inspect heretic:latest --format '{{.Size}}' | numfmt --to=iec

mkdir -p "$CONTAINERS_DIR"
lfs setstripe -c -1 "$CONTAINERS_DIR"
cd "$CONTAINERS_DIR"
rm -f heretic.sqsh

echo "=== importing to squashfs ==="
enroot import -x mount -o heretic.sqsh podman://heretic:latest

echo "=== result ==="
ls -lh heretic.sqsh
unsquashfs -l heretic.sqsh | grep -i 'torch/version.py' && echo "torch OK" || echo "WARNING: torch not found in image"
