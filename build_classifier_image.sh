#!/bin/bash
#SBATCH --job-name=vllm-classifier-build
#SBATCH --partition=normal
#SBATCH --account=a145
#SBATCH --time=03:00:00
#SBATCH --output=/capstor/scratch/cscs/arthur/vllm-classifier-build-%j.log
#
# Builds a vLLM image for serving the HarmBench classifier on GH200.
#
# Deliberately does NOT reimplement vLLM's build as a hand-written
# Containerfile: vLLM has no aarch64 PyPI wheels (github.com/vllm-project/vllm
# issue #23350) and must be compiled from source, with Flash-Attention and
# FlashInfer also compiled from source. vLLM's own docker/Dockerfile already
# does this correctly and is maintained upstream, with a documented GH200
# build path (https://docs.vllm.ai/en/v0.18.0/getting_started/installation/gpu/).
# So: clone vLLM's latest release tag and build THEIR Dockerfile, rather than
# re-deriving the same recipe by hand and risking a subtly wrong flag.
#
# Kept as a separate image from heretic:latest on purpose -- vLLM's build
# pins its own torch/CUDA toolchain at compile time, which would conflict
# with deliberately NOT touching heretic's base image's GH200-tuned torch.
set -euo pipefail

VLLM_DIR="${VLLM_DIR:-/capstor/scratch/cscs/arthur/vllm-src}"
IMAGE_TAG="${IMAGE_TAG:-vllm-gh200-classifier:latest}"

if [ ! -d "$VLLM_DIR" ]; then
    git clone --depth 1 https://github.com/vllm-project/vllm.git "$VLLM_DIR"
fi

cd "$VLLM_DIR"

LATEST_TAG=$(git ls-remote --tags --refs origin | awk -F/ '{print $NF}' | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' | sort -V | tail -1)
echo "=== building vLLM ${LATEST_TAG} for GH200 ==="
git fetch --depth 1 origin "tag" "$LATEST_TAG"
git checkout "$LATEST_TAG"

# max_jobs=66 matches vLLM's own GH200 doc example (GH200 nodes commonly have
# ~72 Grace CPU cores); reduce this if the build OOMs or the allocated node
# has fewer cores.
podman build . \
    --file docker/Dockerfile \
    --target vllm-openai \
    --platform "linux/arm64" \
    -t "$IMAGE_TAG" \
    --build-arg max_jobs=66 \
    --build-arg nvcc_threads=2 \
    --build-arg torch_cuda_arch_list="9.0 10.0+PTX" \
    --build-arg RUN_WHEEL_CHECK=false

echo "=== image size ==="
podman inspect "$IMAGE_TAG" --format '{{.Size}}' | numfmt --to=iec

CONTAINERS_DIR="${CONTAINERS_DIR:-/capstor/scratch/cscs/arthur/containers}"
mkdir -p "$CONTAINERS_DIR"
lfs setstripe -c -1 "$CONTAINERS_DIR" || true
cd "$CONTAINERS_DIR"
rm -f vllm-classifier.sqsh

echo "=== importing to squashfs ==="
enroot import -x mount -o vllm-classifier.sqsh "podman://${IMAGE_TAG}"

echo "=== result ==="
ls -lh vllm-classifier.sqsh
