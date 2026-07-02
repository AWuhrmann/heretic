# GH200-tuned base image (aarch64 + Hopper CUDA, PyTorch preinstalled).
# Pick a tag matching the CUDA/driver on Alps if a specific one is required;
# "latest" works for a first build.
FROM nvcr.io/nvidia/pytorch:25.06-py3

WORKDIR /workspace/heretic

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY config.default.toml config.nohumor.toml config.noslop.toml ./

# Install heretic's own deps with pip, on top of the base image's Python
# environment, so the preinstalled GH200-optimized torch is reused as-is
# instead of being overwritten by a generic wheel.
RUN pip install --no-cache-dir -e .

ENTRYPOINT ["heretic"]
