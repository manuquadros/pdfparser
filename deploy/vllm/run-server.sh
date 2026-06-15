#!/usr/bin/env bash
# Start the LightOnOCR vLLM OpenAI-compatible server in a rootless podman
# container. Keeps the model warm across documents so the pipeline pays the
# ~35 s load once, not per run.
#
# Settings are the ones the §4 determinism spike validated on this 6 GiB card
# (see spike_results/vllm_determinism.md): bf16, enforce-eager (no CUDA-graph
# VRAM), util 0.85, 8 k ctx, one image per prompt, and the flashinfer JIT
# sampler disabled (the box has no nvcc — VLLM_USE_FLASHINFER_SAMPLER=0 falls
# back to the Torch-native sampler, exactly as the spike ran).
#
# Override any value via env, e.g.  PORT=8001 ./run-server.sh
set -euo pipefail

IMAGE="${IMAGE:-docker.io/vllm/vllm-openai:v0.22.1}"
NAME="${NAME:-lighton-vllm}"
MODEL="${MODEL:-lightonai/LightOnOCR-2-1B-bbox}"
PORT="${PORT:-8000}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"

GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"

# Mount the host HF cache at a fixed path and point HF_HOME at it explicitly,
# rather than guessing the image's home dir — the cache is reused (no re-pull of
# the 2.7 GiB model) whether the container runs as root or a non-root user.
# Rootless podman maps the host user onto the container user, so the existing
# weights remain readable through the mount.
exec podman run --rm \
  --name "$NAME" \
  --device nvidia.com/gpu=all \
  --ipc=host \
  -p "${PORT}:8000" \
  -v "${HF_CACHE}:/hf-cache:rw" \
  -e HF_HOME=/hf-cache \
  -e VLLM_USE_FLASHINFER_SAMPLER=0 \
  -e HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}" \
  "$IMAGE" \
    "$MODEL" \
    --served-model-name lightonocr \
    --dtype bfloat16 \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --limit-mm-per-prompt '{"image": 1}' \
    --enforce-eager
