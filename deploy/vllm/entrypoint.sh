#!/usr/bin/env bash
# Unified-image entrypoint: resolve the card-dependent flags at container *start*
# (the GPU is only visible then, never at build time), then hand off to the base
# image's `vllm serve`.
set -euo pipefail

# shellcheck source=deploy/vllm/gpu-defaults.sh
. /usr/local/bin/gpu-defaults.sh
apply_gpu_defaults

# An empty VLLM_ATTENTION_BACKEND is not the same as an unset one; on Ampere+ the
# backend is vLLM's to choose.
if [ -n "${VLLM_ATTENTION_BACKEND:-}" ]; then
  export VLLM_ATTENTION_BACKEND
else
  unset VLLM_ATTENTION_BACKEND
fi

# vLLM rejects a command whose model is not the *first* positional ("`model`
# should be provided as the first positional argument"), so the derived --dtype
# can only be appended, never prepended.  Appended, it would also win over an
# explicit --dtype in the command (vLLM takes the last occurrence) — so add ours
# only when the caller supplied none.
for arg in "$@"; do
  case "$arg" in
    --dtype | --dtype=*)
      exec vllm serve "$@"
      ;;
  esac
done

exec vllm serve "$@" --dtype "$DTYPE"
