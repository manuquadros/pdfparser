#!/usr/bin/env bash
# Unified-image entrypoint: resolve the card-dependent flags at container *start*
# (the GPU is only visible then, never at build time), then hand off to the base
# image's `vllm serve`.
set -euo pipefail

# shellcheck source=deploy/vllm/gpu-defaults.sh
. /usr/local/bin/gpu-defaults.sh
apply_gpu_defaults

# vLLM rejects a command whose model is not the *first* positional ("`model`
# should be provided as the first positional argument"), so derived flags can
# only be appended, never prepended.  Appended, they would also win over an
# explicit setting in the command — vLLM takes the last --dtype, and rejects
# --attention-backend outright when --attention-config already carries one — so
# add each only when the caller supplied nothing equivalent.
derived=()
caller_dtype=""
caller_backend=""
for arg in "$@"; do
  case "$arg" in
    --dtype | --dtype=*) caller_dtype=1 ;;
    --attention-backend | --attention-backend=* | \
      --attention-config | --attention-config=* | -ac) caller_backend=1 ;;
  esac
done

[ -n "$caller_dtype" ] || derived+=(--dtype "$DTYPE")
if [ -z "$caller_backend" ] && [ -n "${ATTENTION_BACKEND:-}" ]; then
  derived+=(--attention-backend "$ATTENTION_BACKEND")
fi

exec vllm serve "$@" "${derived[@]}"
