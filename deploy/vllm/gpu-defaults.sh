#!/usr/bin/env bash
# GPU-capability-derived vLLM flags, shared by the two deployment shapes
# (run-server.sh on the host, entrypoint.sh inside the unified image) so the same
# invocation serves an Ada card and a T4.
#
# Two of the server's settings depend on the card, and they fail very differently:
#
#   --dtype bfloat16    vLLM refuses it below sm_80 and says so at startup
#                       ("Bfloat16 is only supported on GPUs with compute
#                       capability of at least 8.0"), so that half is a legible
#                       stop.
#   attention backend   vLLM's auto-selection picks FlashInfer on a T4 --
#                       FlashInferBackend.supports_compute_capability(7.5)
#                       answers True -- and the image's prebuilt kernels then
#                       kill the engine on the first request with
#                       "BatchPrefillWithPagedKVCache failed with error invalid
#                       argument".  Because that gate is optimistic rather than
#                       wrong-by-omission, auto-selection never corrects itself:
#                       the backend has to be named explicitly.
#
# Whatever the caller already exported wins; this only fills blanks.

# Lowest compute capability among the visible GPUs -- the server has to run on
# every one of them -- as "<major>.<minor>", or empty when nvidia-smi is absent
# or predates the compute_cap query.
_gpu_compute_cap() {
  nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null |
    tr -d '[:blank:]' |
    grep -E '^[0-9]+\.[0-9]+$' |
    sort -t. -k1,1n -k2,2n |
    head -1
}

# Fill in DTYPE and VLLM_ATTENTION_BACKEND to match the detected card.
apply_gpu_defaults() {
  local cap major
  cap="$(_gpu_compute_cap)"
  major="${cap%%.*}"

  if [ -n "$cap" ] && [ "$major" -ge 8 ]; then
    # Ampere or newer: the configuration the determinism spike validated, with
    # the backend left to vLLM (FlashInfer, which does run here).
    : "${DTYPE:=bfloat16}"
    echo "gpu-defaults: compute capability $cap -> --dtype $DTYPE, vLLM's own" \
         "backend choice" >&2
    return 0
  fi

  # Pre-Ampere, or a card we could not identify.  TRITON_ATTN is the backend that
  # reports support down to sm_60 and fp16 the dtype every card takes, so this
  # pair runs anywhere -- slower than FlashInfer, but it runs.  Guessing it for an
  # unidentified card is deliberate: a working server beats a first-request crash,
  # and the line below says how to take the faster path back.
  : "${DTYPE:=float16}"
  : "${VLLM_ATTENTION_BACKEND:=TRITON_ATTN}"
  if [ -n "$cap" ]; then
    echo "gpu-defaults: compute capability $cap is pre-Ampere -> --dtype $DTYPE," \
         "VLLM_ATTENTION_BACKEND=$VLLM_ATTENTION_BACKEND" >&2
  else
    echo "gpu-defaults: could not read a compute capability (no nvidia-smi?);" \
         "assuming pre-Ampere -> --dtype $DTYPE," \
         "VLLM_ATTENTION_BACKEND=$VLLM_ATTENTION_BACKEND." \
         "Set DTYPE / VLLM_ATTENTION_BACKEND to override." >&2
  fi
}
