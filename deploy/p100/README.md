# P100 (Pascal) fallback OCR server

A tiny HF-`transformers` server that runs `lightonai/LightOnOCR-2-1B-bbox`
behind the two OpenAI-compatible routes `pdfparser`'s model seam
(`pipeline/model.py`) calls — so the pipeline can run on a **Tesla P100 /
compute capability sm_60 (Pascal)**, which the pinned vLLM image cannot.

## Why not vLLM on a P100

The pinned `vllm-openai:v0.22.1` image (torch 2.11) is compiled starting at
**sm_75** — it has **no Pascal kernels**, so vLLM crashes on a P100 at the first
CUDA allocation with `cudaErrorNoKernelImageForDevice`. Patching modern vLLM for
sm_60 is research-grade (FlashAttention/FlashInfer/Triton assume sm_75/80+).
`transformers` runs the model with standard ops, so a Pascal-capable torch build
is enough. The trade-off: serialized, eager attention, no continuous batching —
**materially slower** than vLLM on a supported card, and outside the pinned-vLLM
determinism/fidelity gate. If a T4/A10/A40/A6000/A100 (sm_75+) is available,
prefer `deploy/vllm/` instead; this shim becomes unnecessary.

## Setup (host venv)

Runs in its **own** environment with torch+transformers — **not** the pdfparser
venv. The GPU driver + CUDA are already on the host; no container/CDI needed.

```bash
python3 -m venv ~/p100-shim
~/p100-shim/bin/pip install -r deploy/p100/requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu121
```

### Gate: confirm torch has sm_60 (make-or-break)

```bash
~/p100-shim/bin/python -c "import torch; print(torch.cuda.get_arch_list())"
```

The list **must contain `sm_60`**. If it doesn't, this torch dropped Pascal —
lower the pin (see `requirements.txt`) and reinstall, or the shim will hit the
same `cudaErrorNoKernelImageForDevice` crash as vLLM.

## Run

```bash
~/p100-shim/bin/python deploy/p100/shim.py
```

Listens on `127.0.0.1:8000` (override `SHIM_PORT`). The model loads once and
stays warm. Confirm it's up:

```bash
curl -s 127.0.0.1:8000/v1/models
# {"object":"list","data":[{"id":"lightonocr","object":"model","max_model_len":8192}]}
```

Env knobs: `SHIM_MODEL_ID`, `SHIM_SERVED_NAME`, `SHIM_MAX_MODEL_LEN`, `SHIM_PORT`.

## Point the pipeline at it

The shim serves the pipeline's default URL, so on the same host no client config
is needed beyond forcing serial requests (the shim serializes generation on the
single GPU — parallel client requests just queue and risk timeouts):

```bash
PDFPARSER_OCR_CONCURRENCY=1 python -m pdfparser tests/fixtures/30592559.pdf /tmp/out.html
```

If the shim runs elsewhere, also set `PDFPARSER_VLLM_URL=http://<host>:8000/v1`
and `PDFPARSER_VLLM_MODEL=lightonocr`.

## Remote access (Tailscale — private, no public port)

The shim is **unauthenticated and has no TLS**, so it must never bind a public
interface. To reach it from other machines, put them on the same Tailscale
tailnet — membership is the auth, WireGuard encrypts the hop, and **no cloud
firewall port is opened** (Tailscale needs no inbound public port).

On the VM:

```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
tailscale ip -4          # note the VM's tailnet IP, e.g. 100.101.102.103
```

On each client machine: install Tailscale and `tailscale up` into the same
tailnet. Then choose one:

**A. `tailscale serve` (recommended — real HTTPS URL, shim stays on 127.0.0.1).**
Enable HTTPS in the tailnet admin console (DNS → enable MagicDNS + HTTPS), then:

```bash
tailscale serve --bg http://127.0.0.1:8000
tailscale serve status   # shows https://<vm>.<tailnet>.ts.net
```

Point the pipeline at that URL from any tailnet client:

```bash
PDFPARSER_VLLM_URL=https://<vm>.<tailnet>.ts.net/v1 \
PDFPARSER_OCR_CONCURRENCY=1 python -m pdfparser in.pdf out.html
```

**B. Bind to the tailnet IP directly (simplest — plain HTTP over the encrypted
tunnel).** Traffic is still WireGuard-encrypted; no cert needed.

```bash
SHIM_HOST=$(tailscale ip -4) python deploy/p100/shim.py
# client:
PDFPARSER_VLLM_URL=http://100.101.102.103:8000/v1 \
PDFPARSER_OCR_CONCURRENCY=1 python -m pdfparser in.pdf out.html
```

Keep port 8000 **closed** in the cloud/VM firewall in both cases — nothing should
be reachable from the public internet. Optionally tighten with tailnet ACLs so
only specific tailnet users/devices can reach the VM.

## Smoke test

`deploy/vllm/smoke-test.sh` (OCRs fixture page 1 through the chat endpoint) works
unchanged against the shim — it only needs the OpenAI-compatible endpoint up.

## Notes / caveats

- **fp16, not bf16** — Pascal has no bfloat16. Minor numeric differences vs the
  model card's bf16; acceptable for 1B OCR inference.
- **`attn_implementation="eager"`** — no FlashAttention on Pascal. If a
  transformers build rejects the kwarg for this model, drop it in `shim.py`
  (torch-SDPA's math backend also runs on sm_60).
- **Serialized** — one GPU, one generation at a time (`threading.Lock`). Throughput
  is a fraction of vLLM's on a supported card.
- **Not a general OpenAI server** — only `/v1/models` and `/v1/chat/completions`
  (single image, greedy) are implemented, matching exactly what the client parses.
