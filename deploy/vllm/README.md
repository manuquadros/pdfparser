# vLLM server for LightOnOCR (podman)

Runs `lightonai/LightOnOCR-2-1B-bbox` under vLLM as an OpenAI-compatible
server, in a rootless podman container. All GPU/torch work lives in vLLM;
`pdfparser` itself carries no torch or transformers — its model seam
(`pipeline/model.py`) is a thin HTTP client that talks to this server.

The server stays up and keeps the model resident, so each document pays the
HTTP round-trip, not a ~35 s model reload.

Two deployment shapes, both covered below:

- **Single container** (`Containerfile`) — vLLM **and** pdfparser in one image;
  convert PDFs with `podman exec`. Simplest when both run on the same box.
- **Server only** (`run-server.sh`) — just the vLLM server; run pdfparser from
  any environment (it only needs `pip install pdfparser`, no GPU) pointed at the
  server via `PDFPARSER_VLLM_URL`.

## One-time host setup

1. **Install podman** (needs sudo; the only step that does):

   ```
   sudo apt-get install -y podman
   ```

2. **GPU passthrough is already configured.** NVIDIA CDI is generated at
   `/var/run/cdi/nvidia.yaml` (`kind: nvidia.com/gpu`). Verify podman sees it:

   ```
   podman run --rm --device nvidia.com/gpu=all \
     docker.io/nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
   ```

   If the device is ever missing (the `/var/run/cdi` copy is on tmpfs and is
   lost on reboot), regenerate it to the **same** path:
   `sudo nvidia-ctk cdi generate --output=/var/run/cdi/nvidia.yaml`. For a
   reboot-persistent spec, write to `/etc/cdi/nvidia.yaml` instead **and delete
   the `/var/run/cdi` copy** — the same qualified device name in two spec dirs
   makes podman reject the CDI registry as a duplicate.

## Single container (pdfparser + vLLM)

Build the unified image from the repo root (the build context needs `src/` and
`pyproject.toml`):

```
podman build -f deploy/vllm/Containerfile -t lighton-pdfparser .
```

Start it as a warm server (mount the HF cache to skip the model download; the
`HF_HOME` env points vLLM at the mount regardless of the image's home dir):

```
podman run -d --name lighton \
  --device nvidia.com/gpu=all --ipc=host \
  -v "$HOME/.cache/huggingface:/hf-cache:rw" -e HF_HOME=/hf-cache \
  lighton-pdfparser
```

Then convert PDFs against the in-container server with `podman exec` (the image
sets `PDFPARSER_VLLM_URL=http://127.0.0.1:8000/v1`, so no flags needed):

```
podman exec lighton python3 -m pdfparser /data/in.pdf /data/out.html
```

Mount your input/output dir with `-v` on `podman run` to make `/data` visible.
The server stays resident between `exec`s — that's the whole point of one
long-lived container over a one-shot-per-PDF run.

## Server only

```
./run-server.sh
```

First run pulls the ~16 G `vllm/vllm-openai:v0.22.1` image (disk is tight — see
below). Confirm that exact tag is published first — the image tag set can lag
the pip release the spike used; check with
`podman search --list-tags docker.io/vllm/vllm-openai` (or the Docker Hub tags
page) and override via `IMAGE=…/vllm-openai:<tag> ./run-server.sh` if needed.
The model weights are **not** re-downloaded: `run-server.sh` mounts your
existing `~/.cache/huggingface` (already holds the 2.7 G bbox model). The server
is ready when the log prints `Application startup complete` on port 8000.

Tunables are env overrides, e.g.:

```
PORT=8001 GPU_MEM_UTIL=0.80 ./run-server.sh
```

## Smoke test

With the server up, in another shell:

```
./smoke-test.sh
```

Renders fixture page 1 with the project's own renderer and OCRs it through the
chat endpoint; prints the first ~1200 chars of markdown. `PDF=… ./smoke-test.sh`
to pick another file.

## Calling it from the pipeline

OpenAI-compatible chat completions, one image per request, greedy:

```python
import base64, io
from openai import OpenAI

# 127.0.0.1, not localhost — rootless podman forwards the port on IPv4 only.
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="EMPTY")

def ocr_image(img) -> str:  # img: PIL.Image
    buf = io.BytesIO(); img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    resp = client.chat.completions.create(
        model="lightonocr",
        temperature=0.0,
        max_tokens=2048,
        messages=[{"role": "user", "content": [
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ]}],
    )
    return resp.choices[0].message.content
```

This is the drop-in for `_ocr_page`'s model call. Per the determinism spike,
keep **substring/structural** acceptance — greedy vLLM is byte-stable run-to-run
but jitters bbox low-order digits across batch composition.

## Stop

`run-server.sh` uses `--rm`, so:

```
podman stop lighton-vllm
```

removes the container. To keep it running across reboots, install it as a
podman **quadlet** (systemd user unit) — ask and I'll generate the
`~/.config/containers/systemd/lighton-vllm.container` file.

## Notes / caveats

- **Disk:** the host is at ~95 % (53 G free). The image is ~16 G; it fits, but
  prune old images (`podman image prune`) if a pull fails for space.
- **6 GiB card:** `--enforce-eager` (no CUDA-graph capture) and
  `--gpu-memory-utilization 0.85` are what fit on this GPU in the spike. On a
  bigger card, drop `--enforce-eager` for CUDA graphs and raise utilization.
- **Image tag** is pinned to `v0.22.1` to match the validated spike. Bumping it
  re-opens the determinism/fidelity question — re-run the spike against the new
  tag before trusting it.
- **flashinfer:** `VLLM_USE_FLASHINFER_SAMPLER=0` avoids the startup nvcc JIT
  failure on this runtime-only host. The container itself has no CUDA toolkit
  either, so leave it set unless you switch to a `flashinfer-jit-cache` image.
