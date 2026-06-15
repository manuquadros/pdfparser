#!/usr/bin/env bash
# Render page 1 of a fixture PDF to a base64 PNG and OCR it through the running
# vLLM server's OpenAI-compatible chat endpoint. Confirms GPU passthrough, model
# load, and the multimodal path end-to-end. Prints the first lines of markdown.
#
#   ./smoke-test.sh                       # uses tests/fixtures/31051047.pdf p1
#   PDF=tests/fixtures/30592559.pdf ./smoke-test.sh
set -euo pipefail

PORT="${PORT:-8000}"
PDF="${PDF:-tests/fixtures/31051047.pdf}"
# 127.0.0.1, not localhost: rootless podman's port-forwarder answers on IPv4
# only, so a localhost that resolves to IPv6 ::1 gets "connection reset".
ENDPOINT="http://127.0.0.1:${PORT}/v1/chat/completions"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Render with the project's own renderer so the image matches what the pipeline
# would send (200 DPI, long side <= 1540).
IMG_B64="$(pdm run python - "$PDF" <<'PY'
import base64, io, sys
from pathlib import Path
from pdfparser.pipeline.render import _render_page_images
img = _render_page_images(Path(sys.argv[1]))[0]
buf = io.BytesIO()
img.save(buf, format="PNG")
print(base64.b64encode(buf.getvalue()).decode())
PY
)"

# -sS keeps curl quiet but still prints transport errors; HTTP error *bodies*
# (e.g. a 503 while the model loads, or an {"error": ...} JSON) come back with
# exit 0, so the parser below must surface them rather than KeyError on them.
curl -sS "$ENDPOINT" \
  -H "Content-Type: application/json" \
  -d @- <<JSON | python3 -c '
import sys, json
raw = sys.stdin.read()
try:
    data = json.loads(raw)
except json.JSONDecodeError:
    sys.exit(f"server returned non-JSON (still loading?):\n{raw[:500]}")
if "choices" not in data:
    sys.exit(f"server error:\n{json.dumps(data, indent=2)[:500]}")
print(data["choices"][0]["message"]["content"][:1200])
'
{
  "model": "lightonocr",
  "temperature": 0.0,
  "max_tokens": 2048,
  "messages": [
    {"role": "user", "content": [
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,${IMG_B64}"}}
    ]}
  ]
}
JSON
