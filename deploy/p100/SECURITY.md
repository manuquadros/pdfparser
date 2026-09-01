# P100 OCR Shim — Security Design

## Goal

Expose the LightOnOCR transformers shim (`deploy/p100/shim.py`) running on the
**P100 VM** so that it is reachable **solely from the annotation-hub backend
VM** — and from nothing else: not the public internet, not other devices on the
tailnet, not other tenants of the cloud project.

This is a single, machine-to-machine (M2M) trust relationship: the annotation-hub
backend holds one long-lived `OcrModel` (an httpx pool, see
`pipeline/model.py::OcrModel`, which can `reconnect()` across shim restarts) and
is the *only* intended client. That narrow requirement is what makes a strict
lock-down practical.

## System context

```
                          Tailscale tailnet (WireGuard, E2E encrypted)
                        ┌───────────────────────────────────────────────┐
  annotation-hub VM     │                                               │      P100 VM
  ┌───────────────┐     │   only tag:annotation-hub → tag:ocr:8000       │   ┌────────────────────┐
  │ backend       │─────┼──────────────────────────────────────────────▶│   │ shim.py            │
  │ OcrModel      │     │                                               │   │ 127.0.0.1 / tsnet  │
  │ (httpx pool)  │     │   everything else → tag:ocr : DENY             │   │ LightOnOCR (fp16)  │
  └───────────────┘     └───────────────────────────────────────────────┘   └────────────────────┘
        │                                                                            │
        └── no public inbound ──────────────────── no public inbound ───────────────┘
                         (cloud firewall default-deny; port 8000 never opened)
```

The shim itself is an **unauthenticated, plaintext** OpenAI-compatible HTTP
server that binds `127.0.0.1` by default. It has no notion of identity — all
access control is layered *around* it.

## Assets and threat model

**Assets worth protecting**

- The GPU / compute (abuse → DoS, wasted cycles; cost is bounded since it's owned
  hardware, not a metered API).
- The VM itself (a foothold for lateral movement).
- Data in transit: page images and OCR'd text of the PDFs being processed —
  potentially unpublished or sensitive scholarly content.

**Adversaries considered**

1. Internet scanners / opportunistic bots probing the VM's public IP.
2. An on-path network attacker between the two VMs.
3. Another device on the *same tailnet* (e.g. a personal laptop) — trusted to be
   on the tailnet, but **not** trusted to use the OCR GPU.
4. A compromised tailnet peer other than the annotation-hub VM.
5. A malicious or malformed request payload (crafted image) from whatever can
   reach the shim.

**Out of scope (accepted / documented, not defended here)**

- Compromise of the **annotation-hub VM** itself. By design it is the sole client,
  so its compromise *is* full access to the shim. The perimeter therefore
  includes the hub VM's own hardening — that is a deliberate consequence of the
  M2M design, not a gap in it.
- Cloud-provider / hypervisor compromise.
- Supply-chain compromise of `torch` / `transformers` / `Pillow` wheels (noted
  under residual risk).
- Physical access.

## Current controls (baseline)

| Control | Status | Defends against |
|---|---|---|
| Shim binds `127.0.0.1` (or tailnet IP), never a public interface | ✅ in code (`SHIM_HOST` default) | #1 scanners |
| No cloud firewall port opened for 8000 | ✅ operational | #1 scanners |
| Tailscale WireGuard encryption of the hop | ✅ | #2 on-path |
| `tailscale serve` optional TLS on top | ✅ optional | #2 on-path (belt) |
| Tailnet membership required to send a packet | ✅ | #1, external |

**Gap vs. the goal:** a default Tailscale tailnet is **allow-all** — every device
you have ever enrolled can reach the shim. That satisfies "a few known people"
but **not** "solely from the annotation-hub VM." Closing that gap is the core of
this design.

## Recommended design — defense in depth

Five layers, ordered from the one that *enforces the goal* outward. Layer 1 is
mandatory; 2–5 are graduated hardening.

### Layer 1 — Network isolation via Tailscale ACL + tags (enforces the goal)

Tag each VM and write an ACL that permits **only** the annotation-hub tag to
reach the OCR tag's port. This is the control that makes "solely from the
annotation-hub VM" true, and it is enforced by Tailscale's coordination plane,
not by the shim.

Bring each node up with its tag:

```bash
# on the P100 VM
sudo tailscale up --advertise-tags=tag:ocr
# on the annotation-hub VM
sudo tailscale up --advertise-tags=tag:annotation-hub
```

Tailnet policy (admin console → Access Controls). **Defining `acls` replaces the
default allow-all**, so the admin-SSH rule below is required to keep your own
access:

```jsonc
{
  "tagOwners": {
    "tag:ocr":            ["autogroup:admin"],
    "tag:annotation-hub": ["autogroup:admin"]
  },
  "acls": [
    // The annotation-hub backend is the ONLY peer allowed to reach the shim.
    // Use :443 instead of :8000 if you front the shim with `tailscale serve`.
    { "action": "accept", "src": ["tag:annotation-hub"], "dst": ["tag:ocr:8000"] },

    // Admin (you) may SSH the OCR VM — nothing else may.
    { "action": "accept", "src": ["autogroup:admin"], "dst": ["tag:ocr:22"] }
  ]
}
```

Result: a personal laptop on the tailnet, or any compromised non-hub peer,
**cannot even open a connection** to port 8000 — the packet is dropped by the
tailnet policy before it reaches the VM.

### Layer 2 — Bind + host firewall (no path from the public NIC)

- Run the shim on the tailnet interface only, or on loopback behind
  `tailscale serve`:
  ```bash
  SHIM_HOST=$(tailscale ip -4) python deploy/p100/shim.py
  # or keep 127.0.0.1 and:  tailscale serve --bg http://127.0.0.1:8000
  ```
  Never `SHIM_HOST=0.0.0.0` with a public NIC present.
- Host firewall default-deny inbound; allow only the `tailscale0` interface.
  Tailscale needs **no** inbound public port (it does NAT traversal outbound):
  ```bash
  sudo ufw default deny incoming
  sudo ufw allow in on tailscale0
  sudo ufw enable
  ```

### Layer 3 — App-layer auth (defense in depth vs. an ACL mistake)

Because the goal is single-tenant and strict, a static bearer token in the shim
is worthwhile *belt-and-suspenders*: if Layer 1 is ever misconfigured, the shim
still refuses an unauthenticated caller.

- Shim: read `SHIM_API_KEY`; if set, require `Authorization: Bearer <key>` on
  `/v1/chat/completions` (and optionally `/v1/models`), else `401`.
- Client: `pipeline/model.py` does not send an auth header today. A small,
  backward-compatible change — read `PDFPARSER_VLLM_API_KEY` in
  `load_ocr_model` and set `Authorization` on the shared `httpx.Client` — wires
  it through (the annotation-hub worker already owns the `OcrModel` lifecycle, so
  it just sets the env). **This requires that client change to be useful**; until
  then, Layers 1–2 are the enforced controls.

### Layer 4 — Process hardening (contain a payload-parsing exploit)

The shim decodes attacker-influenced bytes through `Pillow` and `transformers`;
Pillow has a CVE history (decompression bombs, malformed-image parsing). Contain
the blast radius:

- **Run as a non-root user.** A `Pillow`/`torch` RCE then isn't root.
- **Cap input size** in `_decode_image`: reject payloads over a few MB and set
  `Image.MAX_IMAGE_PIXELS` before `Image.open` to defang decompression bombs.
- **systemd unit** with `NoNewPrivileges=yes`, `PrivateTmp=yes`,
  `ProtectSystem=strict`, `ProtectHome=yes`, and a `MemoryMax=` ceiling; run it
  under a dedicated service account. (Also gives you restart-on-failure and logs.)

### Layer 5 — Host hygiene

- SSH key-only (no passwords); consider restricting port 22 to `tailscale0` too,
  so the VM has **zero** public inbound.
- Keep Tailscale device-key expiry enabled; patch the base OS.

## Residual risks (accepted)

- **Annotation-hub VM compromise ⇒ OCR access.** Intrinsic to the M2M design; the
  hub VM is now part of the perimeter and must be hardened accordingly.
- **Tailscale control-plane trust.** Tailscale brokers keys but cannot read the
  E2E-encrypted traffic. Removable via self-hosted **Headscale** if that trust is
  unacceptable.
- **Dependency / parsing CVEs** (`Pillow`, `torch`, `transformers`). Mitigated
  by Layer 4 (non-root, size caps, sandboxed unit) and by Layer 1 shrinking the
  set of callers to one, but not eliminated.
- **fp16 vs bf16 numerics** — a fidelity note, not a security one; recorded in the
  shim README.

## Verifying the goal actually holds

Test the property directly — "reachable from the hub, unreachable from anywhere
else":

1. **From the annotation-hub VM** —
   `curl -s https://<vm>.<tailnet>.ts.net/v1/models` (or the tailnet-IP URL)
   returns the model JSON. ✅ intended path works.
2. **From another tailnet device** (your laptop) — the same `curl` **hangs /
   is refused**. ✅ ACL denies non-hub peers.
3. **From the public internet** — `nmap -Pn -p 8000,22 <VM_public_IP>` shows
   8000 filtered/closed. ✅ no public surface.
4. **App-layer (if Layer 3 enabled)** — a request without the bearer token gets
   `401` even from the hub. ✅ defense in depth active.

Re-run 2 and 3 after any tailnet/firewall change — they are the regression tests
for this design.

## Summary

For a single-client M2M deployment, this posture is strong: **the tailnet ACL
(Layer 1) is what enforces "solely from the annotation-hub VM,"** with host
firewall, optional app-layer token, process sandboxing, and host hygiene as
graduated defense in depth. The dominant residual risk is the trust placed in the
annotation-hub VM itself — which is inherent to naming it the sole client, and is
addressed by hardening *that* VM, not this one.
