# Authentication & Enrollment Plan (Ed25519) — HTTP-only edges

This document describes the recommended permanent authentication scheme implemented as part of the Server improvements: Ed25519 asymmetric enrollment and per-request signatures.

Overview
- Each edge generates an Ed25519 keypair locally. The private key never leaves the edge. The server stores only the edge's public key.
- Enrollment is performed using a one-time enrollment token (created by an admin) that authorizes a single registration of an edge public key.
- Edges sign each HTTP request using the following canonical string and include signature headers. The server verifies the signature using the stored public key and enforces timestamp/nonce replay protection.

Headers and canonical signing
- Headers the edge must send on each request:
  - `X-BIL-EDGE-ID`: edge identifier (e.g., `edge-01`)
  - `X-BIL-TIMESTAMP`: UTC ISO8601 timestamp or Unix epoch seconds
  - `X-BIL-NONCE`: UUID or random hex string (>=16 bytes)
  - `X-BIL-SIG-ED25519`: base64(signature)
- Canonical string to sign (UTF-8 bytes):
```
METHOD + "\n" + PATH + "\n" + TIMESTAMP + "\n" + NONCE + "\n" + HEX(SHA256(body_bytes))
```

Enrollment flow
1. Admin creates one-time enrollment token via admin endpoint.
2. Operator on edge runs `generate_keypair` and submits public key + `edge_id` to `/api/enroll/complete` with header `X-ENROLL-TOKEN: <token>`.
3. Server validates token, stores public key, marks token used.
4. Edge signs subsequent requests with private key.

Server-side notes
- Store public keys in `edges` table: `edge_id`, `site_id`, `public_key_hex`, `created_at`, `revoked_at`, `active`.
- Nonce store: in-memory TTL store (single-node) or Redis (multi-node). TTL 300s recommended.
- Timestamp skew: default ±60s (configurable).
- Revocation: admin marks `active=false`. Server rejects signatures for revoked edges.

Operational notes
- Securely deliver enrollment tokens to provisioning operator (out-of-band).
- Protect edge private key file with OS-level permissions; consider DPAPI/OS keystore on Windows.
- Log auth failures and alert on repeated attempts.

Alternatives
- Symmetric HMAC with server-encrypted secrets (quicker to implement but requires server to hold secrets).
- mTLS (mutual TLS) — stronger but operationally heavier.

This file documents the authentication plan discussed and is intended to accompany implementation in the `Server` branch.
