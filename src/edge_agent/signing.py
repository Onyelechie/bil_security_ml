from __future__ import annotations

import base64

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


def sign_message_b64(private_key_b64: str, message: bytes) -> str:
    key_bytes = base64.b64decode(private_key_b64, validate=True)
    if len(key_bytes) != 32:
        raise ValueError("Invalid Ed25519 private key length")
    signing_key = Ed25519PrivateKey.from_private_bytes(key_bytes)
    signature = signing_key.sign(message)
    return base64.b64encode(signature).decode("ascii")
