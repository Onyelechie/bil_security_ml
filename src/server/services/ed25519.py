import base64

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey


def verify_signature_b64(pubkey_b64: str, message: bytes, signature_b64: str) -> bool:
    """Verify a base64-encoded signature against the provided message and
    base64-encoded public key (Ed25519).
    """
    try:
        pub = base64.b64decode(pubkey_b64, validate=True)
        sig = base64.b64decode(signature_b64, validate=True)
        if len(pub) != 32:
            return False
    except Exception:
        return False

    try:
        vk = Ed25519PublicKey.from_public_bytes(pub)
        vk.verify(sig, message)
        return True
    except (InvalidSignature, ValueError, Exception):
        return False
