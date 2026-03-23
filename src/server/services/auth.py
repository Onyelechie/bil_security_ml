import base64
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import json
from typing import Any, Optional

from ..config import settings

# JWT config
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60


class TokenError(ValueError):
    pass


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64url_decode(raw: str) -> bytes:
    padding = "=" * (-len(raw) % 4)
    return base64.urlsafe_b64decode(raw + padding)


def _encode_segment(data: dict[str, Any]) -> str:
    return _b64url_encode(
        json.dumps(data, separators=(",", ":"), sort_keys=True).encode("utf-8")
    )


def _decode_segment(segment: str) -> dict[str, Any]:
    try:
        decoded = _b64url_decode(segment)
        data = json.loads(decoded.decode("utf-8"))
    except (ValueError, json.JSONDecodeError) as exc:
        raise TokenError("Malformed token segment") from exc
    if not isinstance(data, dict):
        raise TokenError("Invalid token payload")
    return data


def _sign(message: str) -> str:
    digest = hmac.new(
        settings.secret_key.encode("utf-8"),
        message.encode("ascii"),
        hashlib.sha256,
    ).digest()
    return _b64url_encode(digest)


def create_access_token(subject: str, expires_delta: Optional[timedelta] = None) -> str:
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    header = {"alg": ALGORITHM, "typ": "JWT"}
    payload = {"sub": subject, "exp": int(expire.timestamp())}
    signing_input = f"{_encode_segment(header)}.{_encode_segment(payload)}"
    return f"{signing_input}.{_sign(signing_input)}"


def verify_access_token(token: str) -> str:
    try:
        header_segment, payload_segment, signature_segment = token.split(".")
    except ValueError as exc:
        raise TokenError("Malformed token") from exc

    signing_input = f"{header_segment}.{payload_segment}"
    expected_signature = _sign(signing_input)
    if not hmac.compare_digest(signature_segment, expected_signature):
        raise TokenError("Invalid token signature")

    header = _decode_segment(header_segment)
    if header.get("alg") != ALGORITHM or header.get("typ") != "JWT":
        raise TokenError("Unexpected token header")

    payload = _decode_segment(payload_segment)
    exp = payload.get("exp")
    if not isinstance(exp, int):
        raise TokenError("Missing expiration")
    now_ts = int(datetime.now(timezone.utc).timestamp())
    if exp < now_ts:
        raise TokenError("Token expired")

    sub = payload.get("sub")
    if not isinstance(sub, str) or not sub:
        raise TokenError("Missing subject")
    return sub
