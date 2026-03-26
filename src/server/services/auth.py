from datetime import datetime, timedelta, timezone
from typing import Optional

from authlib.jose import jwt
from authlib.jose.errors import JoseError

from ..config import settings

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60


class TokenError(ValueError):
    pass


def create_access_token(subject: str, expires_delta: Optional[timedelta] = None) -> str:
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    header = {"alg": ALGORITHM}
    payload = {
        "sub": subject,
        "exp": int(expire.timestamp()),
        "iat": int(datetime.now(timezone.utc).timestamp()),
    }
    token = jwt.encode(header, payload, settings.secret_key)
    return token.decode("utf-8") if isinstance(token, bytes) else token


def verify_access_token(token: str) -> str:
    try:
        claims = jwt.decode(token, settings.secret_key)
        claims.validate()
        subject = claims["sub"]
    except (JoseError, KeyError) as exc:
        raise TokenError("Invalid token") from exc

    if not isinstance(subject, str) or not subject:
        raise TokenError("Missing subject")
    return subject
