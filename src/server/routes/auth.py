from datetime import timedelta
import secrets

from fastapi import APIRouter, Depends, HTTPException, Security, status
from fastapi.security import (
    HTTPAuthorizationCredentials,
    HTTPBearer,
    OAuth2PasswordRequestForm,
)
from pydantic import BaseModel

from ..config import settings
from ..services.auth import TokenError, create_access_token, verify_access_token

router = APIRouter(prefix="/api/auth", tags=["auth"])
security = HTTPBearer()
TOKEN_TYPE_BEARER = "bearer"  # nosec B105


class TokenOut(BaseModel):
    access_token: str
    token_type: str = TOKEN_TYPE_BEARER


DASHBOARD_SESSION_SUBJECT = "dashboard-admin"


# Simple in-memory admin credential for v1, set via environment variable ADMIN_PASSWORD.
def authenticate_admin_password(password: str) -> bool:
    admin_pw = getattr(settings, "admin_password", None) or None
    if not admin_pw:
        return False
    return secrets.compare_digest(password, admin_pw)


def issue_admin_token(subject: str = DASHBOARD_SESSION_SUBJECT, *, hours: int = 1) -> str:
    return create_access_token(
        subject=subject,
        expires_delta=timedelta(hours=hours),
    )


@router.post("/token", response_model=TokenOut)
def token(form_data: OAuth2PasswordRequestForm = Depends()):
    # Only support password grant for admin for now.
    if not authenticate_admin_password(form_data.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )

    access_token = issue_admin_token(subject=form_data.username, hours=1)
    return {"access_token": access_token, "token_type": TOKEN_TYPE_BEARER}


def get_current_admin(
    credentials: HTTPAuthorizationCredentials = Security(security),
) -> str:
    token = credentials.credentials
    try:
        sub = verify_access_token(token)
        return sub
    except TokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )
