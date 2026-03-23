from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Cookie, Form, HTTPException, Request, status
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse

from ..services.auth import TokenError, verify_access_token
from .auth import authenticate_admin_password, issue_admin_token

router = APIRouter(tags=["dashboard"])

_STATIC_ROOT = Path(__file__).resolve().parent.parent / "static" / "dashboard"
_DASHBOARD_COOKIE = "bil_dashboard_session"
_ALLOWED_ASSETS = {"styles.css", "app.js"}


def require_dashboard_session(
    dashboard_session: str | None = Cookie(default=None, alias=_DASHBOARD_COOKIE),
) -> str:
    if not dashboard_session:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Dashboard login required",
        )
    try:
        return verify_access_token(dashboard_session)
    except TokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Dashboard login required",
        ) from exc


def _login_page(*, error: str | None = None) -> HTMLResponse:
    error_html = (
        '<p style="color:#fb7185;margin:0 0 16px;">Invalid password. Try again.</p>'
        if error
        else ""
    )
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>BIL Dashboard Login</title>
  <style>
    body {{
      margin: 0;
      min-height: 100vh;
      display: grid;
      place-items: center;
      background:
        radial-gradient(circle at top left, rgba(79, 209, 197, 0.16), transparent 28%),
        radial-gradient(circle at bottom right, rgba(255, 155, 113, 0.14), transparent 24%),
        linear-gradient(135deg, #07111d 0%, #081523 48%, #050d16 100%);
      color: #e7eff8;
      font-family: "Aptos", "Segoe UI", sans-serif;
      padding: 24px;
    }}
    .card {{
      width: min(100%, 420px);
      padding: 28px;
      border-radius: 24px;
      border: 1px solid rgba(138, 163, 187, 0.16);
      background: rgba(14, 27, 44, 0.92);
      box-shadow: 0 28px 80px rgba(0, 0, 0, 0.38);
    }}
    .eyebrow {{
      margin: 0;
      font-size: 0.74rem;
      letter-spacing: 0.16em;
      text-transform: uppercase;
      color: #4fd1c5;
    }}
    h1 {{ margin: 8px 0 12px; font-size: 1.7rem; }}
    p {{ color: #90a4bb; line-height: 1.55; }}
    label {{ display: block; margin-top: 18px; color: #90a4bb; font-size: 0.9rem; }}
    input {{
      width: 100%;
      margin-top: 8px;
      padding: 12px 14px;
      border-radius: 12px;
      border: 1px solid rgba(138, 163, 187, 0.16);
      background: rgba(8, 18, 31, 0.88);
      color: #e7eff8;
      box-sizing: border-box;
    }}
    button {{
      margin-top: 18px;
      width: 100%;
      border: 0;
      border-radius: 999px;
      padding: 12px 16px;
      font-weight: 700;
      cursor: pointer;
      color: #04121c;
      background: linear-gradient(135deg, #0ea5b7, #4fd1c5);
    }}
  </style>
</head>
<body>
  <form class="card" method="post" action="/dashboard/login">
    <p class="eyebrow">Protected Dashboard</p>
    <h1>BIL Server Console</h1>
    <p>Sign in with the server admin password to open the dashboard.</p>
    {error_html}
    <label>
      Admin Password
      <input type="password" name="password" autocomplete="current-password" required>
    </label>
    <button type="submit">Open Dashboard</button>
  </form>
</body>
</html>"""
    return HTMLResponse(content=html)


@router.get("/dashboard/login", include_in_schema=False)
def dashboard_login_page():
    return _login_page()


@router.post("/dashboard/login", include_in_schema=False)
def dashboard_login(request: Request, password: str = Form(...)):
    if not authenticate_admin_password(password):
        return _login_page(error="invalid")
    response = RedirectResponse(url="/dashboard", status_code=status.HTTP_303_SEE_OTHER)
    response.set_cookie(
        key=_DASHBOARD_COOKIE,
        value=issue_admin_token(),
        httponly=True,
        samesite="lax",
        secure=request.url.scheme == "https",
    )
    return response


@router.post("/dashboard/logout", include_in_schema=False)
def dashboard_logout():
    response = RedirectResponse(url="/dashboard/login", status_code=status.HTTP_303_SEE_OTHER)
    response.delete_cookie(_DASHBOARD_COOKIE)
    return response


@router.get("/dashboard/assets/{asset_name}", include_in_schema=False)
def dashboard_asset(asset_name: str, _subject: str = Cookie(default=None, alias=_DASHBOARD_COOKIE)):
    require_dashboard_session(_subject)
    if asset_name not in _ALLOWED_ASSETS:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Asset not found")
    return FileResponse(_STATIC_ROOT / asset_name)


@router.get("/dashboard", include_in_schema=False)
def dashboard_index(dashboard_session: str | None = Cookie(default=None, alias=_DASHBOARD_COOKIE)):
    if not dashboard_session:
        return RedirectResponse(url="/dashboard/login", status_code=status.HTTP_303_SEE_OTHER)
    try:
        verify_access_token(dashboard_session)
    except TokenError:
        response = RedirectResponse(url="/dashboard/login", status_code=status.HTTP_303_SEE_OTHER)
        response.delete_cookie(_DASHBOARD_COOKIE)
        return response
    return FileResponse(_STATIC_ROOT / "index.html")
