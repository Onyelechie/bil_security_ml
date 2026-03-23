## Windows Runbook — Secure Server Setup (concise)

This runbook documents the recommended, security-focused steps to run the central server on a Windows host. Paths and examples use `C:\srv\bil` and your repo workspace `C:\Users\ebere\Documents\bil_security_ml`.

Prereqs
- Admin PowerShell for initial setup
- Git, Python 3.9+ installed

1) Create a dedicated non-admin service account
- Create `bil_service` and add to `Users`. Do NOT add to `Administrators`.

2) Install Python & create virtualenv
- Install Python (use `winget` or installer), then in project:
```
cd C:\Users\ebere\Documents\bil_security_ml
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

3) Create storage & DB folders and set NTFS ACLs
- Directories:
  - `C:\srv\bil\storage`  — images
  - `C:\srv\bil\db`       — database file
- Set restrictive ACLs (grant `bil_service` modify, remove inheritance):
```
icacls "C:\srv\bil\storage" /inheritance:r
icacls "C:\srv\bil\storage" /grant "bil_service:(OI)(CI)M" /T
icacls "C:\srv\bil\db" /inheritance:r
icacls "C:\srv\bil\db" /grant "bil_service:(OI)(CI)M" /T
```

4) Set environment variables (Machine scope)
- Generate a strong `SECRET_KEY`:
```
$secret = python -c "import secrets; print(secrets.token_hex(32))"
[Environment]::SetEnvironmentVariable("SECRET_KEY", $secret, "Machine")
[Environment]::SetEnvironmentVariable("DEBUG", "false", "Machine")
[Environment]::SetEnvironmentVariable("IMAGE_STORAGE_DIR", "C:\\srv\\bil\\storage", "Machine")
[Environment]::SetEnvironmentVariable("DATABASE_URL", "sqlite:///C:/srv/bil/db/server.db", "Machine")
```

5) Configure TLS reverse proxy (recommended: Caddy)
- Use a reverse proxy on 443 that forwards to `127.0.0.1:8000`. Example `Caddyfile`:
```
your.domain.example {
  reverse_proxy 127.0.0.1:8000
}
```

6) Harden Windows Firewall
- Allow only port 443 (or restricted IPs) and block direct external access to the app port (8000):
```
New-NetFirewallRule -DisplayName "Allow HTTPS (Caddy)" -Direction Inbound -LocalPort 443 -Protocol TCP -Action Allow
New-NetFirewallRule -DisplayName "Block App Port 8000 External" -Direction Inbound -LocalPort 8000 -Protocol TCP -Action Block
```

7) Run app as service (use NSSM)
- Install `nssm`, then configure service to run `python -m uvicorn server.main:app --host 127.0.0.1 --port 8000` under `bil_service`. Ensure service uses Machine env vars or set env in NSSM.

8) Antivirus exclusions (documented & minimal)
- Add Defender exclusions for `C:\srv\bil\storage` and `C:\srv\bil\db`:
```
Add-MpPreference -ExclusionPath "C:\srv\bil\storage"
Add-MpPreference -ExclusionPath "C:\srv\bil\db"
```

9) Backups & rotation
- Daily backup of DB and `storage` to external location (retain 14 days). Use a signed PowerShell script and Task Scheduler.

10) Retention & cleanup
- The app already runs image cleanup in the background using the configured retention settings. If you need extra operational checks, schedule health monitoring rather than a separate cleanup script.

11) Logging & monitoring
- Ensure app writes logs to `C:\srv\bil\logs`. Configure rotation and retention, and forward to centralized collector where possible.

12) Web security hardening
- `DEBUG=false` in production
- Restrict CORS origins
- Add HSTS and security headers at the reverse proxy
- Protect dashboard with reverse-proxy auth or OIDC if public

13) Tests to verify
```
Invoke-WebRequest -Uri "https://your.domain.example/health" -UseBasicParsing
Test-NetConnection -ComputerName <your.public.ip> -Port 8000
```

Operational notes
- Rotate `SECRET_KEY` and service account password periodically
- Keep Windows and Python packages patched
- Consider BitLocker for disk encryption

---
Document created from recent work in March 2026.
