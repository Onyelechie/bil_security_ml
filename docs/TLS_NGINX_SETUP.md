# TLS termination with nginx and Let's Encrypt

This document describes a recommended setup to terminate TLS for the `bil_security_ml` server using nginx and Let's Encrypt. It includes a simple production nginx config, a local `mkcert` helper for development, and notes on optional mTLS device authentication.

Summary
- Terminate TLS at an nginx reverse proxy. nginx handles certificates, HTTP->HTTPS redirects, HSTS, and proxy headers.
- Use Let's Encrypt (ACME) for free automated certificates in production.
- Use `mkcert` for locally-trusted development certificates on Windows and macOS.
- Optional: mTLS (client certificates) can be enabled at nginx to authenticate edge devices.

Files in this repo
- `infra/nginx/` - nginx configuration files (site conf, global nginx.conf, TLS options).
- `scripts/mkcert-setup.ps1` - PowerShell helper to generate a dev certificate using `mkcert`.
- `scripts/obtain_lets_encrypt_cert.ps1` - Guidance / helper for obtaining Let's Encrypt certs on Windows.

High-level steps (production)
1. Install nginx on your host or run `infra/nginx` inside a container.
2. Obtain a domain (e.g. `example.com`) and point DNS to your server.
3. Use Certbot or `win-acme` (Windows) to request a certificate, or use the provided docker-compose for automated ACME.
4. Place the obtained `fullchain.pem` and `privkey.pem` under `/etc/letsencrypt/live/<domain>/` (or the path configured in `infra/nginx/conf.d/bil_security_ml.conf`).
5. Start nginx and verify `https://<domain>/` works. nginx proxies to the app server (uvicorn) running on localhost:8000.

Local development (Windows)
1. Install `mkcert` (https://github.com/FiloSottile/mkcert). On Windows you can install via `choco install mkcert` or `scoop`.
2. Run the provided script: `powershell -ExecutionPolicy Bypass -File scripts/mkcert-setup.ps1`.
3. This will create a dev cert and key under `infra/certs/dev/` (this folder is ignored from git). Use the generated files with uvicorn or your reverse proxy for local HTTPS.

Optional: mTLS for device authentication
- Generate a private CA and issue client certs for each edge device.
- Configure nginx with `ssl_client_certificate /etc/nginx/certs/ca.crt; ssl_verify_client on;` in the server block. See `infra/nginx/conf.d/bil_security_ml.conf` for example comments.

Security notes
- Do NOT commit private keys into the repository. `infra/certs/` is listed in `.gitignore` and should remain local to the host.
- Use modern TLS (TLS 1.3; allow TLS 1.2 only if needed). See `infra/nginx/ssl-params.conf` for recommended options.
- Enable `Strict-Transport-Security` and redirect HTTP to HTTPS.

References
- nginx TLS hardening: https://ssl-config.mozilla.org/
- Let's Encrypt: https://letsencrypt.org/
- mkcert: https://github.com/FiloSottile/mkcert
