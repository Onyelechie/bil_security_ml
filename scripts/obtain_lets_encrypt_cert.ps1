<#
Helper guidance for obtaining Let's Encrypt certs on Windows using win-acme (recommended) or certbot on Linux.

This script will NOT automatically enroll for you. It prints recommended commands and can call `wacs.exe` if present.
#>

param(
    [string]$Domain = "example.com",
    [string]$Email = "admin@example.com"
)

Write-Host "Obtain Let's Encrypt cert for $Domain" -ForegroundColor Cyan

if (Test-Path "C:\Program Files\win-acme\wacs.exe") {
    Write-Host "Found win-acme; launching interactive request..." -ForegroundColor Green
    & "C:\Program Files\win-acme\wacs.exe"
    exit 0
}

Write-Host "win-acme not found. Install win-acme (https://www.win-acme.com/) or run certbot on Linux." -ForegroundColor Yellow

Write-Host "Example certbot (Linux) commands:" -ForegroundColor Cyan
Write-Host "  sudo certbot certonly --standalone -d $Domain -m $Email --agree-tos --non-interactive" -ForegroundColor Gray

Write-Host "After obtaining certs, set nginx config to reference the certificate files under /etc/letsencrypt/live/$Domain/" -ForegroundColor Cyan
