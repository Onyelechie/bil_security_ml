<#
PowerShell helper: create a locally-trusted dev certificate using mkcert.

Requirements:
- mkcert must be installed and available in PATH. Install via Chocolatey: `choco install mkcert`.
#>

param(
    [string]$Domain = "127.0.0.1",
    [string]$OutDir = "infra/certs/dev"
)

Write-Host "Creating dev cert for $Domain in $OutDir" -ForegroundColor Cyan

if (-not (Get-Command mkcert -ErrorAction SilentlyContinue)) {
    Write-Host "mkcert not found in PATH. Install from https://github.com/FiloSottile/mkcert or using Chocolatey: choco install mkcert" -ForegroundColor Yellow
    exit 1
}

New-Item -ItemType Directory -Path $OutDir -Force | Out-Null

$certFile = Join-Path $OutDir "dev-cert.pem"
$keyFile = Join-Path $OutDir "dev-key.pem"

Write-Host "Running mkcert -install (may ask for elevation)..." -ForegroundColor Green
mkcert -install

Write-Host "Generating cert for $Domain..." -ForegroundColor Green
mkcert -cert-file $certFile -key-file $keyFile $Domain

Write-Host "Created:" -ForegroundColor Cyan
Write-Host "  cert: $certFile"
Write-Host "  key:  $keyFile"

Write-Host "Note: Do NOT commit generated certs/keys. infra/certs/ is added to .gitignore." -ForegroundColor Yellow
