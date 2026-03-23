**Quick Start**

Purpose: get the server running quickly on a fresh Windows machine without installing heavy ML packages.

Prerequisites:
- Python 3.11+ (the environment used here is Python 3.13; ensure `python` is on PATH)

Recommended quick start (Windows PowerShell):

1. From the project root create and activate a virtual environment:

```powershell
python -m venv .venv
& .venv\Scripts\Activate.ps1
```

2. Install the minimal server dependencies:

```powershell
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-server.txt
```

3. Start the server:

```powershell
python -m uvicorn --app-dir src server.main:app --reload --host 127.0.0.1 --port 8000
```

Notes:
- `requirements-server.txt` contains a minimal set of dependencies required to run the HTTP/WebSocket APIs and admin endpoints. The full `requirements.txt` contains heavy ML packages such as `torch` and `ultralytics` and is not installed by the quick-start script.
- If you deploy behind nginx and depend on `ProxyHeadersMiddleware`, ensure `starlette==0.27.0` or later is installed; the quick-start pins `starlette==0.27.0` to avoid runtime incompatibilities.

## Edge Agent Bring-Up

Before starting an edge PC against this server, provision these values onto the edge machine:

- `SERVER_BASE_URL`
- `SITE_ID`
- `SITE_NAME`
- `EDGE_PC_ID`
- `DEVICE_ID`
- `DEVICE_PRIVATE_KEY_B64`

Requirements:

- `DEVICE_ID` must equal `EDGE_PC_ID`.
- The matching public key must already be enrolled on the server through `/api/devices/enroll`.

Operationally, the flow is:

1. Generate the keypair for the edge.
2. Keep the private key on the edge PC.
3. Enroll the public key on the server as that edge id.
4. Configure the edge `.env`.
5. Start the edge agent.

Minimum edge `.env` example:

```env
SERVER_BASE_URL=http://127.0.0.1:8000
SITE_ID=site_demo
SITE_NAME=Demo Site
EDGE_PC_ID=edge-demo-1
DEVICE_ID=edge-demo-1
DEVICE_PRIVATE_KEY_B64=<BASE64_ED25519_PRIVATE_KEY>
```

Start the edge agent:

```powershell
$env:PYTHONPATH="$PWD\src"
python -m edge_agent.main --http-serve
```

What success looks like:

- `GET /api/heartbeat` on the server shows the edge id after the first signed heartbeat.
- alerts sent by `edge_agent.sender.ServerSender` are accepted and visible via `GET /api/alerts`.

For the full enrollment/provisioning flow, see [AUTHENTICATION.md](C:/Users/ebere/Documents/bil_security_ml/docs/AUTHENTICATION.md).
