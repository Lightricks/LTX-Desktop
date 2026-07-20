# Remote Backend (Experimental)

LTX Desktop can keep the Electron interface, project files, playback, and export on one computer while running the Python backend and LTX models on a separate GPU machine.

Remote compute is optional. The default `managed_local` mode continues to start and manage the bundled backend on the same computer as Electron.

## Architecture

```text
Mac or PC                         GPU machine
---------------------------      -----------------------------
Electron + React UI              Standalone FastAPI backend
Project assets and playback  ->  Media upload by opaque ID
Local thumbnails and export  <-  Generated artifact download
                                 LTX models and GPU inference
```

The standalone backend does not accept client-provided filesystem paths. Inputs are uploaded through `/api/media`, and generated files are downloaded through authenticated artifact IDs. After download, LTX Desktop imports the result into its normal local project-assets directory.

## Security model

The standalone server is a trusted, single-user service. It is not designed as a public or multi-tenant API.

- Standalone mode requires a bearer token of at least 32 characters.
- Raw filesystem-path inputs, Basic authentication, query-string authentication, and remote shutdown are disabled.
- Plain HTTP is supported only through a loopback address. Use an SSH tunnel for a private machine or an HTTPS reverse proxy for a non-loopback address.
- Do not expose port 8000 directly to an untrusted LAN or the internet.

## Start the backend on the GPU machine

Install the backend environment from the repository:

```bash
cd backend
uv sync --frozen
```

Generate an authentication token:

```bash
python -c 'import secrets; print(secrets.token_urlsafe(32))'
```

Set the standalone environment and start the server:

```bash
export LTX_DEPLOYMENT_MODE=standalone
export LTX_APP_DATA_DIR=/srv/ltx-desktop
export LTX_MODELS_DIR=/srv/ltx-models
export LTX_AUTH_TOKEN='replace-with-the-generated-token'
export LTX_BIND_HOST=127.0.0.1
export LTX_PORT=8000
export LTX_PUBLIC_BASE_URL=http://127.0.0.1:8000
export LTX_ALLOWED_ORIGINS='null,http://localhost:5173,http://127.0.0.1:5173'

uv run python ltx2_server.py
```

`LTX_MODELS_DIR` is optional. When set, it is authoritative and cannot be changed from the desktop client. Ensure the service account can read and write both data directories.

The server is ready when it prints:

```text
Server running on http://127.0.0.1:8000
```

## Connect through SSH

On the computer running LTX Desktop, forward the same local port to the GPU machine:

```bash
ssh -N -L 8000:127.0.0.1:8000 user@atom
```

Keeping port 8000 on both sides also preserves the default Hugging Face OAuth callback address.

In LTX Desktop:

1. Open **Settings → Compute**.
2. Choose **Remote machine**.
3. Enter `http://127.0.0.1:8000`.
4. Enter the bearer token from `LTX_AUTH_TOKEN`.
5. Select **Test connection**, then **Save & reconnect**.

The connection requires standalone API version 2 and verifies media-upload and artifact-download capabilities before it is saved.

## Direct HTTPS connection

For a direct Tailscale or network connection, terminate TLS in a reverse proxy and set `LTX_PUBLIC_BASE_URL` to the externally reachable HTTPS origin. The backend intentionally rejects a non-loopback `http://` public URL.

Example environment:

```bash
export LTX_BIND_HOST=127.0.0.1
export LTX_PUBLIC_BASE_URL=https://ltx-atom.example.ts.net
export LTX_ALLOWED_ORIGINS='null,http://localhost:5173,http://127.0.0.1:5173'
```

Configure the reverse proxy to preserve `Authorization`, `Content-Type`, and `Range` headers and to support large streaming request bodies.

## Storage lifecycle

- Uploaded inputs expire after 24 hours.
- Generated artifacts expire after 7 days unless the desktop client downloads and deletes them earlier.
- Partial transfers are written atomically and removed after failures.
- The Electron client validates downloaded size and SHA-256 before importing an artifact.
- Upload references are refreshed before expiry instead of being reused indefinitely.
- Video and image requests carry a client generation ID. If the connection drops after the Atom accepts a job, the desktop client keeps polling that exact job and downloads its completed artifacts after reconnection.

## Current limitations

- Remote inference does not yet remove every local media-tool dependency. Existing thumbnail and export code still uses the desktop application's packaged Python/ffmpeg tooling.
- Upload cancellation is not yet connected to generation cancellation.
- Remote compute is intended for one desktop client per backend instance; generation state is still single-client.

## Follow-up TODO

- [ ] Add phase-aware generation progress so the UI distinguishes cold model loading, text encoding, inference, export, artifact transfer, and completion instead of showing only `Generating...`.
- [ ] Add a persistent backend status bar showing the connected machine, model location, active model, and live CPU, GPU, and RAM usage.

## Return to local compute

Open **Settings → Compute**, choose **This computer**, and select **Save & reconnect**. LTX Desktop will return to its original managed-local startup and model workflow.
