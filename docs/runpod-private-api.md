# Private RunPod API

This mode runs the LTX Desktop Python backend on a RunPod Pod and lets the
desktop app use it as a private remote GPU provider.

## Step-by-step setup

### 1. Choose a Pod

Recommended for LTX 2.3:

- GPU: A100 80 GB
- RAM: 112 GB or more
- Disk: 150 GB or more for ephemeral testing
- Exposed port: HTTP `8000`

A 100 GB ephemeral disk can work for quick tests, but model caches and generated
outputs may fill it quickly. When budget allows, move `LTX_APP_DATA_DIR` to
persistent storage.

### 2. Build the container image

From the repository root, build the RunPod image:

```bash
docker build -f runpod/Dockerfile -t ltx-desktop-runpod:latest .
```

Push that image to a registry your RunPod account can pull.

### 3. Configure Pod environment variables

Set these variables on the Pod:

```bash
RUNPOD_PRIVATE_API_TOKEN=change-this-long-random-token
LTX_APP_DATA_DIR=/workspace/ltx-data
LTX_PORT=8000
```

Optional:

```bash
USE_SAGE_ATTENTION=1
```

### 4. Start the Pod

Expose HTTP port `8000` through RunPod's HTTP proxy. After the Pod starts, the
private API URL is:

```text
https://<pod-id>-8000.proxy.runpod.net
```

Check health:

```bash
curl -H "Authorization: Bearer <RUNPOD_PRIVATE_API_TOKEN>" \
  https://<pod-id>-8000.proxy.runpod.net/health
```

Expected response includes `status: "ok"` and GPU information.

### 5. Configure LTX Desktop

In LTX Desktop:

1. Open Settings.
2. In General, choose `Private RunPod API` for video generation.
3. In API Keys, set the RunPod URL, for example:

```text
https://<pod-id>-8000.proxy.runpod.net
```

4. Save the same token you set as `RUNPOD_PRIVATE_API_TOKEN`.

The desktop backend sends video and retake jobs to `/v1/*` on the private server.
Input media is uploaded to the Pod before generation; the generated video is
downloaded back into the desktop app's normal outputs folder.

### 6. Generate

Use Gen Space normally. When you select a model variation, the desktop backend
asks the Pod to download any missing required checkpoint files before inference.
This keeps first-use model downloads out of the long generation request, which
avoids RunPod HTTP proxy timeouts during model warm-up.

## Model variations

Private RunPod API mode exposes the local LTX model variations:

- `fast` — LTX 2.3 Fast (Distilled 1.1)
- `fast_legacy` — LTX 2.3 Fast (Distilled 1.0)

The selected model's transformer, upscaler, and text encoder checkpoints are
downloaded to `LTX_APP_DATA_DIR/models` on the Pod if they are missing.

## Current scope

Implemented private remote operations:

- text-to-video
- image-to-video
- audio-to-video
- retake
- prompt enhancement

The private server uses the local LTX pipeline, so it currently exposes local
model limits: `fast` model at `540p`, `720p`, and `1080p`. Official LTX API mode
remains available separately for the existing higher-resolution API specs.
