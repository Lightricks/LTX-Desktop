"""RunPod-facing entrypoint for the private LTX Desktop backend."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

import uvicorn

DEFAULT_DATA_DIR = "/workspace/ltx-data"
DEFAULT_PORT = "8000"

os.environ.setdefault("LTX_APP_DATA_DIR", DEFAULT_DATA_DIR)
os.environ.setdefault("LTX_PORT", DEFAULT_PORT)
os.environ.setdefault("LTX_VIDEO_ONLY_SERVER", "1")

private_token = os.environ.get("RUNPOD_PRIVATE_API_TOKEN") or os.environ.get("LTX_AUTH_TOKEN")
if not private_token:
    raise RuntimeError(
        "RUNPOD_PRIVATE_API_TOKEN must be set. The desktop app uses this as the bearer token for your private server."
    )
os.environ["LTX_AUTH_TOKEN"] = private_token

from ltx2_server import app, log_hardware_info  # noqa: E402

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    data_dir = Path(os.environ["LTX_APP_DATA_DIR"])
    data_dir.mkdir(parents=True, exist_ok=True)
    port = int(os.environ.get("LTX_PORT", DEFAULT_PORT))

    logger.info("=" * 60)
    logger.info("LTX Desktop Private RunPod Server")
    logger.info("Data directory: %s", data_dir)
    log_hardware_info()
    logger.info("=" * 60)

    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
        access_log=True,
    )
    asyncio.run(uvicorn.Server(config).serve())
