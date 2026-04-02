"""Export backend OpenAPI schema without running the real backend server.

This script boots the FastAPI app with fake services (same pattern used in
backend tests) and writes a deterministic OpenAPI JSON file for frontend type
generation.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any, cast

from app_factory import create_app
from app_handler import ServiceBundle
from runtime_config.model_download_specs import (
    DEFAULT_MODEL_DOWNLOAD_SPECS,
    DEFAULT_REQUIRED_MODEL_TYPES,
)
from state import RuntimeConfig, build_initial_state
from state.app_settings import AppSettings
from tests.fake_camera_motion_prompts import FAKE_CAMERA_MOTION_PROMPTS
from tests.fakes.services import FakeServices
import torch

DEFAULT_NEGATIVE_PROMPT = "openapi-export"
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parents[1] / "frontend" / "generated" / "backend-openapi.json"


def _build_schema() -> dict[str, object]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_root = Path(tmp_dir)
        app_data = tmp_root / "app_data"
        default_models_dir = app_data / "models"
        outputs_dir = tmp_root / "outputs"
        for directory in (app_data, default_models_dir, outputs_dir):
            directory.mkdir(parents=True, exist_ok=True)

        config = RuntimeConfig(
            device=torch.device("cpu"),
            default_models_dir=default_models_dir,
            model_download_specs=DEFAULT_MODEL_DOWNLOAD_SPECS,
            required_model_types=DEFAULT_REQUIRED_MODEL_TYPES,
            outputs_dir=outputs_dir,
            settings_file=app_data / "settings.json",
            ltx_api_base_url="https://api.ltx.video",
            force_api_generations=False,
            use_sage_attention=False,
            camera_motion_prompts=FAKE_CAMERA_MOTION_PROMPTS,
            default_negative_prompt=DEFAULT_NEGATIVE_PROMPT,
            dev_mode=False,
        )

        fake = FakeServices()
        bundle = ServiceBundle(
            http=cast(Any, fake.http),
            gpu_cleaner=cast(Any, fake.gpu_cleaner),
            model_downloader=cast(Any, fake.model_downloader),
            gpu_info=cast(Any, fake.gpu_info),
            video_processor=cast(Any, fake.video_processor),
            text_encoder=cast(Any, fake.text_encoder),
            task_runner=cast(Any, fake.task_runner),
            ltx_api_client=cast(Any, fake.ltx_api_client),
            zit_api_client=cast(Any, fake.zit_api_client),
            fast_video_pipeline_class=cast(Any, type(fake.fast_video_pipeline)),
            image_generation_pipeline_class=cast(Any, type(fake.image_generation_pipeline)),
            ic_lora_pipeline_class=cast(Any, type(fake.ic_lora_pipeline)),
            depth_processor_pipeline_class=cast(Any, type(fake.depth_processor_pipeline)),
            pose_processor_pipeline_class=cast(Any, type(fake.pose_processor_pipeline)),
            a2v_pipeline_class=cast(Any, type(fake.a2v_pipeline)),
            retake_pipeline_class=cast(Any, type(fake.retake_pipeline)),
        )

        handler = build_initial_state(config, AppSettings(), service_bundle=bundle)
        app = create_app(handler=handler)
        return app.openapi()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export backend OpenAPI schema to JSON.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output schema JSON path (default: {DEFAULT_OUTPUT_PATH})",
    )
    args = parser.parse_args()

    schema = _build_schema()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(schema, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote OpenAPI schema to {output}")


if __name__ == "__main__":
    main()
