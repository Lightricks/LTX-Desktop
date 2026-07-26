"""Runtime policy query handler."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from api_types import RuntimeEngine, RuntimePolicyResponse, RuntimeProvenanceItem
from runtime_config.runtime_config import RuntimeConfig
from runtime_config.runtime_policy import (
    MLX_RUNTIME_REVISION,
    MLX_RUNTIME_VERSION,
    TORCH_RUNTIME_REVISION,
    decide_fast_video_execution_mode,
)


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "not installed"


class RuntimePolicyHandler:
    def __init__(self, config: RuntimeConfig) -> None:
        self._config = config

    def get_runtime_policy(self) -> RuntimePolicyResponse:
        auto_decision = self._config.decide_fast_video_engine(use_local_text_encoding=True)
        execution_mode = decide_fast_video_execution_mode(
            auto_decision.engine,
            self._config.local_generations_mode,
            self._config.available_ram_gb,
        )
        if self._config.force_api_generations:
            auto_engine = "cloud"
            capability_engines: dict[str, RuntimeEngine] = {
                "fast_t2v_i2v": "cloud",
                "prepared_text_embeddings": "cloud",
                "audio_to_video": "cloud",
                "retake": "cloud",
                "extend": "cloud",
                "ic_lora": "cloud",
                "image_generation": "cloud",
            }
        else:
            auto_engine = auto_decision.engine
            capability_engines = {
                "fast_t2v_i2v": auto_decision.engine,
                "prepared_text_embeddings": "torch",
                "audio_to_video": "torch",
                "retake": "torch",
                "extend": "torch",
                "ic_lora": "torch",
                "image_generation": "torch",
            }

        quality_warning = None
        if self._config.mlx_model_variant == "q8":
            quality_warning = (
                "MLX q8 is an expert-only option and is not auto-selected because "
                "quality qualification found measurable video/audio loss."
            )

        return RuntimePolicyResponse(
            force_api_generations=self._config.force_api_generations,
            fast_video_engine_preference=self._config.fast_video_engine_preference,
            auto_fast_video_engine=auto_engine,
            auto_selection_reason=auto_decision.reason,
            execution_mode=execution_mode,
            automatic_tiling=execution_mode == "low_ram",
            mlx_model_source=self._config.mlx_model_source,
            mlx_model_variant=self._config.mlx_model_variant,
            quality_warning=quality_warning,
            capability_engines=capability_engines,
            provenance=[
                RuntimeProvenanceItem(
                    component="ltx-pipelines",
                    version=_package_version("ltx-pipelines"),
                    revision=TORCH_RUNTIME_REVISION,
                    source="https://github.com/Lightricks/LTX-2",
                ),
                RuntimeProvenanceItem(
                    component="torch",
                    version=_package_version("torch"),
                    source="https://pytorch.org",
                ),
                RuntimeProvenanceItem(
                    component=(
                        "ltx-pipelines-mlx sidecar (actual, dirty)"
                        if self._config.mlx_runtime_dirty
                        else "ltx-pipelines-mlx sidecar (actual)"
                    ),
                    version=self._config.mlx_runtime_version,
                    revision=self._config.mlx_runtime_revision,
                    source=self._config.mlx_runtime_source,
                ),
                RuntimeProvenanceItem(
                    component="ltx-pipelines-mlx compatibility target",
                    version=MLX_RUNTIME_VERSION,
                    revision=MLX_RUNTIME_REVISION,
                    source="https://github.com/dgrauet/ltx-2-mlx",
                ),
                RuntimeProvenanceItem(
                    component="ltx-core-mlx sidecar dependency",
                    version=self._config.mlx_core_version or "not reported",
                    source=self._config.mlx_runtime_source,
                ),
                RuntimeProvenanceItem(
                    component="mlx sidecar framework",
                    version=self._config.mlx_framework_version or "not reported",
                    source=self._config.mlx_runtime_source,
                ),
            ],
        )
