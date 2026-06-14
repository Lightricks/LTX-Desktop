"""Canonical app settings schema and patch models."""

from __future__ import annotations

from typing import Any, Literal, TypeGuard, TypeVar, cast, get_args

from pydantic import BaseModel, ConfigDict, Field, create_model, field_validator


def _to_camel_case(field_name: str) -> str:
    special_aliases = {
        "prompt_enhancer_enabled_t2v": "promptEnhancerEnabledT2V",
        "prompt_enhancer_enabled_i2v": "promptEnhancerEnabledI2V",
    }
    if field_name in special_aliases:
        return special_aliases[field_name]

    head, *tail = field_name.split("_")
    return head + "".join(part.title() for part in tail)


def _clamp_int(value: Any, minimum: int, maximum: int, default: int) -> int:
    if value is None:
        return default

    parsed = int(value)
    return max(minimum, min(maximum, parsed))


class SettingsBaseModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=_to_camel_case,
        populate_by_name=True,
        validate_assignment=True,
        extra="ignore",
    )


class SettingsPatchModel(SettingsBaseModel):
    model_config = ConfigDict(
        alias_generator=_to_camel_case,
        populate_by_name=True,
        validate_assignment=True,
        extra="forbid",
    )


class AppSettings(SettingsBaseModel):
    use_torch_compile: bool = False
    ltx_api_key: str = ""
    user_prefers_ltx_api_video_generations: bool = False
    video_generation_provider: Literal["local", "ltx_api", "runpod"] = "local"
    runpod_api_url: str = ""
    runpod_api_token: str = ""
    fal_api_key: str = ""
    use_local_text_encoder: bool = False
    prompt_cache_size: int = 100
    prompt_enhancer_enabled_t2v: bool = True
    prompt_enhancer_enabled_i2v: bool = False
    gemini_api_key: str = ""
    seed_locked: bool = False
    locked_seed: int = 42
    models_dir: str = ""

    @field_validator("prompt_cache_size", mode="before")
    @classmethod
    def _clamp_prompt_cache_size(cls, value: Any) -> int:
        return _clamp_int(value, minimum=0, maximum=1000, default=100)

    @field_validator("locked_seed", mode="before")
    @classmethod
    def _clamp_locked_seed(cls, value: Any) -> int:
        return _clamp_int(value, minimum=0, maximum=2_147_483_647, default=42)


SettingsModelT = TypeVar("SettingsModelT", bound=SettingsBaseModel)
_PARTIAL_MODEL_CACHE: dict[type[SettingsBaseModel], type[SettingsPatchModel]] = {}


def _wrap_optional(annotation: Any) -> Any:
    if type(None) in get_args(annotation):
        return annotation
    return annotation | None


def _to_partial_annotation(annotation: Any) -> Any:
    if _is_settings_model_annotation(annotation):
        return make_partial_model(annotation)
    return annotation


def make_partial_model(model: type[SettingsModelT]) -> type[SettingsPatchModel]:
    cached = _PARTIAL_MODEL_CACHE.get(model)
    if cached is not None:
        return cached

    fields: dict[str, tuple[Any, Any]] = {}
    for field_name, field_info in model.model_fields.items():
        partial_annotation = _wrap_optional(_to_partial_annotation(field_info.annotation))
        fields[field_name] = (partial_annotation, Field(default=None))

    partial_model = create_model(
        f"{model.__name__}Patch",
        __base__=SettingsPatchModel,
        **cast(Any, fields),
    )

    _PARTIAL_MODEL_CACHE[model] = partial_model
    return partial_model


def _is_settings_model_annotation(annotation: object) -> TypeGuard[type[SettingsBaseModel]]:
    return isinstance(annotation, type) and issubclass(annotation, SettingsBaseModel)


AppSettingsPatch = make_partial_model(AppSettings)
UpdateSettingsRequest = AppSettingsPatch


class SettingsResponse(SettingsBaseModel):
    use_torch_compile: bool = False
    has_ltx_api_key: bool = False
    user_prefers_ltx_api_video_generations: bool = False
    video_generation_provider: Literal["local", "ltx_api", "runpod"] = "local"
    runpod_api_url: str = ""
    has_runpod_api_token: bool = False
    has_fal_api_key: bool = False
    use_local_text_encoder: bool = False
    prompt_cache_size: int = 100
    prompt_enhancer_enabled_t2v: bool = True
    prompt_enhancer_enabled_i2v: bool = False
    has_gemini_api_key: bool = False
    seed_locked: bool = False
    locked_seed: int = 42
    models_dir: str = ""


def to_settings_response(settings: AppSettings) -> SettingsResponse:
    data = settings.model_dump(by_alias=False)
    ltx_key = data.pop("ltx_api_key", "")
    runpod_token = data.pop("runpod_api_token", "")
    fal_key = data.pop("fal_api_key", "")
    gemini_key = data.pop("gemini_api_key", "")
    data["has_ltx_api_key"] = bool(ltx_key)
    data["has_runpod_api_token"] = bool(runpod_token)
    data["has_fal_api_key"] = bool(fal_key)
    data["has_gemini_api_key"] = bool(gemini_key)
    # models_dir passes through as-is (not secret)
    return SettingsResponse.model_validate(data)


VideoGenerationProvider = Literal["local", "ltx_api", "runpod"]


def resolve_video_generation_provider(*, force_api_generations: bool, settings: AppSettings) -> VideoGenerationProvider:
    configured_provider = settings.video_generation_provider
    if configured_provider == "runpod":
        return "runpod"
    if configured_provider == "ltx_api":
        return "ltx_api"

    has_ltx_api_key = bool(settings.ltx_api_key.strip())
    if force_api_generations:
        return "ltx_api"
    if settings.user_prefers_ltx_api_video_generations and has_ltx_api_key:
        return "ltx_api"
    return "local"


def should_video_generate_with_ltx_api(*, force_api_generations: bool, settings: AppSettings) -> bool:
    return resolve_video_generation_provider(
        force_api_generations=force_api_generations,
        settings=settings,
    ) == "ltx_api"


def should_video_generate_with_remote_api(*, force_api_generations: bool, settings: AppSettings) -> bool:
    return resolve_video_generation_provider(
        force_api_generations=force_api_generations,
        settings=settings,
    ) in ("ltx_api", "runpod")
