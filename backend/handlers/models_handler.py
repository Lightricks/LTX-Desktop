"""Checkpoint recommendation and filesystem model state helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from threading import RLock
from typing import TYPE_CHECKING

from _routes._errors import HTTPError
from api_types import (
    CheckpointDescriptor,
    CheckpointRole,
    DescribeCheckpointsResponse,
    ImageGenRecommendationResponse,
    InstalledModelResponse,
    InstalledModelsResponse,
    LtxDownloadRecommendationResponse,
    LtxIcLoraRecommendationResponse,
    LtxModelVersionItem,
    LtxModelVersionsResponse,
    LtxOkRecommendationResponse,
    LtxRecommendationResponse,
    LtxUpgradeRecommendationResponse,
    LTXLocalModelId,
    ModelCheckpointID,
    TextEncoderRecommendationResponse,
)
from handlers.base import StateHandlerBase
from handlers.settings_handler import SettingsHandler
from runtime_config.models_scanner import scan_models_dir
from runtime_config.ltx_capabilities import local_caps, supports
from runtime_config.model_download_specs import (
    ALL_LTX_LOCAL_MODEL_IDS,
    ALL_MODEL_CP_IDS,
    DEPTH_PROCESSOR_CP_ID,
    IMG_GEN_MODEL_CP_ID,
    LTXLocalModelRelevant,
    get_downloaded_ltx_model_id,
    get_ic_loras_cp_ids,
    get_latest_ltx_model_id,
    get_ltx_cps,
    get_ltx_model_cp_ids,
    get_ltx_model_id_for_cp,
    get_ltx_model_spec,
    get_model_cp_spec,
    is_cp_downloaded,
    resolve_active_ltx_model_id,
    resolve_downloaded_prompt_enhancer_cp,
    selected_video_vae_cp,
    unused_video_vae_cp,
    delete_cp_path,
)

if TYPE_CHECKING:
    from runtime_config.runtime_config import RuntimeConfig
    from state.app_state_types import AppState

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ResolvedUpgradeDownload:
    current_model_id: LTXLocalModelId
    target_model_id: LTXLocalModelId
    cp_ids: tuple[ModelCheckpointID, ...]


class ModelsHandler(StateHandlerBase):
    def __init__(
        self,
        state: AppState,
        lock: RLock,
        config: RuntimeConfig,
        settings_handler: SettingsHandler,
    ) -> None:
        super().__init__(state, lock, config)
        self._settings = settings_handler

    def _ordered_cp_ids(self, cp_ids: set[ModelCheckpointID]) -> list[ModelCheckpointID]:
        return [cp_id for cp_id in ALL_MODEL_CP_IDS if cp_id in cp_ids]

    def _ensure_local_model_mode(self) -> None:
        if self.config.force_api_generations:
            raise HTTPError(409, "LOCAL_MODEL_RECOMMENDATIONS_DISABLED_IN_FORCE_API_MODE")

    def _current_downloaded_ltx_model_id(self) -> LTXLocalModelId | None:
        # Upgrade / "do we already have latest" still keys off whatever transformers are on disk
        # (newest-first), not the user-selected active version.
        return get_downloaded_ltx_model_id(self.models_dir)

    def _current_active_ltx_model_id(self) -> LTXLocalModelId | None:
        return resolve_active_ltx_model_id(
            self.models_dir, self.state.app_settings.active_ltx_model_id
        )

    def _has_api_key(self) -> bool:
        return bool(self.state.app_settings.ltx_api_key.strip())

    def is_cp_downloaded(self, cp_id: ModelCheckpointID) -> bool:
        return is_cp_downloaded(self.models_dir, cp_id)

    def get_downloaded_checkpoints(self) -> set[ModelCheckpointID]:
        return {cp_id for cp_id in ALL_MODEL_CP_IDS if self.is_cp_downloaded(cp_id)}

    def _cp_role(self, cp_id: ModelCheckpointID) -> CheckpointRole:
        # Walk every version, not only latest: otherwise a 2.5 VAE (or an older upscaler/TE)
        # falls through to "support" and first-run tooltips call it a depth/edges/pose model.
        for model_id in ALL_LTX_LOCAL_MODEL_IDS:
            spec = get_ltx_model_spec(model_id)
            if cp_id == spec.model_cp:
                return "base"
            if cp_id == spec.upscale_cp:
                return "upscaler"
            if cp_id == spec.text_encoder_cp:
                return "text_encoder"
            if cp_id in (spec.video_vae_cp, spec.video_vae_conv_cp, spec.audio_vae_cp):
                return "vae"
            if cp_id == spec.duration_head_cp:
                return "support"
        if cp_id == IMG_GEN_MODEL_CP_ID:
            return "image"
        return "support"

    def describe_checkpoints(self, cp_ids: list[ModelCheckpointID]) -> DescribeCheckpointsResponse:
        # Static spec metadata for a set of checkpoints, used by the first-run download
        # list and (later) the model version picker. Order follows the canonical cp order.
        ordered = self._ordered_cp_ids(set(cp_ids))
        descriptors: list[CheckpointDescriptor] = []
        for cp_id in ordered:
            spec = get_model_cp_spec(cp_id)
            role = self._cp_role(cp_id)
            descriptors.append(
                CheckpointDescriptor(
                    cp_id=cp_id,
                    name=spec.description,
                    role=role,
                    size_bytes=spec.expected_size_bytes,
                    downloaded=self.is_cp_downloaded(cp_id),
                )
            )
        return DescribeCheckpointsResponse(checkpoints=descriptors)

    def _use_conv_vae(self) -> bool:
        from state.app_settings import resolved_use_conv_vae

        return resolved_use_conv_vae(self.state.app_settings)

    def _get_required_ltx_cp_ids(self, model_id: LTXLocalModelId) -> set[ModelCheckpointID]:
        spec = get_ltx_model_spec(model_id)
        required: set[ModelCheckpointID] = {spec.model_cp, spec.upscale_cp}
        selected_vae = selected_video_vae_cp(spec, use_conv_vae=self._use_conv_vae())
        if selected_vae is not None:
            required.add(selected_vae)
        # Conv VAE is always part of the 2.5 download set so Fast decode can be turned on
        # later without a hidden extra fetch. DiffVAE stays selected-only (Mac default).
        if spec.video_vae_conv_cp is not None:
            required.add(spec.video_vae_conv_cp)
        if spec.audio_vae_cp is not None:
            required.add(spec.audio_vae_cp)
        if spec.duration_head_cp is not None:
            required.add(spec.duration_head_cp)
        if not self._has_api_key() or not spec.supports_api_text_encoding:
            required.add(spec.text_encoder_cp)
        return required

    def _get_optional_ltx_cp_ids(self, model_id: LTXLocalModelId) -> set[ModelCheckpointID]:
        """Missing checkpoints the user can still choose to download.

        Includes the unused 2.5 DiffVAE when Fast decode is on, and any text encoder an
        LTX API key excused. Never overlaps the required set.
        """
        spec = get_ltx_model_spec(model_id)
        optional: set[ModelCheckpointID] = set()
        unused_vae = unused_video_vae_cp(spec, use_conv_vae=self._use_conv_vae())
        if unused_vae is not None:
            optional.add(unused_vae)
        if self._has_api_key() and spec.supports_api_text_encoding:
            optional.add(spec.text_encoder_cp)
        optional -= self._get_required_ltx_cp_ids(model_id)
        return self._get_missing_cp_ids(optional)

    def _get_missing_cp_ids(self, cp_ids: set[ModelCheckpointID]) -> set[ModelCheckpointID]:
        return {cp_id for cp_id in cp_ids if not self.is_cp_downloaded(cp_id)}

    def _get_upgrade_message(self, current_model_id: LTXLocalModelId, target_model_id: LTXLocalModelId) -> str | None:
        relevance = get_ltx_model_spec(target_model_id).relevance
        if not isinstance(relevance, LTXLocalModelRelevant):
            return None
        return relevance.upgrade_messages.get(current_model_id)

    def _maybe_add_upgrade_companion(
        self,
        cp_ids: set[ModelCheckpointID],
        *,
        current_cp: ModelCheckpointID | None,
        target_cp: ModelCheckpointID | None,
    ) -> None:
        if (
            target_cp is not None
            and current_cp != target_cp
            and (current_cp is None or self.is_cp_downloaded(current_cp))
            and not self.is_cp_downloaded(target_cp)
        ):
            cp_ids.add(target_cp)

    def _get_upgrade_dependency_downloads(
        self,
        current_model_id: LTXLocalModelId,
        target_model_id: LTXLocalModelId,
    ) -> set[ModelCheckpointID]:
        current_spec = get_ltx_model_spec(current_model_id)
        target_spec = get_ltx_model_spec(target_model_id)
        cp_ids: set[ModelCheckpointID] = {target_spec.model_cp}

        self._maybe_add_upgrade_companion(
            cp_ids, current_cp=current_spec.upscale_cp, target_cp=target_spec.upscale_cp
        )
        # Same rule as a fresh install: an LTX API key that can encode this version makes the
        # text encoder optional, so don't force it onto the upgrade download.
        if not self._has_api_key() or not target_spec.supports_api_text_encoding:
            self._maybe_add_upgrade_companion(
                cp_ids, current_cp=current_spec.text_encoder_cp, target_cp=target_spec.text_encoder_cp
            )
        self._maybe_add_upgrade_companion(
            cp_ids, current_cp=current_spec.video_vae_cp, target_cp=target_spec.video_vae_cp
        )
        self._maybe_add_upgrade_companion(
            cp_ids, current_cp=current_spec.video_vae_conv_cp, target_cp=target_spec.video_vae_conv_cp
        )
        self._maybe_add_upgrade_companion(
            cp_ids, current_cp=current_spec.audio_vae_cp, target_cp=target_spec.audio_vae_cp
        )
        self._maybe_add_upgrade_companion(
            cp_ids, current_cp=current_spec.duration_head_cp, target_cp=target_spec.duration_head_cp
        )

        current_ic = current_spec.ic_loras_spec
        target_ic = target_spec.ic_loras_spec
        if current_ic is not None and target_ic is not None:
            ic_lora_pairs: tuple[tuple[ModelCheckpointID, ModelCheckpointID], ...] = (
                (current_ic.depth_cp, target_ic.depth_cp),
                (current_ic.canny_cp, target_ic.canny_cp),
                (current_ic.pose_cp, target_ic.pose_cp),
            )
            for current_cp_id, target_cp_id in ic_lora_pairs:
                self._maybe_add_upgrade_companion(cp_ids, current_cp=current_cp_id, target_cp=target_cp_id)

        return cp_ids

    def _get_upgrade_delete_cp_ids(
        self,
        current_model_id: LTXLocalModelId,
        target_model_id: LTXLocalModelId,
    ) -> set[ModelCheckpointID]:
        current_cp_ids = set(get_ltx_model_cp_ids(current_model_id))
        target_cp_ids = set(get_ltx_model_cp_ids(target_model_id))
        return {
            cp_id
            for cp_id in current_cp_ids - target_cp_ids
            if self.is_cp_downloaded(cp_id)
        }

    def get_ltx_recommendation(self) -> LtxRecommendationResponse:
        self._ensure_local_model_mode()

        current_model_id = self._current_downloaded_ltx_model_id()
        latest_model_id = get_latest_ltx_model_id()

        if current_model_id is None:
            cps_to_download = self._ordered_cp_ids(
                self._get_missing_cp_ids(self._get_required_ltx_cp_ids(latest_model_id))
            )
            return LtxDownloadRecommendationResponse(
                status="download",
                cps_to_download=cps_to_download,
                optional_cp_ids=self._ordered_cp_ids(self._get_optional_ltx_cp_ids(latest_model_id)),
            )

        # A required checkpoint for the current model can be missing even when its base
        # transformer is present — e.g. a hotfixed shared companion (the 2x upscaler) that
        # superseded the version already on disk. Surface that download before offering any
        # base upgrade: the current setup needs it regardless of whether the user upgrades,
        # and routing it through the 'download' status lets the missing-models gate prompt it.
        missing_current = self._ordered_cp_ids(
            self._get_missing_cp_ids(self._get_required_ltx_cp_ids(current_model_id))
        )
        if missing_current:
            return LtxDownloadRecommendationResponse(
                status="download",
                cps_to_download=missing_current,
                optional_cp_ids=self._ordered_cp_ids(self._get_optional_ltx_cp_ids(current_model_id)),
            )

        if current_model_id == latest_model_id:
            return LtxOkRecommendationResponse(status="ok")

        cps_to_download = self._ordered_cp_ids(
            self._get_upgrade_dependency_downloads(current_model_id, latest_model_id)
        )
        cps_to_delete = self._ordered_cp_ids(
            self._get_upgrade_delete_cp_ids(current_model_id, latest_model_id)
        )
        current_spec = get_ltx_model_spec(current_model_id)
        target_spec = get_ltx_model_spec(latest_model_id)
        loses_control = (
            current_spec.ic_loras_spec is not None
            and target_spec.ic_loras_spec is None
            and bool(set(cps_to_delete) & set(get_ic_loras_cp_ids(current_spec.ic_loras_spec)))
        )
        return LtxUpgradeRecommendationResponse(
            status="upgrade",
            ltx_model_id=latest_model_id,
            upgrade_message=self._get_upgrade_message(current_model_id, latest_model_id),
            cps_to_download=cps_to_download,
            cps_to_delete=cps_to_delete,
            loses_built_in_control=loses_control,
        )

    def get_img_gen_recommendation(self) -> ImageGenRecommendationResponse:
        self._ensure_local_model_mode()
        cp_to_download = None if self.is_cp_downloaded(IMG_GEN_MODEL_CP_ID) else IMG_GEN_MODEL_CP_ID
        return ImageGenRecommendationResponse(cp_to_download=cp_to_download)

    def list_installed_models(self, model_type: str | None) -> InstalledModelsResponse:
        # "lora" -> regular LoRAs only (IC-LoRAs excluded; they need a reference video),
        # "ic-lora" -> IC-LoRAs only, None -> all installed models. Reject typos so a
        # bad ?type can't silently leak ic-loras into the lora picker (or vice versa).
        if model_type is not None and model_type not in ("lora", "ic-lora"):
            raise HTTPError(400, f"Unknown model type: {model_type}")
        entries = scan_models_dir(self.models_dir, lora_only=model_type in ("lora", "ic-lora"))
        if model_type == "lora":
            entries = [e for e in entries if e.is_lora and not e.is_ic_lora]
        elif model_type == "ic-lora":
            entries = [e for e in entries if e.is_ic_lora]
        return InstalledModelsResponse(
            models=[
                InstalledModelResponse(
                    path=str(e.path),
                    name=e.name,
                    kind=e.kind,
                    size_bytes=e.size_bytes,
                    is_lora=e.is_lora,
                    is_ic_lora=e.is_ic_lora,
                )
                for e in entries
            ]
        )

    def _require_downloaded_ltx_model_id(self) -> LTXLocalModelId:
        model_id = self._current_downloaded_ltx_model_id()
        if model_id is None:
            raise HTTPError(409, "NO_DOWNLOADED_LTX_MODEL")
        return model_id

    def _require_active_ltx_model_id(self) -> LTXLocalModelId:
        model_id = self._current_active_ltx_model_id()
        if model_id is None:
            raise HTTPError(409, "NO_DOWNLOADED_LTX_MODEL")
        return model_id

    def get_ltx_ic_lora_recommendation(self) -> LtxIcLoraRecommendationResponse:
        self._ensure_local_model_mode()
        model_id = self._require_active_ltx_model_id()
        spec = get_ltx_model_spec(model_id)
        if not supports(local_caps(model_id), "ic_lora") or spec.ic_loras_spec is None:
            return LtxIcLoraRecommendationResponse(cps_to_download=[], supported=False)
        required_cp_ids: set[ModelCheckpointID] = set(get_ic_loras_cp_ids(spec.ic_loras_spec))
        required_cp_ids.add(DEPTH_PROCESSOR_CP_ID)
        cp_ids = self._get_missing_cp_ids(required_cp_ids)
        return LtxIcLoraRecommendationResponse(
            cps_to_download=self._ordered_cp_ids(cp_ids),
            supported=True,
        )

    def get_text_encoder_recommendation(self) -> TextEncoderRecommendationResponse:
        self._ensure_local_model_mode()
        model_id = self._require_active_ltx_model_id()
        ltx_spec = get_ltx_model_spec(model_id)
        cp_id = ltx_spec.text_encoder_cp
        spec = get_model_cp_spec(cp_id)
        enhancer_cp = ltx_spec.prompt_enhancer_cp
        active_enhancer_cp = resolve_downloaded_prompt_enhancer_cp(self.models_dir, ltx_spec)
        return TextEncoderRecommendationResponse(
            cp_to_download=None if self.is_cp_downloaded(cp_id) else cp_id,
            expected_size_bytes=spec.expected_size_bytes,
            expected_size_gb=round(spec.expected_size_bytes / (1024**3), 1),
            api_encoding_supported=ltx_spec.supports_api_text_encoding,
            ltx_version_label=ltx_spec.version_label,
            local_enhancement_supported=active_enhancer_cp is not None,
            local_enhancer_cp=enhancer_cp,
            local_enhancer_expected_size_gb=(
                None if enhancer_cp is None
                else round(get_model_cp_spec(enhancer_cp).expected_size_bytes / (1024**3), 1)
            ),
            active_local_enhancer_cp=active_enhancer_cp,
        )

    def resolve_upgrade_download(self, requested_cp_ids: set[ModelCheckpointID]) -> ResolvedUpgradeDownload:
        self._ensure_local_model_mode()

        current_model_id = self._current_downloaded_ltx_model_id()
        if current_model_id is None:
            raise HTTPError(409, "NO_DOWNLOADED_LTX_MODEL")

        latest_model_id = get_latest_ltx_model_id()
        if current_model_id == latest_model_id:
            raise HTTPError(409, "ALREADY_ON_LATEST_LTX_MODEL")

        requested_ltx_cp_ids = requested_cp_ids & get_ltx_cps()
        if len(requested_ltx_cp_ids) != 1:
            raise HTTPError(409, "INVALID_UPGRADE_REQUEST")

        target_model_cp_id = next(iter(requested_ltx_cp_ids))
        target_model_id = get_ltx_model_id_for_cp(target_model_cp_id)
        if target_model_id is None:
            raise HTTPError(500, "INVALID_LTX_MODEL_CONFIG")

        if target_model_id != latest_model_id:
            raise HTTPError(409, "INVALID_UPGRADE_REQUEST")
        target_relevance = get_ltx_model_spec(target_model_id).relevance
        if not isinstance(target_relevance, LTXLocalModelRelevant):
            raise HTTPError(500, "INVALID_LTX_MODEL_CONFIG")

        recommendation = self.get_ltx_recommendation()
        if not isinstance(recommendation, LtxUpgradeRecommendationResponse):
            raise HTTPError(409, "INVALID_UPGRADE_REQUEST")

        expected_cp_ids = set(recommendation.cps_to_download)
        if requested_cp_ids != expected_cp_ids:
            raise HTTPError(409, "INVALID_UPGRADE_REQUEST")

        return ResolvedUpgradeDownload(
            current_model_id=current_model_id,
            target_model_id=target_model_id,
            cp_ids=tuple(self._ordered_cp_ids(expected_cp_ids)),
        )

    def get_protected_cp_ids(self) -> set[ModelCheckpointID]:
        active_model_id = resolve_active_ltx_model_id(
            self.models_dir, self.state.app_settings.active_ltx_model_id
        )
        if active_model_id is None:
            return set()
        return set(get_ltx_model_cp_ids(active_model_id))

    def delete_checkpoints(self, cp_ids: set[ModelCheckpointID]) -> None:
        protected = self.get_protected_cp_ids()
        if cp_ids & protected:
            raise HTTPError(409, "DELETE_PROTECTED_CHECKPOINT")
        for cp_id in self._ordered_cp_ids(cp_ids):
            logger.info("Deleting checkpoint %s from %s", cp_id, self.models_dir)
            delete_cp_path(self.models_dir, cp_id)

    def list_ltx_versions(self) -> LtxModelVersionsResponse:
        self._ensure_local_model_mode()
        latest = get_latest_ltx_model_id()
        active = resolve_active_ltx_model_id(self.models_dir, self.state.app_settings.active_ltx_model_id)
        items: list[LtxModelVersionItem] = []
        for model_id in ALL_LTX_LOCAL_MODEL_IDS:
            spec = get_ltx_model_spec(model_id)
            missing = self._get_missing_cp_ids(self._get_required_ltx_cp_ids(model_id))
            # "installed" must mean runnable (transformer + companions), matching the bundle
            # set_active requires — otherwise a partial install reports installed yet 409s on
            # activation, and BaseModelSection hides the Download button that would repair it.
            installed = not missing
            # Sum the required bundle so Settings doesn't imply "42 GB" when VAEs + upscaler
            # (and the TE when no API key covers it) are also part of the install.
            size_bytes = sum(
                get_model_cp_spec(cp_id).expected_size_bytes
                for cp_id in self._get_required_ltx_cp_ids(model_id)
            )
            items.append(
                LtxModelVersionItem(
                    model_id=model_id,
                    label=spec.version_label,
                    model_cp=spec.model_cp,
                    size_bytes=size_bytes,
                    installed=installed,
                    active=model_id == active,
                    is_newest=model_id == latest,
                    cps_to_download=self._ordered_cp_ids(missing),
                )
            )
        return LtxModelVersionsResponse(versions=items)

    def set_active_ltx_model(self, model_id: LTXLocalModelId) -> None:
        self._ensure_local_model_mode()
        # Require the version's full required bundle (transformer + companions per the
        # API-key rules), not just the transformer — otherwise the user could activate a
        # version that can't actually generate (missing upscaler / text encoder).
        if self._get_missing_cp_ids(self._get_required_ltx_cp_ids(model_id)):
            raise HTTPError(409, "LTX_MODEL_NOT_INSTALLED")
        self._settings.set_active_ltx_model_id(model_id)
