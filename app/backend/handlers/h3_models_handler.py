"""Manage the local MiniMax H3 recipes in the selected Comfy worker's model folders."""

from __future__ import annotations

import time
from collections.abc import Callable
from functools import cache
from pathlib import Path
from threading import Lock

import httpx
from pydantic import BaseModel

from _routes._errors import HTTPError
from handlers.comfy_models_handler import (
    ComfyBundleInstaller,
    ComfyBundleStatus,
    ComfyInstalledFile,
    locate_comfy_weight,
)
from runtime_config.community_runtime import local_video_backend
from services.amd_comfy_client import AmdComfyClient
from services.amd_comfy_models import ComfyWeight
from services.amd_h3_models import (
    h3_model_recipes,
    H3Variant,
    acknowledge_h3_license,
    h3_license_acknowledged,
)
from services.model_downloader import ModelDownloader
from services.task_runner import TaskRunner

_LABELS: dict[H3Variant, str] = {
    "h3-fl2va": "MiniMax H3 FL2VA",
    "h3-fl2va-distill": "MiniMax H3 FL2VA 8-step Turbo",
    "h3-ref2va": "MiniMax H3 Ref2VA",
}
_FL = "MiniMaxH3ImageToVideo"
_REF = "MiniMaxH3ReferenceToVideo"


class H3ModelsStatus(BaseModel):
    variants: dict[H3Variant, ComfyBundleStatus]
    licenseAcknowledged: bool


def running_h3_folders() -> tuple[dict[str, list[str]], set[str]]:
    worker = AmdComfyClient.discover()
    with httpx.Client(
        base_url=worker.base_url, timeout=8, trust_env=False, follow_redirects=False
    ) as http:
        nodes: set[str] = set()
        for node in (_FL, _REF):
            response = http.get(f"/object_info/{node}")
            response.raise_for_status()
            if node in response.json():
                nodes.add(node)
    return worker.get_folder_paths(), nodes


class H3ModelsHandler:
    def __init__(
        self,
        app_data_dir: Path,
        token: Callable[[], str | None],
        downloader: ModelDownloader,
        tasks: TaskRunner,
        folders: Callable[
            [], tuple[dict[str, list[str]], set[str]]
        ] = running_h3_folders,
        recipes: dict[H3Variant, tuple[ComfyWeight, ...]] | None = None,
    ) -> None:
        self._data = app_data_dir
        self._folders = folders
        self._lock = Lock()
        self._cached: tuple[float, dict[H3Variant, ComfyBundleStatus]] | None = None
        if recipes is None:
            recipes = h3_model_recipes(metal=local_video_backend() == "mps")
        self._recipes: dict[H3Variant, ComfyBundleInstaller] = {
            key: ComfyBundleInstaller(
                _LABELS[key],
                weights,
                token,
                None,
                downloader,
                tasks,
                self._invalidate,
                shared_files=True,
            )
            for key, weights in recipes.items()
        }

    def _invalidate(self) -> None:
        with self._lock:
            self._cached = None

    def license_acknowledged(self) -> bool:
        return h3_license_acknowledged(self._data)

    @staticmethod
    def _require_consent(accepted: bool) -> None:
        if not accepted:
            raise HTTPError(
                403,
                "Confirm the MiniMax H3 license or your separate authorization to use these models.",
                code="H3_LICENSE_ACK_REQUIRED",
            )

    def acknowledge(self, accepted: bool) -> H3ModelsStatus:
        self._require_consent(accepted)
        acknowledge_h3_license(self._data)
        return self.status()

    def _scan(self, fresh: bool) -> dict[H3Variant, ComfyBundleStatus]:
        with self._lock:
            if not fresh and self._cached and time.monotonic() - self._cached[0] < 3:
                return self._cached[1]
            try:
                folders, nodes = self._folders()

                # Three recipes share both VAEs and their text encoder.
                @cache
                def locate(weight: ComfyWeight) -> ComfyInstalledFile:
                    return locate_comfy_weight(weight, folders)

                results: dict[H3Variant, ComfyBundleStatus] = {
                    key: recipe.inspect(folders, locate=locate)
                    if (_REF if key == "h3-ref2va" else _FL) in nodes
                    else recipe.offline(
                        "The selected ComfyUI worker lacks the built-in MiniMax H3 nodes. "
                        "Use the included renderer or update your external ComfyUI installation.",
                        status="error",
                    )
                    for key, recipe in self._recipes.items()
                }
            except (httpx.HTTPError, RuntimeError, OSError, ValueError) as exc:
                results = {
                    key: recipe.offline(f"The local video engine is unavailable. {exc}")
                    for key, recipe in self._recipes.items()
                }
            self._cached = time.monotonic(), results
            return results

    def status(self, *, fresh: bool = False) -> H3ModelsStatus:
        observed = self._scan(fresh)
        return H3ModelsStatus(
            variants={
                key: recipe.status(observed[key])
                for key, recipe in self._recipes.items()
            },
            licenseAcknowledged=self.license_acknowledged(),
        )

    def start(self, variant: H3Variant, *, accepted: bool) -> H3ModelsStatus:
        self._require_consent(accepted and self.license_acknowledged())
        self._recipes[variant].start(self._scan(fresh=True)[variant])
        return self.status()
