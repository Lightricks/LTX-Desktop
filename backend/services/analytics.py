"""Forward privacy-safe analytics events to the Electron sink."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Literal, cast

from api_types import PromptProvenance
from services.features.lora_recipes import get_lora_recipe
from services.http_client.http_client import HTTPClient, HttpTransportError
from services.records import GenerationRecord
from services.services_utils import JSONValue
from services.task_runner.task_runner import TaskRunner

logger = logging.getLogger(__name__)

GenerationEventName = Literal["generate_started", "generate_ended"]
GenerationOutcome = Literal["succeeded", "failed", "cancelled"]
PromptProvenanceLabel = Literal["raw", "manually-enhanced", "auto-enhanced"]
_HOME_QUEUE_KIND = "video"
_HOME_QUEUE_EXECUTION = "local"
_FORCED_UNENHANCED_FEATURES = frozenset({"retake", "extend"})
# generate_ended joins generate_started on generation_id. It does not repeat
# the start payload; attempt stays so a retry of the same id still pairs.
# The full set is shared/analytics-ended-details.json. Electron drops anything else.
ENDED_DETAIL_KEYS = (
    "generation_id",
    "attempt",
    "outcome",
    "runtime_ms",
    "error_code",
)
_ENDED_LINK_KEYS = ("generation_id", "attempt")


class QueuedEnhancement:
    """Explore enhance settings at the moment a home-queue event is sent."""

    def __init__(
        self,
        *,
        explore_auto_enhance_prompts: bool,
        prompt_enhancer_enabled: bool,
        use_local_encoding: bool,
        local_enhancer_available: bool = False,
    ) -> None:
        self.explore_auto_enhance_prompts = explore_auto_enhance_prompts
        self.prompt_enhancer_enabled = prompt_enhancer_enabled
        self.use_local_encoding = use_local_encoding
        self.local_enhancer_available = local_enhancer_available


def prompt_provenance_label(
    provenance: str,
    *,
    enhanced_locally: bool,
    enhance_via_api: bool,
) -> PromptProvenanceLabel:
    """How this run asked for the prompt to be passed. Not the prompt text."""
    if provenance == "enhanced":
        return "manually-enhanced"
    if enhanced_locally or enhance_via_api:
        return "auto-enhanced"
    return "raw"


def unenhanced_prompt_provenance(provenance: str) -> PromptProvenanceLabel:
    """Retake and extend always submit the prompt as-is."""
    if provenance == "enhanced":
        return "manually-enhanced"
    return "raw"


def resolution_for_provenance(
    provenance: str,
    *,
    prompt: str,
    explore_generation: bool,
    explore_auto_enhance_prompts: bool,
    use_local_encoding: bool,
    prompt_enhancer_enabled: bool,
    force_api: bool = False,
    local_enhancer_available: bool = False,
) -> tuple[bool, bool, bool] | None:
    """Shared enhance decision for generation and the analytics preview.

    None means a local rewrite will be attempted. Otherwise the tuple is
    enhance_via_api, enhanced_locally, skip_recipe_wrap.
    """
    skip_recipe_wrap = (
        explore_generation
        and not explore_auto_enhance_prompts
        and provenance != "enhanced"
    )
    if provenance == "enhanced" or not prompt.strip() or skip_recipe_wrap:
        return (False, False, skip_recipe_wrap)
    # Home auto-enhance rewrites on the local checkpoint when it is installed.
    # API text encoding is a different switch: leaving the rewrite to that path
    # drops it for style recipes, because the scaffold wrap turns the API flag
    # off so /prompt-embedding cannot rewrite the style lock.
    if explore_generation and explore_auto_enhance_prompts and local_enhancer_available:
        return None
    if force_api or not use_local_encoding:
        return (prompt_enhancer_enabled, False, skip_recipe_wrap)
    return None


def _recipe_clears_api_enhance(feature: str) -> bool:
    """Queued LoRA recipes pass a prompt wrap that forces API enhance off.

    A local rewrite still happens before that wrap. An API rewrite does not:
    the scaffold would be rewritten, so the job encodes the wrapped prompt as-is.
    """
    recipe = get_lora_recipe(feature)
    return recipe is not None and recipe.mode in ("t2v", "i2v")


def _stored_provenance(value: object) -> PromptProvenance:
    if value == "enhanced":
        return "enhanced"
    return "typed"


def queued_prompt_provenance(
    generation: GenerationRecord,
    enhancement: QueuedEnhancement,
) -> str | None:
    params = generation.spec.get("params")
    if not isinstance(params, dict):
        return None
    provenance = _stored_provenance(params.get("promptProvenance"))
    prompt = params.get("prompt")
    prompt_text = prompt if isinstance(prompt, str) else ""
    if generation.feature in _FORCED_UNENHANCED_FEATURES:
        return unenhanced_prompt_provenance(provenance)
    resolved = resolution_for_provenance(
        provenance,
        prompt=prompt_text,
        explore_generation=True,
        explore_auto_enhance_prompts=enhancement.explore_auto_enhance_prompts,
        use_local_encoding=enhancement.use_local_encoding,
        prompt_enhancer_enabled=enhancement.prompt_enhancer_enabled,
        local_enhancer_available=enhancement.local_enhancer_available,
    )
    if resolved is None:
        return prompt_provenance_label(provenance, enhanced_locally=True, enhance_via_api=False)
    enhance_via_api, enhanced_locally, skip_recipe_wrap = resolved
    if _recipe_clears_api_enhance(generation.feature) and not skip_recipe_wrap:
        enhance_via_api = False
    return prompt_provenance_label(
        provenance,
        enhanced_locally=enhanced_locally,
        enhance_via_api=enhance_via_api,
    )


def pixel_resolution(width: int, height: int) -> str:
    return f"{width}x{height}"


def _resolution_value(value: object) -> JSONValue | None:
    if isinstance(value, str) and value:
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    if not isinstance(value, dict):
        return None
    fields = cast(dict[str, object], value)
    width = fields.get("width")
    height = fields.get("height")
    if (
        isinstance(width, int)
        and not isinstance(width, bool)
        and isinstance(height, int)
        and not isinstance(height, bool)
        and width > 0
        and height > 0
    ):
        return pixel_resolution(width, height)
    return None


def queued_generation_details(
    generation: GenerationRecord,
    *,
    enhancement: QueuedEnhancement | None = None,
) -> dict[str, JSONValue] | None:
    analytics = generation.spec.get("_analytics")
    if not isinstance(analytics, dict):
        return None
    client = analytics.get("client")
    if not isinstance(client, str) or client not in ("desktop", "remote"):
        return None

    details: dict[str, JSONValue] = {
        "generation_id": generation.id,
        "attempt": generation.attempt_count,
        "client": client,
        "surface": "home",
        "feature": generation.feature,
        "kind": _HOME_QUEUE_KIND,
        "execution": _HOME_QUEUE_EXECUTION,
    }
    params = generation.spec.get("params")
    if not isinstance(params, dict):
        return details
    for source, target in (
        ("model", "model"),
        ("duration", "duration_sec"),
        ("fps", "fps"),
    ):
        value = params.get(source)
        if isinstance(value, (str, int, float)) and not isinstance(value, bool):
            details[target] = value
    if "duration_sec" not in details:
        num_frames = params.get("numFrames")
        fps = params.get("fps")
        if (
            isinstance(num_frames, int)
            and not isinstance(num_frames, bool)
            and isinstance(fps, (int, float))
            and not isinstance(fps, bool)
            and fps > 0
        ):
            details["duration_sec"] = (num_frames - 1) / float(fps)
    resolution = _resolution_value(params.get("resolution"))
    if resolution is not None:
        details["resolution"] = resolution

    loras = params.get("loras")
    if isinstance(loras, list):
        catalog_ids: list[JSONValue] = []
        has_custom_lora = False
        for lora in loras:
            if not isinstance(lora, dict):
                continue
            catalog_id = lora.get("catalogId")
            if isinstance(catalog_id, str) and catalog_id:
                catalog_ids.append(catalog_id)
            else:
                has_custom_lora = True
        if catalog_ids:
            details["lora_catalog_ids"] = catalog_ids
        if has_custom_lora:
            details["has_custom_lora"] = True
    if enhancement is not None:
        provenance = queued_prompt_provenance(generation, enhancement)
        if provenance is not None:
            details["prompt_provenance"] = provenance
    return details


def _ended_details(
    details: dict[str, JSONValue],
    *,
    outcome: GenerationOutcome,
    runtime_ms: int,
    error_code: str | None,
) -> dict[str, JSONValue]:
    """Identity plus what is only known when the run finishes."""
    ended: dict[str, JSONValue] = {
        key: details[key] for key in _ENDED_LINK_KEYS if key in details
    }
    ended["outcome"] = outcome
    ended["runtime_ms"] = max(0, runtime_ms)
    if error_code is not None:
        ended["error_code"] = error_code
    return ended


class AnalyticsService:
    def __init__(
        self,
        *,
        http: HTTPClient,
        task_runner: TaskRunner,
        sink_url: str = "",
        token: str = "",
    ) -> None:
        self._http = http
        self._task_runner = task_runner
        self._sink_url = sink_url
        self._token = token
        self._send_order = threading.Lock()
        self._send_tail: threading.Lock | None = None

    def send_generation_started(self, details: dict[str, JSONValue]) -> None:
        self._send("generate_started", details)

    def send_generation_ended(
        self,
        details: dict[str, JSONValue],
        *,
        outcome: GenerationOutcome,
        runtime_ms: int,
        error_code: str | None = None,
    ) -> None:
        self._send(
            "generate_ended",
            _ended_details(
                details,
                outcome=outcome,
                runtime_ms=runtime_ms,
                error_code=error_code,
            ),
        )

    @contextmanager
    def generation(
        self, details: dict[str, JSONValue]
    ) -> Iterator[GenerationAnalyticsTracker]:
        tracker = GenerationAnalyticsTracker(self, details)
        try:
            tracker.start()
            yield tracker
        except Exception:
            tracker.finish("failed")
            raise
        else:
            tracker.finish("succeeded")

    def _send(self, event_name: GenerationEventName, details: dict[str, JSONValue]) -> None:
        if self._sink_url == "" or self._token == "":
            return
        payload: dict[str, JSONValue] = {
            "eventName": event_name,
            "extraDetails": details,
        }
        # Chain posts in emission order. Each background post waits for the
        # previous one, so a fast generate_ended cannot overtake its start.
        finished = threading.Lock()
        finished.acquire()
        with self._send_order:
            previous = self._send_tail
            self._send_tail = finished

        def post() -> None:
            try:
                if previous is not None:
                    with previous:
                        pass
                self._post(payload)
            finally:
                finished.release()

        try:
            self._task_runner.run_background(
                post,
                task_name=f"analytics-{event_name}",
                on_error=lambda exc: logger.warning("Analytics forward failed: %s", exc),
            )
        except Exception as exc:
            finished.release()
            logger.warning("Analytics schedule failed: %s", exc)

    def _post(self, payload: dict[str, JSONValue]) -> None:
        try:
            response = self._http.post(
                self._sink_url,
                headers={
                    "Authorization": f"Bearer {self._token}",
                    "Content-Type": "application/json",
                },
                json_payload=payload,
                timeout=5,
            )
        except HttpTransportError as exc:
            logger.warning("Analytics sink request failed: %s", exc)
            return
        if not 200 <= response.status_code < 300:
            logger.warning("Analytics sink returned HTTP %s", response.status_code)


class GenerationAnalyticsTracker:
    def __init__(self, service: AnalyticsService, details: dict[str, JSONValue]) -> None:
        self._service = service
        self._details = details
        self._started_at = 0
        self._finished = False

    def start(self) -> None:
        if self._started_at != 0:
            return
        self._started_at = int(time.time() * 1000)
        self._service.send_generation_started(self._details)

    def finish(
        self, outcome: GenerationOutcome, *, error_code: str | None = None
    ) -> None:
        if self._finished:
            return
        self._finished = True
        now = int(time.time() * 1000)
        self._service.send_generation_ended(
            self._details,
            outcome=outcome,
            runtime_ms=now - self._started_at,
            error_code=error_code,
        )
