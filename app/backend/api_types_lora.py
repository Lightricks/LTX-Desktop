"""Wire models for catalog LoRA downloads and user-supplied Custom LoRAs."""

from __future__ import annotations

from typing import Annotated, Literal, TypeAlias, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    SerializerFunctionWrapHandler,
    StringConstraints,
    model_serializer,
    model_validator,
)


CustomLoraId: TypeAlias = Annotated[
    str,
    StringConstraints(pattern=r"^[a-z0-9-]+$", max_length=64),
]


class LoraEntry(BaseModel):
    model_config = ConfigDict(strict=True)

    ref: str
    scale: float = Field(default=1.0, ge=0.0, le=4.0)
    # Catalog id when this LoRA was picked from the library, so automatic enhancement can look
    # up its trigger/instructions. Absent for a custom LoRA the user supplied themselves, which
    # stays weights-only — the weights' filename is never matched against the catalog.
    catalogId: str | None = None
    # Catalog name at enqueue time. Kept after the LoRA leaves the catalog so
    # the dashboard can still label the run.
    displayName: str | None = None
    # Checkpoint chosen at enqueue, next to scale, so a later variant can be told apart.
    variantId: str | None = None
    # Id of the user's saved custom LoRA. Lets a client select it again when it reopens the run.
    # The weights stay in ``ref``; a custom LoRA is never matched against the catalog.
    customLoraId: CustomLoraId | None = None

    @model_serializer(mode="wrap")
    def _omit_unset_optional_ids(
        self, handler: SerializerFunctionWrapHandler
    ) -> dict[str, JsonValue]:
        data = cast("dict[str, JsonValue]", handler(self))
        for key in ("variantId", "customLoraId"):
            if data.get(key) is None:
                data.pop(key, None)
        return data


class IcLoraDownloadRequest(BaseModel):
    model_config = ConfigDict(strict=True)
    ic_lora_id: str
    # Optional catalog download.variants[].id; omit to download the default filename.
    variant_id: str | None = None
    # Dev escape hatch: attach the in-app HuggingFace token even when HF gating is off
    # (gated IC-LoRA repos need auth; gating is normally env-driven in packaged builds).
    use_hf_auth: bool = False


class IcLoraDeleteRequest(BaseModel):
    model_config = ConfigDict(strict=True)
    ic_lora_id: str
    variant_id: str | None = None


# Shared start-response for both catalog download kinds (IC-LoRA + plain LoRA).
class CatalogDownloadStartResponse(BaseModel):
    model_config = ConfigDict(strict=True)
    status: str
    sessionId: str


CatalogDownloadStatus: TypeAlias = Literal["downloading", "complete", "error"]


class IcLoraDownloadProgressResponse(BaseModel):
    model_config = ConfigDict(strict=True)
    status: CatalogDownloadStatus
    ic_lora_id: str | None = None
    downloaded_bytes: int = 0
    expected_bytes: int = 0
    progress: float = 0.0
    speed_bytes_per_sec: float = 0.0
    error: str | None = None


class LoraDownloadRequest(BaseModel):
    model_config = ConfigDict(strict=True)
    lora_id: str
    # Optional catalog download.variants[].id; omit to download the default filename.
    variant_id: str | None = None
    use_hf_auth: bool = False


class LoraDeleteRequest(BaseModel):
    model_config = ConfigDict(strict=True)
    lora_id: str
    variant_id: str | None = None


class LoraDownloadProgressResponse(BaseModel):
    model_config = ConfigDict(strict=True)
    status: CatalogDownloadStatus
    lora_id: str | None = None
    downloaded_bytes: int = 0
    expected_bytes: int = 0
    progress: float = 0.0
    speed_bytes_per_sec: float = 0.0
    error: str | None = None


class ActiveLoraDownloadResponse(BaseModel):
    model_config = ConfigDict(strict=True)
    # The currently-running plain-LoRA download session. Nulls when idle.
    session_id: str | None
    lora_id: str | None
    # 0–99 while downloading so a remount can seed the progress label instead of flashing 0%.
    progress: float | None = None


class ActiveIcLoraDownloadResponse(BaseModel):
    model_config = ConfigDict(strict=True)
    # The currently-running IC-LoRA download session. Nulls when idle.
    session_id: str | None
    ic_lora_id: str | None
    progress: float | None = None


# --- Custom LoRA (user-supplied weights: a local file or a URL) ---
CustomLoraSource: TypeAlias = Literal["file", "url"]

# What the user tells the prompt enhancer about their LoRA. The text goes into the enhancer's
# system prompt, so each part has a cap.
CUSTOM_LORA_TRIGGER_MAX = 80
CUSTOM_LORA_DESCRIPTION_MAX = 600
CUSTOM_LORA_EXAMPLE_MAX = 1000
CUSTOM_LORA_EXAMPLES_MAX = 3

# A trigger is one line: it is quoted inside the enhancer's system prompt.
CustomLoraTrigger: TypeAlias = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=CUSTOM_LORA_TRIGGER_MAX,
        pattern=r"^[^\r\n]+$",
    ),
]
CustomLoraDescription: TypeAlias = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=CUSTOM_LORA_DESCRIPTION_MAX),
]
CustomLoraExample: TypeAlias = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=CUSTOM_LORA_EXAMPLE_MAX),
]


class CustomLoraItem(BaseModel):
    """One saved custom LoRA.

    The desktop list puts the cover file path in ``cover_url``. The phone list replaces that
    path with a signed URL, so a paired phone never receives a filesystem path.
    """

    model_config = ConfigDict(strict=True)
    id: str
    name: str
    size_bytes: int
    source: CustomLoraSource
    # False after the user deleted the weights. The LoRA stays listed and needs the weights again.
    downloaded: bool
    # True when the weights can be fetched again from the saved link.
    can_download: bool
    # The Hugging Face model page, for a LoRA added from a Hugging Face link. Else `None`.
    page_url: str | None = None
    # What the user told the prompt enhancer. A LoRA saved before this existed has no description.
    trigger: str | None = None
    description: str = ""
    examples: list[str] = Field(default_factory=list[str])
    # The cover picture. On this computer it is the file path. On a paired phone the list
    # route replaces that path with a signed URL. `None` when the LoRA has no cover.
    cover_url: str | None = None


class CustomLoraListResponse(BaseModel):
    model_config = ConfigDict(strict=True)
    loras: list[CustomLoraItem]


class CustomLoraImportRequest(BaseModel):
    """Add a custom LoRA from exactly one source: a local file path or a URL."""

    model_config = ConfigDict(strict=True, extra="forbid")
    name: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=60)]
    source_path: str | None = Field(default=None, min_length=1)
    url: str | None = Field(default=None, min_length=1, max_length=2048)
    # What the LoRA does. Without it the prompt enhancer gives the LoRA a generic rewrite.
    description: CustomLoraDescription | None = None
    # A word the prompt must contain for the LoRA to work. Many LoRAs have none.
    trigger: CustomLoraTrigger | None = None
    # Prompts that work well with this LoRA. The enhancer copies their style, not their wording.
    examples: list[CustomLoraExample] = Field(
        default_factory=list[CustomLoraExample], max_length=CUSTOM_LORA_EXAMPLES_MAX
    )
    # A picture for the LoRA's card. Left out when the user skips the cover.
    cover_path: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _require_one_source(self) -> "CustomLoraImportRequest":
        if (self.source_path is None) == (self.url is None):
            raise ValueError("Give either source_path or url, not both")
        return self


class CustomLoraLinkInfoRequest(BaseModel):
    """Check whether a URL is an accepted Custom LoRA link and read its Hugging Face model card."""

    model_config = ConfigDict(strict=True, extra="forbid")
    url: str = Field(min_length=1, max_length=2048)


class CustomLoraLinkInfoResponse(BaseModel):
    """Link validity and suggestions from a model card. The user checks and edits them. Nothing is saved."""

    model_config = ConfigDict(strict=True)
    accepted: bool = False
    trigger: str | None = None
    description: str = ""
    examples: list[str] = Field(default_factory=list[str])


class CustomLoraRestoreRequest(BaseModel):
    """Put the weights of a saved LoRA back. A `source_path` copies that file. Without it, the
    weights are fetched again from the saved link."""

    model_config = ConfigDict(strict=True, extra="forbid")
    source_path: str | None = Field(default=None, min_length=1)


class CustomLoraImportStartResponse(BaseModel):
    model_config = ConfigDict(strict=True)
    status: str
    sessionId: str
    # The id the LoRA has once the import completes, so the client can select it.
    custom_lora_id: str


class CustomLoraImportProgressResponse(BaseModel):
    model_config = ConfigDict(strict=True)
    status: CatalogDownloadStatus
    custom_lora_id: str | None = None
    downloaded_bytes: int = 0
    expected_bytes: int = 0
    progress: float = 0.0
    speed_bytes_per_sec: float = 0.0
    error: str | None = None


class ActiveCustomLoraImportResponse(BaseModel):
    model_config = ConfigDict(strict=True)
    # The running import session. Nulls when idle.
    session_id: str | None
    custom_lora_id: str | None
    progress: float | None = None
