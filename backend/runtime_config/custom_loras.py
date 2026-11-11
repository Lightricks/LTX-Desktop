"""On-disk storage for custom LoRAs (weights the user supplies as a file or a URL).

Layout under the models dir::

    loras/custom/<id>/<id>.safetensors
    loras/custom/<id>/meta.json
    loras/custom/<id>/cover.<ext>

``<id>`` is a slug of the name the user typed. A folder is one custom LoRA. A folder without
``meta.json`` is an unfinished import and is ignored. A folder with ``meta.json`` and no weights
is a LoRA whose weights the user deleted: it stays listed and asks for the weights again.
The ``loras`` directory also makes the file visible to the plain-LoRA scanner, so a saved
custom LoRA shows up in the GenSpace picker without extra work.
"""

from __future__ import annotations

import json
import os
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    StrictInt,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

from api_types import (
    CustomLoraSource,
    InstructionSection,
    TriggerPlacement,
)
from runtime_config.models_scanner import is_ic_lora_file

CUSTOM_LORAS_SUBDIR = Path("loras") / "custom"
META_FILENAME = "meta.json"
STAGING_DIRNAME = ".tmp"
WEIGHTS_SUFFIX = ".safetensors"
# The cover is stored under a fixed name so meta.json never points outside the LoRA folder.
COVER_MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}
COVER_FILENAMES = frozenset(f"cover{suffix}" for suffix in COVER_MEDIA_TYPES)
_NOT_SAFETENSORS = "The file is not a .safetensors file."

# LTX LoRAs are hundreds of megabytes to a few gigabytes. The cap stops a wrong link
# (a model checkpoint, a video) from filling the disk before the header check can run.
MAX_CUSTOM_LORA_BYTES = 8 * 1024**3

_MAX_SLUG_LENGTH = 48
_MAX_ID_LENGTH = 64
_MAX_HEADER_BYTES = 16 * 1024 * 1024
_SLUG_RE = re.compile(r"[^a-z0-9]+")
_CUSTOM_LORA_ID_RE = re.compile(r"^[a-z0-9-]+$")

class CustomLoraFileError(ValueError):
    """The file is not usable as a custom LoRA. The message is safe to show to the user."""


@dataclass(frozen=True, slots=True)
class CustomLoraRecord:
    id: str
    name: str
    filename: str
    # The size of the weights. Kept in ``meta.json`` so the row shows it after the weights are deleted.
    size_bytes: int
    source: CustomLoraSource
    # The file URL that was fetched. ``None`` for a file import: only a link can be fetched again.
    source_url: str | None = None
    # The Hugging Face model page, when the link pointed at Hugging Face. ``None`` otherwise.
    page_url: str | None = None
    downloaded: bool = True
    # What the user told the prompt enhancer. A LoRA saved before this existed has no description.
    trigger: str | None = None
    description: str = ""
    examples: tuple[str, ...] = ()
    # ``cover.png`` (or jpg, jpeg, webp) inside the LoRA folder. ``None`` when the user skipped it.
    cover_filename: str | None = None

    @property
    def ref(self) -> str:
        """The weights path relative to the models dir, in the form ``resolve_lora_ref`` takes."""
        return (CUSTOM_LORAS_SUBDIR / self.id / self.filename).as_posix()


def custom_loras_root(models_dir: Path) -> Path:
    return models_dir / CUSTOM_LORAS_SUBDIR


def custom_lora_dir(models_dir: Path, custom_lora_id: str) -> Path:
    if not _CUSTOM_LORA_ID_RE.fullmatch(custom_lora_id) or len(custom_lora_id) > _MAX_ID_LENGTH:
        raise ValueError(f"Invalid custom LoRA id: {custom_lora_id!r}")
    return custom_loras_root(models_dir) / custom_lora_id


def slugify_name(name: str) -> str:
    """The folder id for a name, or ``""`` when the name has no letters or digits."""
    slug = _SLUG_RE.sub("-", name.strip().lower()).strip("-")
    return slug[:_MAX_SLUG_LENGTH].strip("-")


class _TensorEntry(BaseModel):
    """One tensor of a safetensors header. Only the byte range matters here."""

    model_config = ConfigDict(strict=True, extra="ignore")

    # [start, end]. A strict int refuses True and False, which are ints in Python.
    data_offsets: Annotated[list[NonNegativeInt], Field(min_length=2, max_length=2)]

    @model_validator(mode="after")
    def _ordered(self) -> "_TensorEntry":
        if self.data_offsets[0] > self.data_offsets[1]:
            raise ValueError("data_offsets are out of order")
        return self


# A header maps tensor names to entries. ``__metadata__`` is not a tensor and is taken out first.
_HEADER = TypeAdapter(dict[str, _TensorEntry])


def validate_safetensors_weights(path: Path) -> None:
    """Raise ``CustomLoraFileError`` unless ``path`` is a complete plain-LoRA safetensors file.

    Reads the header only. The tensor offsets must fit inside the file, which catches a
    truncated download. A file with the IC-LoRA marker is rejected: this feature runs plain
    LoRAs, and an IC-LoRA needs a control video.

    The ``safetensors`` library is not used for this check: it memory-maps the whole file, and the
    app patches it on purpose because of the Windows commit-charge cost of that
    (``services/patches/safetensors_metadata_fix.py``).
    """
    try:
        file_size = path.stat().st_size
        with open(path, "rb") as f:
            raw_size = f.read(8)
            if len(raw_size) != 8:
                raise CustomLoraFileError(_NOT_SAFETENSORS)
            (header_size,) = struct.unpack("<Q", raw_size)
            if header_size == 0 or header_size > _MAX_HEADER_BYTES or 8 + header_size > file_size:
                raise CustomLoraFileError(_NOT_SAFETENSORS)
            header: object = json.loads(f.read(header_size).decode("utf-8"))
        if not isinstance(header, dict):
            raise CustomLoraFileError(_NOT_SAFETENSORS)
        tensors = _HEADER.validate_python(
            {name: entry for name, entry in cast("dict[str, object]", header).items() if name != "__metadata__"}
        )
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        # json.JSONDecodeError and pydantic's ValidationError are both ValueErrors.
        raise CustomLoraFileError(_NOT_SAFETENSORS) from exc
    if not tensors:
        raise CustomLoraFileError("The file has no weights.")
    if 8 + header_size + max(t.data_offsets[1] for t in tensors.values()) > file_size:
        raise CustomLoraFileError("The file is incomplete. Download it again.")
    if is_ic_lora_file(path):
        raise CustomLoraFileError(
            "This is an IC-LoRA. Custom LoRA runs plain LoRAs only."
        )
    lora_a = {k[: -len(".lora_A.weight")] for k in tensors if k.endswith(".lora_A.weight")}
    lora_b = {k[: -len(".lora_B.weight")] for k in tensors if k.endswith(".lora_B.weight")}
    if not (lora_a & lora_b):
        raise CustomLoraFileError("The file does not contain LoRA weights.")


def write_meta(directory: Path, record: CustomLoraRecord) -> None:
    """Write ``meta.json`` last, so a folder with a meta file is always a finished import."""
    payload: dict[str, object] = {
        "name": record.name,
        "filename": record.filename,
        "source": record.source,
        "size_bytes": record.size_bytes,
        "source_url": record.source_url,
        "page_url": record.page_url,
        "trigger": record.trigger,
        "description": record.description,
        "examples": list(record.examples),
        "cover_filename": record.cover_filename,
    }
    tmp = directory / f"{META_FILENAME}.part"
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp, directory / META_FILENAME)


class _Meta(BaseModel):
    """The shape of ``meta.json``. A file that does not fit is treated as no LoRA."""

    model_config = ConfigDict(strict=True, extra="ignore")

    name: str
    filename: str
    source: CustomLoraSource
    size_bytes: StrictInt = 0
    source_url: str | None = None
    page_url: str | None = None
    trigger: str | None = None
    description: str = ""
    examples: list[str] = Field(default_factory=list[str])
    cover_filename: str | None = None

    @field_validator("source_url", "page_url", "trigger")
    @classmethod
    def _empty_is_none(cls, value: str | None) -> str | None:
        return value or None

    @field_validator("filename")
    @classmethod
    def _bare_name(cls, value: str) -> str:
        # The filename comes from a file on disk: keep it a bare name so the ref stays inside the folder.
        if Path(value).name != value:
            raise ValueError("filename must be a bare name")
        return value

    @field_validator("cover_filename")
    @classmethod
    def _cover_name(cls, value: str | None) -> str | None:
        # A bad name is no cover. It must not hide the rest of the LoRA.
        if not value or value not in COVER_FILENAMES:
            return None
        return value


def read_custom_lora(models_dir: Path, custom_lora_id: str) -> CustomLoraRecord | None:
    """The saved LoRA, or ``None`` when its folder or its meta file is missing or unreadable.

    A LoRA whose weights are gone is still returned, with ``downloaded=False``.
    """
    try:
        directory = custom_lora_dir(models_dir, custom_lora_id)
        meta = _Meta.model_validate_json((directory / META_FILENAME).read_bytes())
    except (OSError, ValidationError, ValueError):
        return None
    size = meta.size_bytes
    downloaded = False
    try:
        size = (directory / meta.filename).stat().st_size
        downloaded = True
    except OSError:
        pass
    return CustomLoraRecord(
        id=custom_lora_id,
        name=meta.name,
        filename=meta.filename,
        size_bytes=size,
        source=meta.source,
        source_url=meta.source_url,
        page_url=meta.page_url,
        downloaded=downloaded,
        trigger=meta.trigger,
        description=meta.description,
        examples=tuple(meta.examples),
        cover_filename=meta.cover_filename,
    )


def custom_lora_cover_path(models_dir: Path, record: CustomLoraRecord) -> Path | None:
    """The cover file inside the LoRA folder, or ``None`` when there is no cover.

    The filename is a fixed bare name. A path that resolves outside the folder is ignored.
    """
    if record.cover_filename is None:
        return None
    directory = custom_lora_dir(models_dir, record.id).resolve()
    path = (directory / record.cover_filename).resolve()
    if path.parent != directory or not path.is_file():
        return None
    return path


@dataclass(frozen=True, slots=True)
class CustomLoraPromptGuidance:
    name: str
    instructions: tuple[InstructionSection, ...]
    trigger: str | None
    trigger_placement: TriggerPlacement | None
    enhancement_examples: tuple[str, ...]


def custom_lora_prompt_guidance(record: CustomLoraRecord) -> CustomLoraPromptGuidance:
    """The prompt-enhancer guidance for a saved custom LoRA.

    Uses a generic heading instead of the user's label so the enhancer does not write the
    arbitrary label into the prompt text.
    """
    return CustomLoraPromptGuidance(
        name="Selected LoRA",
        instructions=(
            (InstructionSection(kind="summary", title="What it does", body=record.description),)
            if record.description
            else ()
        ),
        trigger=record.trigger,
        trigger_placement="anywhere" if record.trigger else None,
        enhancement_examples=record.examples,
    )


def list_custom_loras(models_dir: Path) -> list[CustomLoraRecord]:
    root = custom_loras_root(models_dir)
    if not root.is_dir():
        return []
    records: list[CustomLoraRecord] = []
    for entry in sorted(root.iterdir(), key=lambda p: p.name):
        if not entry.is_dir() or entry.name == STAGING_DIRNAME:
            continue
        record = read_custom_lora(models_dir, entry.name)
        if record is not None:
            records.append(record)
    return records
