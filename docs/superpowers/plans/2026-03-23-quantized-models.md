# Quantized Video Model Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Support GGUF, NF4, and FP8 checkpoint model formats so Directors Desktop runs on 24GB GPUs.

**Architecture:** New `ModelScanner` service (Protocol + Fake pattern) scans a user-specified folder for model files and detects their format from file metadata. New `GGUFFastVideoPipeline` and `NF4FastVideoPipeline` classes implement the existing `FastVideoPipeline` protocol. `PipelinesHandler._create_video_pipeline()` reads the selected model from settings and picks the right pipeline class. Frontend gets a new "Models" tab in SettingsModal and a `ModelGuideDialog` popup with GPU-based recommendations and HuggingFace download links.

**Tech Stack:** Python (FastAPI, pydantic, diffusers GGUF quantizer, bitsandbytes NF4), TypeScript (React, Tailwind CSS), gguf Python package (new dependency)

**Spec:** `docs/superpowers/specs/2026-03-22-quantized-models-design.md`

---

### Task 1: Add `gguf` dependency and new settings fields

**Context:** Before anything else, we need the `gguf` package for reading GGUF file metadata, and two new settings fields (`custom_video_model_path`, `selected_video_model`) that the rest of the system depends on.

**Files:**
- Modify: `backend/pyproject.toml` — add `gguf>=0.10.0` to dependencies
- Modify: `backend/state/app_settings.py` — add 2 new fields to `AppSettings` and `SettingsResponse`, update `to_settings_response()`
- Modify: `backend/api_types.py` — add `DetectedModel`, `ModelFormatInfo`, `SelectModelRequest`, response types

- [ ] **Step 1: Add gguf dependency**

In `backend/pyproject.toml`, add `"gguf>=0.10.0"` to the `dependencies` list.

- [ ] **Step 2: Install the dependency**

Run: `cd backend && uv sync`
Expected: Resolves and installs gguf package

- [ ] **Step 3: Add settings fields to AppSettings**

In `backend/state/app_settings.py`, add these two fields to `AppSettings` class (after `civitai_api_key`):

```python
custom_video_model_path: str = ""
selected_video_model: str = ""
```

- [ ] **Step 4: Add settings fields to SettingsResponse**

In `backend/state/app_settings.py`, add these two fields to `SettingsResponse` class (after `has_civitai_api_key`):

```python
custom_video_model_path: str = ""
selected_video_model: str = ""
```

- [ ] **Step 5: Update to_settings_response()**

In `backend/state/app_settings.py`, the `to_settings_response()` function currently pops API keys and replaces them with `has_*` booleans. The new fields are NOT sensitive — they should pass through as-is. Since `SettingsResponse` uses `extra="ignore"` via `SettingsBaseModel`, and both fields are simple strings with the same names, they will pass through `model_validate(data)` automatically. **No code changes needed to `to_settings_response()` itself** — the fields are already in `data` from `model_dump()` and `SettingsResponse` has matching fields.

Verify this works: `cd backend && uv run python -c "from state.app_settings import AppSettings, to_settings_response; s = AppSettings(custom_video_model_path='/test', selected_video_model='model.gguf'); r = to_settings_response(s); print(r.custom_video_model_path, r.selected_video_model)"`

Expected: `/test model.gguf`

- [ ] **Step 6: Add new types to api_types.py**

At the end of `backend/api_types.py` (before the final empty line), add:

```python
# ============================================================
# Video Model Scanner Types
# ============================================================


class DetectedModel(BaseModel):
    filename: str
    path: str
    format: str  # "bf16" | "fp8" | "gguf" | "nf4"
    quant_type: str | None = None
    size_bytes: int
    size_gb: float
    is_distilled: bool
    display_name: str


class ModelFormatInfo(BaseModel):
    id: str
    name: str
    size_gb: float
    min_vram_gb: int
    quality_tier: str
    needs_distilled_lora: bool
    download_url: str
    description: str


class DistilledLoraInfo(BaseModel):
    name: str
    size_gb: float
    download_url: str
    description: str


class VideoModelScanResponse(BaseModel):
    models: list[DetectedModel]
    distilled_lora_found: bool


class VideoModelGuideResponse(BaseModel):
    gpu_name: str | None
    vram_gb: int | None
    recommended_format: str
    formats: list[ModelFormatInfo]
    distilled_lora: DistilledLoraInfo


class SelectModelRequest(BaseModel):
    model: str
```

- [ ] **Step 7: Run typecheck to verify**

Run: `cd backend && uv run pyright`
Expected: `0 errors, 0 warnings, 0 informations`

- [ ] **Step 8: Commit**

```bash
cd D:/git/directors-desktop
git add backend/pyproject.toml backend/uv.lock backend/state/app_settings.py backend/api_types.py
git commit -m "feat(quantized-models): add gguf dependency, settings fields, and API types"
```

---

### Task 2: Create ModelScanner service (Protocol + Impl + Fake)

**Context:** The `ModelScanner` service scans a folder for video model files and returns structured `DetectedModel` results. It follows the codebase's Protocol + Impl + Fake pattern (see `services/palette_sync_client/` for reference). Detection uses file metadata, NOT file size heuristics.

**Files:**
- Create: `backend/services/model_scanner/__init__.py`
- Create: `backend/services/model_scanner/model_scanner.py` — Protocol
- Create: `backend/services/model_scanner/model_scanner_impl.py` — Real implementation
- Create: `backend/services/model_scanner/model_guide_data.py` — Static format metadata
- Modify: `backend/services/interfaces.py` — re-export ModelScanner
- Modify: `backend/tests/fakes/services.py` — add FakeModelScanner
- Create: `backend/tests/test_model_scanner.py` — tests

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_model_scanner.py`:

```python
"""Tests for the ModelScanner service."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from services.model_scanner.model_scanner_impl import ModelScannerImpl


class TestScanVideoModels:
    def test_empty_folder_returns_empty_list(self, tmp_path: Path) -> None:
        scanner = ModelScannerImpl()
        result = scanner.scan_video_models(tmp_path)
        assert result == []

    def test_nonexistent_folder_returns_empty_list(self, tmp_path: Path) -> None:
        scanner = ModelScannerImpl()
        result = scanner.scan_video_models(tmp_path / "nonexistent")
        assert result == []

    def test_detects_gguf_file(self, tmp_path: Path) -> None:
        # Create a minimal GGUF file with magic bytes and version
        gguf_path = tmp_path / "model-Q4_K.gguf"
        _write_minimal_gguf(gguf_path)

        scanner = ModelScannerImpl()
        result = scanner.scan_video_models(tmp_path)

        assert len(result) == 1
        assert result[0].filename == "model-Q4_K.gguf"
        assert result[0].format == "gguf"
        assert result[0].is_distilled is False

    def test_detects_safetensors_file(self, tmp_path: Path) -> None:
        st_path = tmp_path / "model.safetensors"
        st_path.write_bytes(b"\x00" * 4096)  # minimal file

        scanner = ModelScannerImpl()
        result = scanner.scan_video_models(tmp_path)

        assert len(result) == 1
        assert result[0].filename == "model.safetensors"
        assert result[0].format in ("bf16", "fp8")

    def test_detects_nf4_folder(self, tmp_path: Path) -> None:
        nf4_dir = tmp_path / "my-nf4-model"
        nf4_dir.mkdir()
        (nf4_dir / "model.safetensors").write_bytes(b"\x00" * 1024)
        (nf4_dir / "quantize_config.json").write_text('{"quant_type": "nf4"}')

        scanner = ModelScannerImpl()
        result = scanner.scan_video_models(tmp_path)

        assert len(result) == 1
        assert result[0].format == "nf4"
        assert result[0].quant_type == "nf4"

    def test_skips_corrupt_gguf_file(self, tmp_path: Path) -> None:
        bad_path = tmp_path / "corrupt.gguf"
        bad_path.write_bytes(b"not a gguf file")

        scanner = ModelScannerImpl()
        result = scanner.scan_video_models(tmp_path)

        assert len(result) == 0

    def test_skips_non_model_files(self, tmp_path: Path) -> None:
        (tmp_path / "readme.txt").write_text("hello")
        (tmp_path / "config.json").write_text("{}")

        scanner = ModelScannerImpl()
        result = scanner.scan_video_models(tmp_path)

        assert len(result) == 0

    def test_multiple_models_in_folder(self, tmp_path: Path) -> None:
        # GGUF file
        _write_minimal_gguf(tmp_path / "model-q4.gguf")
        # Safetensors file
        (tmp_path / "model-bf16.safetensors").write_bytes(b"\x00" * 4096)

        scanner = ModelScannerImpl()
        result = scanner.scan_video_models(tmp_path)

        assert len(result) == 2
        formats = {m.format for m in result}
        assert "gguf" in formats


def _write_minimal_gguf(path: Path) -> None:
    """Write a minimal valid GGUF file header (magic + version + tensor/kv counts)."""
    with open(path, "wb") as f:
        f.write(b"GGUF")  # magic
        f.write(struct.pack("<I", 3))  # version 3
        f.write(struct.pack("<Q", 0))  # tensor_count
        f.write(struct.pack("<Q", 0))  # metadata_kv_count
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_model_scanner.py -v --tb=short`
Expected: FAIL — `ModuleNotFoundError: No module named 'services.model_scanner'`

- [ ] **Step 3: Create the ModelScanner Protocol**

Create `backend/services/model_scanner/__init__.py`:
```python
```

Create `backend/services/model_scanner/model_scanner.py`:
```python
"""Protocol for scanning model files in a directory."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from api_types import DetectedModel


class ModelScanner(Protocol):
    def scan_video_models(self, folder: Path) -> list[DetectedModel]: ...
```

- [ ] **Step 4: Create model_guide_data.py**

Create `backend/services/model_scanner/model_guide_data.py`:
```python
"""Static metadata about available video model formats and download URLs."""

from __future__ import annotations

from api_types import DistilledLoraInfo, ModelFormatInfo

MODEL_FORMATS: list[ModelFormatInfo] = [
    ModelFormatInfo(
        id="bf16",
        name="BF16 (Full Precision)",
        size_gb=43,
        min_vram_gb=32,
        quality_tier="Best",
        needs_distilled_lora=False,
        download_url="https://huggingface.co/Lightricks/LTX-Video-2.3-22b-distilled",
        description="Best quality. Requires 32GB+ VRAM. Auto-downloaded by default.",
    ),
    ModelFormatInfo(
        id="fp8",
        name="FP8 Distilled Checkpoint",
        size_gb=22,
        min_vram_gb=20,
        quality_tier="Excellent",
        needs_distilled_lora=False,
        download_url="https://huggingface.co/Lightricks/LTX-Video-2.3-22b-distilled",
        description="Excellent quality, smaller file. Good for 20-31GB VRAM GPUs.",
    ),
    ModelFormatInfo(
        id="gguf_q8",
        name="GGUF Q8",
        size_gb=22,
        min_vram_gb=18,
        quality_tier="Excellent",
        needs_distilled_lora=True,
        download_url="https://huggingface.co/city96/LTX-Video-2.3-22b-0.9.7-dev-gguf",
        description="Excellent quality quantized model. Needs distilled LoRA.",
    ),
    ModelFormatInfo(
        id="gguf_q5k",
        name="GGUF Q5_K",
        size_gb=15,
        min_vram_gb=13,
        quality_tier="Very Good",
        needs_distilled_lora=True,
        download_url="https://huggingface.co/city96/LTX-Video-2.3-22b-0.9.7-dev-gguf",
        description="Very good quality, balanced size. Good for 16-19GB VRAM GPUs.",
    ),
    ModelFormatInfo(
        id="gguf_q4k",
        name="GGUF Q4_K",
        size_gb=12,
        min_vram_gb=10,
        quality_tier="Good",
        needs_distilled_lora=True,
        download_url="https://huggingface.co/city96/LTX-Video-2.3-22b-0.9.7-dev-gguf",
        description="Good quality, smallest file. Good for 10-15GB VRAM GPUs.",
    ),
    ModelFormatInfo(
        id="nf4",
        name="NF4 (4-bit BitsAndBytes)",
        size_gb=12,
        min_vram_gb=10,
        quality_tier="Good",
        needs_distilled_lora=True,
        download_url="https://huggingface.co/Lightricks/LTX-Video-2.3-22b-distilled",
        description="4-bit quantization via BitsAndBytes. Good for 10-15GB VRAM GPUs.",
    ),
]

DISTILLED_LORA_INFO = DistilledLoraInfo(
    name="LTX 2.3 Distilled LoRA",
    size_gb=0.5,
    download_url="https://huggingface.co/Lightricks/LTX-Video-2.3-22b-distilled",
    description="Required for GGUF and NF4 models to enable fast distilled generation.",
)


def recommend_format(vram_gb: int | None) -> str:
    """Return the recommended format ID based on available VRAM."""
    if vram_gb is None:
        return "bf16"
    if vram_gb >= 32:
        return "bf16"
    if vram_gb >= 20:
        return "fp8"
    if vram_gb >= 16:
        return "gguf_q5k"
    if vram_gb >= 10:
        return "gguf_q4k"
    return "api_only"
```

- [ ] **Step 5: Create ModelScannerImpl**

Create `backend/services/model_scanner/model_scanner_impl.py`:
```python
"""Real implementation of ModelScanner that reads file metadata."""

from __future__ import annotations

import json
import logging
import struct
from pathlib import Path

from api_types import DetectedModel

logger = logging.getLogger(__name__)

# Known distilled checkpoint filenames (these don't need distilled LoRA)
_KNOWN_DISTILLED = {
    "ltx-video-2.3-22b-distilled.safetensors",
    "ltx-video-2.3-22b-distilled-fp8.safetensors",
}

_GGUF_MAGIC = b"GGUF"


class ModelScannerImpl:
    def scan_video_models(self, folder: Path) -> list[DetectedModel]:
        """Scan folder for supported video model files."""
        if not folder.exists():
            return []

        models: list[DetectedModel] = []

        for entry in sorted(folder.iterdir()):
            try:
                if entry.is_file() and entry.suffix == ".gguf":
                    model = self._scan_gguf(entry)
                    if model is not None:
                        models.append(model)
                elif entry.is_file() and entry.suffix == ".safetensors":
                    models.append(self._scan_safetensors(entry))
                elif entry.is_dir():
                    model = self._scan_nf4_folder(entry)
                    if model is not None:
                        models.append(model)
            except Exception:
                logger.warning("Failed to scan model file %s", entry, exc_info=True)

        return models

    def _scan_gguf(self, path: Path) -> DetectedModel | None:
        """Read GGUF file header to verify validity and extract quant type."""
        with open(path, "rb") as f:
            magic = f.read(4)
            if magic != _GGUF_MAGIC:
                logger.warning("Skipping %s: invalid GGUF magic bytes", path.name)
                return None

            version_bytes = f.read(4)
            if len(version_bytes) < 4:
                return None
            struct.unpack("<I", version_bytes)  # validate version is readable

        # Try to extract quant type from filename (common convention: model-Q4_K.gguf)
        quant_type = self._quant_type_from_filename(path.name)
        size_bytes = path.stat().st_size
        size_gb = round(size_bytes / (1024**3), 1)

        return DetectedModel(
            filename=path.name,
            path=str(path),
            format="gguf",
            quant_type=quant_type,
            size_bytes=size_bytes,
            size_gb=size_gb,
            is_distilled=False,
            display_name=self._gguf_display_name(path.name, quant_type, size_gb),
        )

    def _scan_safetensors(self, path: Path) -> DetectedModel:
        """Scan a safetensors file. Detect FP8 from companion config or header."""
        size_bytes = path.stat().st_size
        size_gb = round(size_bytes / (1024**3), 1)
        is_distilled = path.name.lower() in _KNOWN_DISTILLED
        fmt = self._detect_safetensors_format(path)

        return DetectedModel(
            filename=path.name,
            path=str(path),
            format=fmt,
            quant_type="fp8" if fmt == "fp8" else None,
            size_bytes=size_bytes,
            size_gb=size_gb,
            is_distilled=is_distilled,
            display_name=f"LTX 2.3 — {fmt.upper()} ({size_gb} GB)",
        )

    def _scan_nf4_folder(self, folder: Path) -> DetectedModel | None:
        """Check if a subfolder contains an NF4 quantized model."""
        config_path = folder / "quantize_config.json"
        if not config_path.exists():
            return None

        try:
            config = json.loads(config_path.read_text())
        except (json.JSONDecodeError, OSError):
            return None

        quant_type = config.get("quant_type")
        if quant_type != "nf4":
            return None

        size_bytes = sum(f.stat().st_size for f in folder.rglob("*") if f.is_file())
        size_gb = round(size_bytes / (1024**3), 1)

        return DetectedModel(
            filename=folder.name,
            path=str(folder),
            format="nf4",
            quant_type="nf4",
            size_bytes=size_bytes,
            size_gb=size_gb,
            is_distilled=False,
            display_name=f"LTX 2.3 — NF4 ({size_gb} GB)",
        )

    def _detect_safetensors_format(self, path: Path) -> str:
        """Detect whether a safetensors file is BF16 or FP8.

        Checks companion config.json first, then falls back to reading
        the safetensors header for dtype metadata.
        """
        # Check companion config.json
        config_path = path.parent / "config.json"
        if config_path.exists():
            try:
                config = json.loads(config_path.read_text())
                dtype = config.get("torch_dtype", "")
                if "float8" in dtype or "fp8" in dtype:
                    return "fp8"
            except (json.JSONDecodeError, OSError):
                pass

        # Check safetensors header for dtype info
        try:
            with open(path, "rb") as f:
                header_size_bytes = f.read(8)
                if len(header_size_bytes) == 8:
                    header_size = struct.unpack("<Q", header_size_bytes)[0]
                    if header_size < 10_000_000:  # Sanity check: header < 10MB
                        header_bytes = f.read(header_size)
                        header_text = header_bytes.decode("utf-8", errors="ignore")
                        if "float8" in header_text or "F8_E4M3" in header_text:
                            return "fp8"
        except OSError:
            pass

        return "bf16"

    @staticmethod
    def _quant_type_from_filename(filename: str) -> str | None:
        """Extract quantization type from common GGUF naming conventions."""
        name_upper = filename.upper()
        for qt in ("Q8_0", "Q6_K", "Q5_K", "Q5_1", "Q5_0", "Q4_K", "Q4_1", "Q4_0", "Q3_K", "Q2_K"):
            if qt in name_upper:
                return qt
        return None

    @staticmethod
    def _gguf_display_name(filename: str, quant_type: str | None, size_gb: float) -> str:
        qt = quant_type or "unknown quant"
        return f"LTX 2.3 — GGUF {qt} ({size_gb} GB)"
```

- [ ] **Step 6: Add re-export in interfaces.py**

In `backend/services/interfaces.py`, add this import at the top (after the PaletteSyncClient import):

```python
from services.model_scanner.model_scanner import ModelScanner
```

And add `"ModelScanner"` to the `__all__` list.

- [ ] **Step 7: Add FakeModelScanner to test fakes**

In `backend/tests/fakes/services.py`, add `DetectedModel` to the imports from `api_types`, then add this class (before the `FakeServices` dataclass at the bottom):

```python
class FakeModelScanner:
    def __init__(self) -> None:
        self._models: list[DetectedModel] = []

    def set_models(self, models: list[DetectedModel]) -> None:
        self._models = models

    def scan_video_models(self, folder: Path) -> list[DetectedModel]:
        return self._models
```

Also add `model_scanner: FakeModelScanner = field(default_factory=FakeModelScanner)` to the `FakeServices` dataclass.

- [ ] **Step 8: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/test_model_scanner.py -v --tb=short`
Expected: All 7 tests PASS

- [ ] **Step 9: Run pyright**

Run: `cd backend && uv run pyright`
Expected: `0 errors, 0 warnings, 0 informations`

- [ ] **Step 10: Commit**

```bash
cd D:/git/directors-desktop
git add backend/services/model_scanner/ backend/services/interfaces.py backend/tests/fakes/services.py backend/tests/test_model_scanner.py
git commit -m "feat(quantized-models): add ModelScanner service with Protocol, impl, fake, and tests"
```

---

### Task 3: Add model guide recommendation logic and tests

**Context:** The `recommend_format()` function maps GPU VRAM to a recommended format. This is simple logic but critical for the user experience — it's what tells users with a 3090 to download GGUF Q5_K.

**Files:**
- Already created: `backend/services/model_scanner/model_guide_data.py` (in Task 2)
- Create: `backend/tests/test_model_guide.py`

- [ ] **Step 1: Write the tests**

Create `backend/tests/test_model_guide.py`:

```python
"""Tests for model guide recommendation logic."""

from __future__ import annotations

from services.model_scanner.model_guide_data import MODEL_FORMATS, DISTILLED_LORA_INFO, recommend_format


class TestRecommendFormat:
    def test_48gb_recommends_bf16(self) -> None:
        assert recommend_format(48) == "bf16"

    def test_32gb_recommends_bf16(self) -> None:
        assert recommend_format(32) == "bf16"

    def test_24gb_recommends_fp8(self) -> None:
        assert recommend_format(24) == "fp8"

    def test_20gb_recommends_fp8(self) -> None:
        assert recommend_format(20) == "fp8"

    def test_16gb_recommends_gguf_q5k(self) -> None:
        assert recommend_format(16) == "gguf_q5k"

    def test_12gb_recommends_gguf_q4k(self) -> None:
        assert recommend_format(12) == "gguf_q4k"

    def test_10gb_recommends_gguf_q4k(self) -> None:
        assert recommend_format(10) == "gguf_q4k"

    def test_8gb_recommends_api_only(self) -> None:
        assert recommend_format(8) == "api_only"

    def test_none_vram_defaults_to_bf16(self) -> None:
        assert recommend_format(None) == "bf16"


class TestModelFormatsData:
    def test_all_formats_have_required_fields(self) -> None:
        for fmt in MODEL_FORMATS:
            assert fmt.id
            assert fmt.name
            assert fmt.min_vram_gb > 0
            assert fmt.download_url.startswith("https://")

    def test_distilled_lora_info_has_url(self) -> None:
        assert DISTILLED_LORA_INFO.download_url.startswith("https://")
        assert DISTILLED_LORA_INFO.size_gb > 0
```

- [ ] **Step 2: Run tests**

Run: `cd backend && uv run pytest tests/test_model_guide.py -v --tb=short`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
cd D:/git/directors-desktop
git add backend/tests/test_model_guide.py
git commit -m "test(quantized-models): add model guide recommendation tests"
```

---

### Task 4: Wire ModelScanner into handler + routes + integration tests

**Context:** Now connect the scanner to the API layer. Add `scan_video_models()`, `select_video_model()`, and `video_model_guide()` to `ModelsHandler`, add thin routes, and wire `ModelScannerImpl` into `ServiceBundle` and `AppHandler`. The test infrastructure (`conftest.py`) needs to wire `FakeModelScanner`.

**Files:**
- Modify: `backend/handlers/models_handler.py` — add 3 new methods
- Modify: `backend/_routes/models.py` — add 3 new routes
- Modify: `backend/app_handler.py` — add `ModelScanner` to `AppHandler.__init__()` and `ServiceBundle`, wire `ModelScannerImpl` in `build_default_service_bundle()`
- Modify: `backend/tests/conftest.py` — wire `FakeModelScanner` in test `ServiceBundle`
- Create: `backend/tests/test_model_selection.py` — integration tests

- [ ] **Step 1: Write the failing integration test**

Create `backend/tests/test_model_selection.py`:

```python
"""Integration tests for video model scan, select, and guide endpoints."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from app_handler import AppHandler


def _write_minimal_gguf(path: Path) -> None:
    with open(path, "wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", 3))
        f.write(struct.pack("<Q", 0))
        f.write(struct.pack("<Q", 0))


class TestVideoModelScan:
    def test_scan_returns_empty_for_default_path(self, client) -> None:
        resp = client.get("/api/models/video/scan")
        assert resp.status_code == 200
        data = resp.json()
        assert data["models"] == []
        assert data["distilled_lora_found"] is False


class TestVideoModelSelect:
    def test_select_nonexistent_model_returns_400(self, client) -> None:
        resp = client.post("/api/models/video/select", json={"model": "nonexistent.gguf"})
        assert resp.status_code == 400

    def test_select_valid_model_updates_settings(self, client, test_state: AppHandler) -> None:
        # Create a model file in the models dir
        models_dir = test_state.config.models_dir
        gguf_path = models_dir / "test-model.gguf"
        _write_minimal_gguf(gguf_path)

        # First set the custom path to models_dir
        client.post("/api/settings", json={"customVideoModelPath": str(models_dir)})

        resp = client.post("/api/models/video/select", json={"model": "test-model.gguf"})
        assert resp.status_code == 200

        # Verify settings updated
        assert test_state.state.app_settings.selected_video_model == "test-model.gguf"


class TestVideoModelGuide:
    def test_guide_returns_formats_and_recommendation(self, client) -> None:
        resp = client.get("/api/models/video/guide")
        assert resp.status_code == 200
        data = resp.json()
        assert "formats" in data
        assert "recommended_format" in data
        assert "distilled_lora" in data
        assert len(data["formats"]) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_model_selection.py -v --tb=short`
Expected: FAIL — routes don't exist yet

- [ ] **Step 3: Update ModelsHandler**

In `backend/handlers/models_handler.py`, update the imports:

```python
from __future__ import annotations

import logging
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING

from api_types import (
    DetectedModel,
    ModelFileStatus,
    ModelInfo,
    ModelsStatusResponse,
    TextEncoderStatus,
    VideoModelGuideResponse,
    VideoModelScanResponse,
)
from handlers.base import StateHandlerBase, with_state_lock
from runtime_config.model_download_specs import MODEL_FILE_ORDER, resolve_required_model_types
from services.model_scanner.model_guide_data import DISTILLED_LORA_INFO, MODEL_FORMATS, recommend_format
from state.app_state_types import AppState, AvailableFiles

if TYPE_CHECKING:
    from runtime_config.runtime_config import RuntimeConfig
    from services.gpu_info.gpu_info import GpuInfo
    from services.model_scanner.model_scanner import ModelScanner

logger = logging.getLogger(__name__)
```

Update `ModelsHandler.__init__()` to accept `model_scanner`:

```python
class ModelsHandler(StateHandlerBase):
    def __init__(
        self,
        state: AppState,
        lock: RLock,
        config: RuntimeConfig,
        model_scanner: ModelScanner,
        gpu_info_service: GpuInfo,
    ) -> None:
        super().__init__(state, lock)
        self._config = config
        self._model_scanner = model_scanner
        self._gpu_info = gpu_info_service
```

Add these three methods to `ModelsHandler` (after `get_models_status()`):

```python
    def _get_video_models_dir(self) -> Path:
        """Get the folder to scan for video models."""
        custom = self.state.app_settings.custom_video_model_path
        if custom:
            return Path(custom)
        return self._config.models_dir

    def scan_video_models(self) -> VideoModelScanResponse:
        """Scan the configured folder for video model files."""
        folder = self._get_video_models_dir()
        models = self._model_scanner.scan_video_models(folder)

        # Check for distilled LoRA in the same folder or loras/ subfolder
        lora_found = self._check_distilled_lora(folder)

        return VideoModelScanResponse(
            models=models,
            distilled_lora_found=lora_found,
        )

    def select_video_model(self, filename: str) -> dict[str, str]:
        """Select a video model by filename. Validates it exists and no generation running."""
        from _routes._errors import HTTPError
        from state.app_state_types import GenerationRunning, GpuSlot

        # Guard: don't swap model while generating (matches _ensure_no_running_generation pattern)
        match self.state.gpu_slot:
            case GpuSlot(generation=GenerationRunning()):
                raise HTTPError(409, "Cannot change model while generation is running")

        folder = self._get_video_models_dir()
        target = folder / filename

        # Check file or folder exists
        if not target.exists():
            raise HTTPError(400, f"Model file not found: {filename}")

        self.state.app_settings.selected_video_model = filename
        return {"status": "ok", "selected": filename}

    def video_model_guide(self) -> VideoModelGuideResponse:
        """Return GPU info and format recommendations for the model guide UI."""
        gpu_name: str | None = None
        vram_gb: int | None = None
        try:
            gpu_name = self._gpu_info.get_device_name()
            vram_gb = self._gpu_info.get_vram_total_gb()
        except Exception:
            logger.warning("Could not get GPU info for model guide", exc_info=True)

        return VideoModelGuideResponse(
            gpu_name=gpu_name,
            vram_gb=vram_gb,
            recommended_format=recommend_format(vram_gb),
            formats=MODEL_FORMATS,
            distilled_lora=DISTILLED_LORA_INFO,
        )

    @staticmethod
    def _check_distilled_lora(folder: Path) -> bool:
        """Check if a distilled LoRA file exists in the folder or loras/ subfolder."""
        if not folder.exists():
            return False

        search_dirs = [folder]
        loras_sub = folder / "loras"
        if loras_sub.exists():
            search_dirs.append(loras_sub)

        try:
            for d in search_dirs:
                for f in d.iterdir():
                    if f.is_file() and "distill" in f.name.lower() and f.suffix in (".safetensors", ".bin"):
                        return True
        except OSError:
            pass
        return False
```

- [ ] **Step 4: Add routes**

In `backend/_routes/models.py`, add these imports at the top:

```python
from api_types import (
    DownloadProgressResponse,
    ModelDownloadRequest,
    ModelDownloadStartResponse,
    ModelInfo,
    ModelsStatusResponse,
    SelectModelRequest,
    TextEncoderDownloadResponse,
    VideoModelGuideResponse,
    VideoModelScanResponse,
)
```

Add these routes at the end of the file:

```python
@router.get("/models/video/scan", response_model=VideoModelScanResponse)
def route_scan_video_models(handler: AppHandler = Depends(get_state_service)) -> VideoModelScanResponse:
    return handler.models.scan_video_models()


@router.post("/models/video/select")
def route_select_video_model(
    req: SelectModelRequest,
    handler: AppHandler = Depends(get_state_service),
) -> dict[str, str]:
    result = handler.models.select_video_model(req.model)
    handler.settings.save_settings()
    return result


@router.get("/models/video/guide", response_model=VideoModelGuideResponse)
def route_video_model_guide(handler: AppHandler = Depends(get_state_service)) -> VideoModelGuideResponse:
    return handler.models.video_model_guide()
```

- [ ] **Step 5: Update AppHandler and ServiceBundle**

In `backend/app_handler.py`:

Add to imports:
```python
from services.model_scanner.model_scanner import ModelScanner
```

Add `model_scanner: ModelScanner` parameter to `AppHandler.__init__()` (after `ic_lora_model_downloader`):
```python
        model_scanner: ModelScanner,
```

Store it:
```python
        self.model_scanner = model_scanner
```

Update the `self.models = ModelsHandler(...)` call to pass scanner and gpu_info:
```python
        self.models = ModelsHandler(
            state=self.state,
            lock=self._lock,
            config=config,
            model_scanner=model_scanner,
            gpu_info_service=gpu_info,
        )
```

Add `model_scanner: ModelScanner` to the `ServiceBundle` dataclass (after `ic_lora_model_downloader`):
```python
    model_scanner: ModelScanner
```

Update `build_default_service_bundle()` — add import:
```python
    from services.model_scanner.model_scanner_impl import ModelScannerImpl
```

Add to the returned `ServiceBundle(...)`:
```python
        model_scanner=ModelScannerImpl(),
```

Update `build_initial_state()` — pass `model_scanner=bundle.model_scanner` to `AppHandler(...)`.

- [ ] **Step 6: Update conftest.py**

In `backend/tests/conftest.py`, add `model_scanner=fake_services.model_scanner` to the `ServiceBundle(...)` constructor call.

- [ ] **Step 7: Run integration tests**

Run: `cd backend && uv run pytest tests/test_model_selection.py -v --tb=short`
Expected: All 4 tests PASS

- [ ] **Step 8: Run full test suite + pyright**

Run: `cd backend && uv run pyright && uv run pytest -v --tb=short`
Expected: pyright clean, all tests pass

- [ ] **Step 9: Commit**

```bash
cd D:/git/directors-desktop
git add backend/handlers/models_handler.py backend/_routes/models.py backend/app_handler.py backend/tests/conftest.py backend/tests/test_model_selection.py
git commit -m "feat(quantized-models): wire ModelScanner to handler, routes, and integration tests"
```

---

### Task 5: Add Models tab to SettingsModal

**Context:** Add a new "Models" tab to the existing SettingsModal. This tab shows: (1) a dropdown of detected model files, (2) the model folder path with Change/Open/Scan buttons, (3) GPU info summary, (4) a "Model Guide" button, and (5) distilled LoRA status. The tab calls the 3 new API endpoints from Task 4.

**Files:**
- Modify: `frontend/components/SettingsModal.tsx` — add `models` tab
- Modify: `frontend/contexts/AppSettingsContext.tsx` — if needed, add new settings fields to TypeScript types

- [ ] **Step 1: Add TypeScript types for new settings**

In `frontend/contexts/AppSettingsContext.tsx`:

Add these fields to the `AppSettings` interface (after `hasCivitaiApiKey`):

```typescript
customVideoModelPath: string
selectedVideoModel: string
```

Add these default values to the `DEFAULT_APP_SETTINGS` constant (after `hasCivitaiApiKey: false`):

```typescript
customVideoModelPath: '',
selectedVideoModel: '',
```

- [ ] **Step 2: Add Models tab to SettingsModal**

In `frontend/components/SettingsModal.tsx`:

Update the `TabId` type:
```typescript
type TabId = 'general' | 'apiKeys' | 'inference' | 'models' | 'promptEnhancer' | 'about'
```

Add `Cpu` icon to the lucide imports (for GPU info display).

Add state variables for the Models tab (inside the `SettingsModal` component, near the other state):

```typescript
const [videoModels, setVideoModels] = useState<any[]>([])
const [distilledLoraFound, setDistilledLoraFound] = useState(false)
const [modelScanning, setModelScanning] = useState(false)
const [gpuInfo, setGpuInfo] = useState<{ name: string | null; vram: number | null } | null>(null)
const [showModelGuide, setShowModelGuide] = useState(false)
```

Add a `useEffect` to load model data when the Models tab is active:

```typescript
useEffect(() => {
  if (!isOpen || activeTab !== 'models') return
  let cancelled = false
  const load = async () => {
    try {
      const backendUrl = await window.electronAPI.getBackendUrl()
      const [scanRes, guideRes] = await Promise.all([
        fetch(`${backendUrl}/api/models/video/scan`),
        fetch(`${backendUrl}/api/models/video/guide`),
      ])
      if (cancelled) return
      if (scanRes.ok) {
        const data = await scanRes.json()
        setVideoModels(data.models)
        setDistilledLoraFound(data.distilled_lora_found)
      }
      if (guideRes.ok) {
        const guide = await guideRes.json()
        setGpuInfo({ name: guide.gpu_name, vram: guide.vram_gb })
      }
    } catch (err) {
      logger.error('Failed to load model data', err)
    }
  }
  load()
  return () => { cancelled = true }
}, [isOpen, activeTab])
```

Add the Models tab button to the tab bar (between Inference and About):
```tsx
<button
  className={`px-3 py-2 text-xs font-medium rounded-lg transition-colors ${
    activeTab === 'models'
      ? 'bg-purple-600/20 text-purple-400 border border-purple-500/30'
      : 'text-zinc-400 hover:text-zinc-300 hover:bg-zinc-800'
  }`}
  onClick={() => setActiveTab('models')}
>
  <Cpu className="w-3.5 h-3.5 inline mr-1.5" />
  Models
</button>
```

Add the Models tab content (inside the tab content switch):

```tsx
{activeTab === 'models' && (
  <div className="space-y-6">
    {/* GPU Info */}
    {gpuInfo && (
      <div className="bg-zinc-800/50 rounded-lg p-4 border border-zinc-700/50">
        <div className="flex items-center gap-2 text-sm">
          <Cpu className="w-4 h-4 text-purple-400" />
          <span className="text-zinc-300">
            {gpuInfo.name || 'Unknown GPU'}
            {gpuInfo.vram && ` — ${gpuInfo.vram} GB VRAM`}
          </span>
        </div>
      </div>
    )}

    {/* Video Model Selection */}
    <div>
      <label className="block text-xs font-medium text-zinc-300 mb-2">Video Model</label>
      <select
        className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-sm text-zinc-200"
        value={settings.selectedVideoModel || ''}
        onChange={async (e) => {
          const model = e.target.value
          if (!model) return
          try {
            const backendUrl = await window.electronAPI.getBackendUrl()
            await fetch(`${backendUrl}/api/models/video/select`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ model }),
            })
            onSettingsChange({ ...settings, selectedVideoModel: model })
          } catch (err) {
            logger.error('Failed to select model', err)
          }
        }}
      >
        <option value="">Default (BF16)</option>
        {videoModels.map((m: any) => (
          <option key={m.filename} value={m.filename}>
            {m.display_name}
          </option>
        ))}
      </select>
    </div>

    {/* Model Folder */}
    <div>
      <label className="block text-xs font-medium text-zinc-300 mb-2">Model Folder</label>
      <div className="flex gap-2">
        <input
          className="flex-1 bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-sm text-zinc-400"
          value={settings.customVideoModelPath || 'Default models directory'}
          readOnly
        />
        <Button
          variant="outline"
          size="sm"
          onClick={async () => {
            const dir = await window.electronAPI?.showOpenDirectoryDialog({ title: 'Select Video Models Folder' })
            if (dir) {
              onSettingsChange({ ...settings, customVideoModelPath: dir })
            }
          }}
        >
          Change
        </Button>
        <Button
          variant="outline"
          size="sm"
          onClick={() => {
            const folder = settings.customVideoModelPath
            if (folder) {
              window.electronAPI?.showItemInFolder(folder)
            }
          }}
        >
          Open Folder
        </Button>
        <Button
          variant="outline"
          size="sm"
          onClick={async () => {
            setModelScanning(true)
            try {
              const backendUrl = await window.electronAPI.getBackendUrl()
              const res = await fetch(`${backendUrl}/api/models/video/scan`)
              if (res.ok) {
                const data = await res.json()
                setVideoModels(data.models)
                setDistilledLoraFound(data.distilled_lora_found)
              }
            } finally {
              setModelScanning(false)
            }
          }}
        >
          {modelScanning ? 'Scanning...' : 'Scan'}
        </Button>
      </div>
    </div>

    {/* Distilled LoRA Status */}
    {settings.selectedVideoModel && !distilledLoraFound && (
      <div className="bg-amber-500/10 border border-amber-500/30 rounded-lg p-3">
        <p className="text-xs text-amber-300">
          <AlertCircle className="w-3.5 h-3.5 inline mr-1" />
          This model may need a distilled LoRA for fast generation.
          Check the Model Guide for download links.
        </p>
      </div>
    )}

    {/* Model Guide Button */}
    <Button
      className="w-full bg-purple-600 hover:bg-purple-700 text-white"
      onClick={() => setShowModelGuide(true)}
    >
      <Info className="w-4 h-4 mr-2" />
      Open Model Guide
    </Button>
  </div>
)}
```

- [ ] **Step 3: Run TypeScript typecheck**

Run: `pnpm typecheck:ts`
Expected: No errors

- [ ] **Step 4: Commit**

```bash
cd D:/git/directors-desktop
git add frontend/components/SettingsModal.tsx frontend/contexts/AppSettingsContext.tsx
git commit -m "feat(quantized-models): add Models tab to SettingsModal"
```

---

### Task 6: Create ModelGuideDialog component

**Context:** A standalone modal dialog that shows GPU-based recommendations and download links for each model format. Fetches data from `GET /api/models/video/guide`. Design follows the CLAUDE.md design standards (OKLCH colors, rounded corners, hover states, etc).

**Files:**
- Create: `frontend/components/ModelGuideDialog.tsx`
- Modify: `frontend/components/SettingsModal.tsx` — import and render ModelGuideDialog

- [ ] **Step 1: Create ModelGuideDialog.tsx**

Create `frontend/components/ModelGuideDialog.tsx` with the full dialog implementation. It should:

- Fetch from `/api/models/video/guide` on mount
- Show GPU name + VRAM at the top
- Grid of format cards (responsive: 1 col on small, 2 on medium+)
- Each card: format name, size, quality tier badge, "Recommended" pill if it matches recommended_format, "Download" button that opens URL in browser via `window.electronAPI.openExternal` or `window.open`
- Distilled LoRA callout at the bottom (amber background) with its own download button
- Model folder path display
- Close button

Style with Tailwind using the project's dark theme (`bg-zinc-900`, `border-zinc-700`, `text-zinc-300`, purple accents for recommended items).

```tsx
import { AlertCircle, Download, ExternalLink, Monitor, X } from 'lucide-react'
import React, { useEffect, useState } from 'react'
import { Button } from './ui/button'
import { logger } from '../lib/logger'

interface ModelFormat {
  id: string
  name: string
  size_gb: number
  min_vram_gb: number
  quality_tier: string
  needs_distilled_lora: boolean
  download_url: string
  description: string
}

interface DistilledLora {
  name: string
  size_gb: number
  download_url: string
  description: string
}

interface GuideData {
  gpu_name: string | null
  vram_gb: number | null
  recommended_format: string
  formats: ModelFormat[]
  distilled_lora: DistilledLora
}

interface ModelGuideDialogProps {
  isOpen: boolean
  onClose: () => void
}

const QUALITY_COLORS: Record<string, string> = {
  'Best': 'bg-green-500/20 text-green-400 border-green-500/30',
  'Excellent': 'bg-blue-500/20 text-blue-400 border-blue-500/30',
  'Very Good': 'bg-purple-500/20 text-purple-400 border-purple-500/30',
  'Good': 'bg-amber-500/20 text-amber-400 border-amber-500/30',
}

export function ModelGuideDialog({ isOpen, onClose }: ModelGuideDialogProps) {
  const [data, setData] = useState<GuideData | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    if (!isOpen) return
    let cancelled = false
    setLoading(true)
    const load = async () => {
      try {
        const backendUrl = await window.electronAPI.getBackendUrl()
        const res = await fetch(`${backendUrl}/api/models/video/guide`)
        if (!cancelled && res.ok) {
          setData(await res.json())
        }
      } catch (err) {
        logger.error('Failed to load model guide', err)
      } finally {
        if (!cancelled) setLoading(false)
      }
    }
    load()
    return () => { cancelled = true }
  }, [isOpen])

  if (!isOpen) return null

  const openUrl = (url: string) => {
    window.open(url, '_blank')
  }

  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div className="bg-zinc-900 border border-zinc-700 rounded-xl w-full max-w-2xl max-h-[85vh] overflow-y-auto shadow-2xl">
        {/* Header */}
        <div className="flex items-center justify-between p-5 border-b border-zinc-800">
          <h2 className="text-lg font-semibold text-white">Video Model Guide</h2>
          <button onClick={onClose} className="text-zinc-400 hover:text-zinc-200 transition-colors">
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="p-5 space-y-5">
          {loading ? (
            <div className="text-center text-zinc-400 py-8">Loading...</div>
          ) : data ? (
            <>
              {/* GPU Info Banner */}
              <div className="bg-zinc-800/60 rounded-lg p-4 border border-zinc-700/50 flex items-center gap-3">
                <Monitor className="w-5 h-5 text-purple-400 shrink-0" />
                <div>
                  <p className="text-sm font-medium text-white">
                    {data.gpu_name || 'GPU not detected'}
                  </p>
                  <p className="text-xs text-zinc-400">
                    {data.vram_gb
                      ? `${data.vram_gb} GB VRAM available`
                      : 'VRAM could not be determined'}
                  </p>
                </div>
              </div>

              {/* Format Cards */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {data.formats.map((fmt) => {
                  const isRecommended = fmt.id === data.recommended_format
                  return (
                    <div
                      key={fmt.id}
                      className={`relative rounded-lg p-4 border transition-all ${
                        isRecommended
                          ? 'bg-purple-500/10 border-purple-500/40 ring-1 ring-purple-500/20'
                          : 'bg-zinc-800/40 border-zinc-700/50 hover:border-zinc-600'
                      }`}
                    >
                      {isRecommended && (
                        <span className="absolute -top-2 right-3 bg-purple-600 text-white text-[10px] font-bold px-2 py-0.5 rounded-full uppercase tracking-wider">
                          Recommended
                        </span>
                      )}
                      <h3 className="text-sm font-medium text-white mb-1">{fmt.name}</h3>
                      <div className="flex items-center gap-2 mb-2">
                        <span className="text-xs text-zinc-400">{fmt.size_gb} GB</span>
                        <span className="text-zinc-600">·</span>
                        <span className={`text-[10px] px-1.5 py-0.5 rounded border ${QUALITY_COLORS[fmt.quality_tier] || 'bg-zinc-700 text-zinc-300'}`}>
                          {fmt.quality_tier}
                        </span>
                        <span className="text-zinc-600">·</span>
                        <span className="text-xs text-zinc-500">≥{fmt.min_vram_gb} GB VRAM</span>
                      </div>
                      <p className="text-xs text-zinc-400 mb-3">{fmt.description}</p>
                      <Button
                        variant="outline"
                        size="sm"
                        className="w-full text-xs"
                        onClick={() => openUrl(fmt.download_url)}
                      >
                        <ExternalLink className="w-3 h-3 mr-1.5" />
                        Download from HuggingFace
                      </Button>
                    </div>
                  )
                })}
              </div>

              {/* Distilled LoRA Notice */}
              <div className="bg-amber-500/10 border border-amber-500/30 rounded-lg p-4">
                <div className="flex items-start gap-3">
                  <AlertCircle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
                  <div className="flex-1">
                    <p className="text-sm font-medium text-amber-300 mb-1">{data.distilled_lora.name}</p>
                    <p className="text-xs text-amber-200/70 mb-2">{data.distilled_lora.description}</p>
                    <Button
                      variant="outline"
                      size="sm"
                      className="text-xs border-amber-500/40 text-amber-300 hover:bg-amber-500/10"
                      onClick={() => openUrl(data.distilled_lora.download_url)}
                    >
                      <Download className="w-3 h-3 mr-1.5" />
                      Download LoRA ({data.distilled_lora.size_gb} GB)
                    </Button>
                  </div>
                </div>
              </div>

              {/* Instructions */}
              <div className="bg-zinc-800/40 rounded-lg p-4 border border-zinc-700/50">
                <h4 className="text-xs font-medium text-zinc-300 mb-2">Setup Instructions</h4>
                <ol className="text-xs text-zinc-400 space-y-1 list-decimal list-inside">
                  <li>Download the model file for your GPU from the links above</li>
                  <li>Go to Settings → Models and set your model folder</li>
                  <li>If using GGUF or NF4, also download the distilled LoRA</li>
                  <li>Select your model from the dropdown</li>
                  <li>Generate!</li>
                </ol>
              </div>
            </>
          ) : (
            <div className="text-center text-zinc-400 py-8">Failed to load model guide data.</div>
          )}
        </div>
      </div>
    </div>
  )
}
```

- [ ] **Step 2: Import and render in SettingsModal**

In `frontend/components/SettingsModal.tsx`, add import:
```typescript
import { ModelGuideDialog } from './ModelGuideDialog'
```

Render the dialog at the end of the SettingsModal component's return, just before the closing `</>`:
```tsx
<ModelGuideDialog isOpen={showModelGuide} onClose={() => setShowModelGuide(false)} />
```

- [ ] **Step 3: Run TypeScript typecheck**

Run: `pnpm typecheck:ts`
Expected: No errors

- [ ] **Step 4: Commit**

```bash
cd D:/git/directors-desktop
git add frontend/components/ModelGuideDialog.tsx frontend/components/SettingsModal.tsx
git commit -m "feat(quantized-models): add ModelGuideDialog popup with GPU recommendations"
```

---

### Task 7: Update PipelinesHandler to support multiple pipeline classes

**Context:** Currently `PipelinesHandler._create_video_pipeline()` always uses `self._fast_video_pipeline_class`. We need it to check `selected_video_model` from settings, determine the file format, and use the appropriate pipeline class. For now, we wire the logic but use the existing `LTXFastVideoPipeline` for all safetensors files. The actual GGUF and NF4 pipeline classes will be added in Tasks 8 and 9.

**Files:**
- Modify: `backend/handlers/pipelines_handler.py` — update `_create_video_pipeline()` to route by format
- Modify: `backend/app_handler.py` — pass additional pipeline classes to PipelinesHandler

- [ ] **Step 1: Update PipelinesHandler constructor**

In `backend/handlers/pipelines_handler.py`, add `gguf_video_pipeline_class` and `nf4_video_pipeline_class` parameters to `__init__()`:

```python
    def __init__(
        self,
        state: AppState,
        lock: RLock,
        text_handler: TextHandler,
        gpu_cleaner: GpuCleaner,
        fast_video_pipeline_class: type[FastVideoPipeline],
        gguf_video_pipeline_class: type[FastVideoPipeline] | None,
        nf4_video_pipeline_class: type[FastVideoPipeline] | None,
        image_generation_pipeline_class: type[ImageGenerationPipeline],
        # ... rest unchanged
    ) -> None:
        # ... existing code ...
        self._gguf_video_pipeline_class = gguf_video_pipeline_class
        self._nf4_video_pipeline_class = nf4_video_pipeline_class
```

- [ ] **Step 2: Update _create_video_pipeline() to route by format**

In `_create_video_pipeline()`, replace the line that calls `self._fast_video_pipeline_class.create(...)` with format-aware routing:

```python
    def _create_video_pipeline(
        self,
        model_type: VideoPipelineModelType,
        lora_path: str | None = None,
        lora_weight: float = 1.0,
    ) -> VideoPipelineState:
        gemma_root = self._text_handler.resolve_gemma_root()

        # Determine checkpoint path and pipeline class based on selected model
        selected = self.state.app_settings.selected_video_model
        custom_dir = self.state.app_settings.custom_video_model_path
        pipeline_class = self._fast_video_pipeline_class

        if selected:
            base_dir = Path(custom_dir) if custom_dir else self._config.models_dir
            model_path = base_dir / selected
            checkpoint_path = str(model_path)

            # Validate model file/folder still exists on disk
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Selected model not found: {checkpoint_path}. "
                    "Go to Settings → Models to select a different model."
                )

            if selected.endswith(".gguf") and self._gguf_video_pipeline_class is not None:
                pipeline_class = self._gguf_video_pipeline_class
            elif model_path.is_dir() and self._nf4_video_pipeline_class is not None:
                # NF4 models are folders
                pipeline_class = self._nf4_video_pipeline_class
            # else: safetensors files use default pipeline
        else:
            checkpoint_path = str(self._config.model_path("checkpoint"))

        upsampler_path = str(self._config.model_path("upsampler"))

        pipeline = pipeline_class.create(
            checkpoint_path,
            gemma_root,
            upsampler_path,
            self._device,
            lora_path=lora_path,
            lora_weight=lora_weight,
        )

        state = VideoPipelineState(
            pipeline=pipeline,
            warmth=VideoPipelineWarmth.COLD,
            is_compiled=False,
            lora_path=lora_path,
        )
        state = self._compile_if_enabled(state)

        # Apply FFN chunking if enabled and torch.compile is not active
        chunk_count = self.state.app_settings.ffn_chunk_count
        if chunk_count > 0 and not state.is_compiled:
            try:
                transformer: torch.nn.Module = state.pipeline.pipeline.model_ledger.transformer()  # type: ignore[union-attr]
                patch_ffn_chunking(transformer, num_chunks=chunk_count)  # pyright: ignore[reportUnknownArgumentType]
            except AttributeError:
                logger.debug("FFN chunking skipped — pipeline has no model_ledger")

        # Install TeaCache denoising loop patch
        tea_threshold = self.state.app_settings.tea_cache_threshold
        try:
            install_tea_cache_patch(tea_threshold)
        except (ImportError, AttributeError):
            logger.debug("TeaCache skipped — ltx_pipelines not available")

        return state
```

Add `from pathlib import Path` to the imports.

- [ ] **Step 3: Update AppHandler to pass new pipeline class params**

In `backend/app_handler.py`, update the `self.pipelines = PipelinesHandler(...)` call to include:

```python
            gguf_video_pipeline_class=None,  # Will be set in Task 8
            nf4_video_pipeline_class=None,   # Will be set in Task 9
```

- [ ] **Step 4: Run full test suite + pyright**

Run: `cd backend && uv run pyright && uv run pytest -v --tb=short`
Expected: pyright clean, all tests pass

- [ ] **Step 5: Commit**

```bash
cd D:/git/directors-desktop
git add backend/handlers/pipelines_handler.py backend/app_handler.py
git commit -m "feat(quantized-models): update PipelinesHandler to route by model format"
```

---

### Task 8: Create GGUF Video Pipeline

**Context:** This is the pipeline class that loads GGUF quantized LTX models using the `diffusers` GGUF quantizer infrastructure. It implements the `FastVideoPipeline` protocol so it can be swapped in by PipelinesHandler. This is the most complex task — it requires integrating the diffusers GGUF loading with the ltx_pipelines inference code.

**Note:** This pipeline cannot be fully tested without real GGUF model files and a GPU. The implementation is based on the diffusers GGUF quantizer API. Manual testing required.

**Files:**
- Create: `backend/services/fast_video_pipeline/gguf_fast_video_pipeline.py`
- Modify: `backend/app_handler.py` — wire the GGUF pipeline class

- [ ] **Step 1: Create the GGUF pipeline class**

Create `backend/services/fast_video_pipeline/gguf_fast_video_pipeline.py`:

```python
"""GGUF quantized LTX video pipeline.

Loads LTX-Video transformer weights from a GGUF file using the diffusers
GGUF quantizer, while loading VAE, text encoder, and upsampler normally.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from typing import Any, Final, cast

import torch

from api_types import ImageConditioningInput
from services.ltx_pipeline_common import default_tiling_config, encode_video_output, video_chunks_number
from services.services_utils import AudioOrNone, TilingConfigType

logger = logging.getLogger(__name__)


class GGUFFastVideoPipeline:
    """FastVideoPipeline implementation for GGUF quantized models.

    Uses diffusers GGUF quantizer to load quantized transformer weights.
    Falls back to error with install instructions if gguf package missing.
    """

    pipeline_kind: Final = "fast"

    @staticmethod
    def create(
        checkpoint_path: str,
        gemma_root: str | None,
        upsampler_path: str,
        device: torch.device,
        lora_path: str | None = None,
        lora_weight: float = 1.0,
    ) -> "GGUFFastVideoPipeline":
        return GGUFFastVideoPipeline(
            checkpoint_path=checkpoint_path,
            gemma_root=gemma_root,
            upsampler_path=upsampler_path,
            device=device,
            lora_path=lora_path,
            lora_weight=lora_weight,
        )

    def __init__(
        self,
        checkpoint_path: str,
        gemma_root: str | None,
        upsampler_path: str,
        device: torch.device,
        lora_path: str | None = None,
        lora_weight: float = 1.0,
    ) -> None:
        try:
            import gguf  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "GGUF model support requires the 'gguf' package. "
                "Install it with: pip install gguf>=0.10.0"
            ) from None

        # TODO: Implement GGUF model loading using diffusers GGUFQuantizer.
        # This requires understanding the exact diffusers API for loading
        # LTX-Video transformer from GGUF format. The implementation will:
        #
        # 1. Load the GGUF file using diffusers.quantizers.gguf
        # 2. Build the LTX transformer model architecture
        # 3. Load quantized weights into the transformer
        # 4. Load VAE, text encoder, and upsampler normally via ltx_pipelines
        # 5. Assemble into a pipeline that matches DistilledPipeline's interface
        #
        # For now, raise NotImplementedError until we can test with real model files.
        raise NotImplementedError(
            "GGUF pipeline loading is not yet fully implemented. "
            "This requires testing with real GGUF model files to validate the "
            "diffusers GGUF quantizer integration with ltx_pipelines."
        )

    def _run_inference(
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[ImageConditioningInput],
        tiling_config: TilingConfigType,
    ) -> tuple[torch.Tensor | Iterator[torch.Tensor], AudioOrNone]:
        raise NotImplementedError

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[ImageConditioningInput],
        output_path: str,
    ) -> None:
        tiling_config = default_tiling_config()
        video, audio = self._run_inference(
            prompt=prompt, seed=seed, height=height, width=width,
            num_frames=num_frames, frame_rate=frame_rate, images=images,
            tiling_config=tiling_config,
        )
        chunks = video_chunks_number(num_frames, tiling_config)
        encode_video_output(video=video, audio=audio, fps=int(frame_rate),
                          output_path=output_path, video_chunks_number_value=chunks)

    @torch.inference_mode()
    def warmup(self, output_path: str) -> None:
        try:
            self.generate(
                prompt="test warmup", seed=42, height=256, width=384,
                num_frames=9, frame_rate=8, images=[], output_path=output_path,
            )
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def compile_transformer(self) -> None:
        logger.info("Skipping torch.compile for GGUF pipeline — not supported with quantized weights")
```

- [ ] **Step 2: Wire in AppHandler**

In `backend/app_handler.py`, in `build_default_service_bundle()`, add import:
```python
    from services.fast_video_pipeline.gguf_fast_video_pipeline import GGUFFastVideoPipeline
```

In the `self.pipelines = PipelinesHandler(...)` call in `AppHandler.__init__`, update:
```python
            gguf_video_pipeline_class=gguf_video_pipeline_class,
```

Add `gguf_video_pipeline_class: type[FastVideoPipeline] | None` to `AppHandler.__init__()` params and `ServiceBundle`.

In `build_default_service_bundle()`, add to the returned bundle:
```python
        gguf_video_pipeline_class=GGUFFastVideoPipeline,
```

- [ ] **Step 3: Run pyright**

Run: `cd backend && uv run pyright`
Expected: `0 errors`

- [ ] **Step 4: Commit**

```bash
cd D:/git/directors-desktop
git add backend/services/fast_video_pipeline/gguf_fast_video_pipeline.py backend/app_handler.py
git commit -m "feat(quantized-models): add GGUF video pipeline scaffold (NotImplementedError until tested with real models)"
```

---

### Task 9: Create NF4 Video Pipeline

**Context:** NF4 pipeline using BitsAndBytes 4-bit quantization, following the same pattern as the existing FLUX Klein pipeline. Same situation as GGUF — scaffold with NotImplementedError until tested with real models.

**Files:**
- Create: `backend/services/fast_video_pipeline/nf4_fast_video_pipeline.py`
- Modify: `backend/app_handler.py` — wire the NF4 pipeline class

- [ ] **Step 1: Create the NF4 pipeline class**

Create `backend/services/fast_video_pipeline/nf4_fast_video_pipeline.py`:

```python
"""NF4 (4-bit BitsAndBytes) quantized LTX video pipeline.

Uses BitsAndBytes NF4 quantization to load the LTX transformer at 4-bit
precision, following the same pattern as FluxKleinImagePipeline.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from typing import Any, Final, cast

import torch

from api_types import ImageConditioningInput
from services.ltx_pipeline_common import default_tiling_config, encode_video_output, video_chunks_number
from services.services_utils import AudioOrNone, TilingConfigType

logger = logging.getLogger(__name__)


class NF4FastVideoPipeline:
    """FastVideoPipeline implementation for NF4 quantized models.

    Uses BitsAndBytes 4-bit quantization (same approach as FLUX Klein).
    """

    pipeline_kind: Final = "fast"

    @staticmethod
    def create(
        checkpoint_path: str,
        gemma_root: str | None,
        upsampler_path: str,
        device: torch.device,
        lora_path: str | None = None,
        lora_weight: float = 1.0,
    ) -> "NF4FastVideoPipeline":
        return NF4FastVideoPipeline(
            checkpoint_path=checkpoint_path,
            gemma_root=gemma_root,
            upsampler_path=upsampler_path,
            device=device,
            lora_path=lora_path,
            lora_weight=lora_weight,
        )

    def __init__(
        self,
        checkpoint_path: str,
        gemma_root: str | None,
        upsampler_path: str,
        device: torch.device,
        lora_path: str | None = None,
        lora_weight: float = 1.0,
    ) -> None:
        # TODO: Implement NF4 model loading using BitsAndBytes.
        # Pattern from FLUX Klein pipeline (flux_klein_pipeline.py):
        #
        # 1. Create BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
        # 2. Load LTX transformer with quantization config
        # 3. Load VAE and text encoder normally
        # 4. Enable model CPU offload for text encoder
        # 5. Assemble pipeline matching DistilledPipeline interface
        #
        # Requires testing with real NF4 quantized LTX model files.
        raise NotImplementedError(
            "NF4 pipeline loading is not yet fully implemented. "
            "This requires testing with real NF4 quantized model files to validate "
            "the BitsAndBytes integration with ltx_pipelines."
        )

    def _run_inference(
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[ImageConditioningInput],
        tiling_config: TilingConfigType,
    ) -> tuple[torch.Tensor | Iterator[torch.Tensor], AudioOrNone]:
        raise NotImplementedError

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[ImageConditioningInput],
        output_path: str,
    ) -> None:
        tiling_config = default_tiling_config()
        video, audio = self._run_inference(
            prompt=prompt, seed=seed, height=height, width=width,
            num_frames=num_frames, frame_rate=frame_rate, images=images,
            tiling_config=tiling_config,
        )
        chunks = video_chunks_number(num_frames, tiling_config)
        encode_video_output(video=video, audio=audio, fps=int(frame_rate),
                          output_path=output_path, video_chunks_number_value=chunks)

    @torch.inference_mode()
    def warmup(self, output_path: str) -> None:
        try:
            self.generate(
                prompt="test warmup", seed=42, height=256, width=384,
                num_frames=9, frame_rate=8, images=[], output_path=output_path,
            )
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def compile_transformer(self) -> None:
        logger.info("Skipping torch.compile for NF4 pipeline — not supported with quantized weights")
```

- [ ] **Step 2: Wire in AppHandler**

In `backend/app_handler.py`, in `build_default_service_bundle()`, add import:
```python
    from services.fast_video_pipeline.nf4_fast_video_pipeline import NF4FastVideoPipeline
```

Add `nf4_video_pipeline_class` to `AppHandler.__init__()` params and `ServiceBundle`.

Update `self.pipelines = PipelinesHandler(...)` call:
```python
            nf4_video_pipeline_class=nf4_video_pipeline_class,
```

In `build_default_service_bundle()`, add to returned bundle:
```python
        nf4_video_pipeline_class=NF4FastVideoPipeline,
```

- [ ] **Step 3: Run pyright + full tests**

Run: `cd backend && uv run pyright && uv run pytest -v --tb=short`
Expected: pyright clean, all tests pass

- [ ] **Step 4: Commit**

```bash
cd D:/git/directors-desktop
git add backend/services/fast_video_pipeline/nf4_fast_video_pipeline.py backend/app_handler.py
git commit -m "feat(quantized-models): add NF4 video pipeline scaffold"
```

---

### Task 10: Add README section for custom video models

**Context:** Add a "Custom Video Models" section to the README explaining what formats are supported, how to download, and how to set up. Keep it brief and user-friendly.

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Read the current README**

Read `README.md` to find the right place to add the section.

- [ ] **Step 2: Add Custom Video Models section**

Add this section to the README (after the main features/usage section, before any contribution or license section):

```markdown
## Custom Video Models

Directors Desktop supports multiple LTX 2.3 model formats, so you can run on GPUs with less VRAM.

| Your GPU VRAM | Recommended Format | File Size |
|---------------|-------------------|-----------|
| 32 GB+ | BF16 (auto-downloaded) | ~43 GB |
| 20-31 GB | [FP8 Checkpoint](https://huggingface.co/Lightricks/LTX-Video-2.3-22b-distilled) | ~22 GB |
| 16-19 GB | [GGUF Q5_K](https://huggingface.co/city96/LTX-Video-2.3-22b-0.9.7-dev-gguf) | ~15 GB |
| 10-15 GB | [GGUF Q4_K](https://huggingface.co/city96/LTX-Video-2.3-22b-0.9.7-dev-gguf) | ~12 GB |

### Setup

1. Download the model file for your GPU from the links above
2. Open **Settings → Models** and set your model folder
3. If using GGUF or NF4, also download the [distilled LoRA](https://huggingface.co/Lightricks/LTX-Video-2.3-22b-distilled)
4. Select your model from the dropdown
5. Generate!

The app also has a built-in **Model Guide** (Settings → Models → Open Model Guide) that detects your GPU and recommends the best format automatically.
```

- [ ] **Step 3: Commit**

```bash
cd D:/git/directors-desktop
git add README.md
git commit -m "docs(quantized-models): add Custom Video Models section to README"
```

---

## Summary of Tasks

| Task | Description | Dependencies |
|------|-------------|-------------|
| 1 | Add gguf dependency, settings fields, API types | None |
| 2 | Create ModelScanner service (Protocol + Impl + Fake + Tests) | Task 1 |
| 3 | Add model guide recommendation logic and tests | Task 2 |
| 4 | Wire scanner to handler + routes + integration tests | Tasks 2, 3 |
| 5 | Add Models tab to SettingsModal | Task 4 |
| 6 | Create ModelGuideDialog component | Task 5 |
| 7 | Update PipelinesHandler for format routing | Task 4 |
| 8 | Create GGUF Video Pipeline scaffold | Task 7 |
| 9 | Create NF4 Video Pipeline scaffold | Task 7 |
| 10 | Add README section | None (can run in parallel) |

**Note on Tasks 8 & 9:** The GGUF and NF4 pipeline classes are scaffolded with `NotImplementedError`. Completing the actual model loading requires downloading real quantized model files and testing on GPU hardware. The scaffold ensures the routing, settings, and UI all work end-to-end — when the pipeline `__init__` is implemented, everything else is already wired up.
