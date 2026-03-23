# Quantized Video Model Support — Design Spec

## Goal

Let Directors Desktop run on 24GB GPUs (RTX 3090, 4070 Ti Super, etc.) by supporting quantized LTX 2.3 model formats (GGUF, NF4, FP8 checkpoints) alongside the existing BF16. Make it dead simple for users to understand what they need, where to get it, and how to set it up.

## Architecture

Three new components:

1. **Model Scanner Service** (backend) — New service with Protocol + Fake that scans a folder for model files, detects format/quant type from file metadata (not file size heuristics), and returns structured results
2. **Extended Pipeline System** (backend) — New `FastVideoPipeline` implementations for GGUF and NF4 formats. The existing `PipelinesHandler` already accepts `type[FastVideoPipeline]` — we extend `_create_video_pipeline()` to pick the right class based on the selected model's format. The `FastVideoPipeline.create()` signature stays unchanged; format-specific config (quant type, BnB config) is passed via the checkpoint path pointing to the right file.
3. **Model Setup UI** (frontend) — New "Models" tab in SettingsModal + a "Model Guide" popup that recommends what to download based on GPU, with instructions in-app, in a popup, and in the GitHub README

## Supported Formats

| Format | Extension | Typical Size | Min VRAM | Quality | Speed | Needs Distilled LoRA? |
|--------|-----------|-------------|----------|---------|-------|-----------------------|
| BF16 (current default) | `.safetensors` | ~43 GB | 32 GB | Best | Baseline | No (already distilled) |
| FP8 Checkpoint | `.safetensors` | ~22 GB | ~20 GB | Excellent | ~Same | No (distilled version available) |
| GGUF Q8 | `.gguf` | ~22 GB | ~18 GB | Excellent | Slightly slower | Yes |
| GGUF Q5_K | `.gguf` | ~15 GB | ~13 GB | Very Good | Slightly slower | Yes |
| GGUF Q4_K | `.gguf` | ~12 GB | ~10 GB | Good | Slightly slower | Yes |
| NF4 (4-bit) | `.safetensors` + bnb config | ~12 GB | ~10 GB | Good | Slightly slower | Yes |

## User Experience

### Model Guide Popup

Triggered on:
- First app launch (after initial model download completes)
- User clicks "Model Guide" button in Models tab
- No usable video model detected in configured path

The popup is a friendly, visual guide — NOT a wall of text. It:

1. **Detects GPU** — Shows "You have: NVIDIA RTX 3090 (24 GB VRAM)" at the top
2. **Recommends a format** — Based on VRAM:
   - 32+ GB → "You can run the full BF16 model. You're all set!"
   - 20-31 GB → "We recommend the FP8 checkpoint for your GPU"
   - 16-19 GB → "We recommend GGUF Q8 or Q5_K for your GPU"
   - 10-15 GB → "We recommend GGUF Q4_K or NF4 for your GPU"
   - <10 GB → "Your GPU doesn't have enough VRAM for local generation. Use API mode instead."
3. **Shows download cards** — Each format as a card with:
   - Format name + file size
   - Quality rating (stars or bar)
   - "Download from HuggingFace" button (opens browser to exact file URL)
   - Recommended badge on the best fit for their GPU
4. **Shows where to put it** — Current model path displayed, with "Change Folder" button
5. **Distilled LoRA notice** — If they pick GGUF/NF4: "This model also needs the distilled LoRA file. Download it here: [link]" with its own download button

### Models Tab in Settings

New tab in SettingsModal (between "Inference" and "About"):

```
[General] [API Keys] [Inference] [Models] [About]
```

Contents:
- **Video Model** section:
  - Dropdown: lists all detected model files in the configured path with format/size info
  - Current selection highlighted, e.g. "LTX-2.3-Q4_K.gguf (12 GB) — GGUF Q4"
  - Model folder path with "Change" and "Open Folder" buttons
  - "Scan for Models" button (re-scans the folder)
  - "Model Guide" button (opens the popup)
- **Distilled LoRA** section (only shown when active model needs it):
  - Status: "Found" (green) or "Not found — required for this model" (red with download link)
  - Path display
- **GPU Info** section:
  - GPU name, VRAM total, VRAM currently used
  - Estimated VRAM usage for selected model

### README Section

New section in GitHub README:

```markdown
## Custom Video Models

Directors Desktop supports multiple LTX 2.3 model formats. Pick the one
that fits your GPU:

| Your GPU VRAM | Recommended Format | Download |
|---------------|-------------------|----------|
| 32 GB+ | BF16 (default, auto-downloaded) | Included |
| 20-31 GB | FP8 Checkpoint | [HuggingFace link] |
| 16-19 GB | GGUF Q8 or Q5_K | [HuggingFace link] |
| 10-15 GB | GGUF Q4_K | [HuggingFace link] |

### Setup
1. Download the model file for your GPU
2. Open Settings → Models → Change model folder (or drop it in the default folder)
3. If using GGUF or NF4, also download the distilled LoRA: [link]
4. Select your model from the dropdown
5. Generate!
```

## Backend Design

### New Types (`backend/api_types.py`)

```python
class DetectedModel(TypedDict):
    filename: str             # e.g. "LTX-2.3-Q4_K.gguf"
    path: str                 # Absolute path
    format: str               # "bf16" | "fp8" | "gguf" | "nf4"
    quant_type: str | None    # e.g. "Q4_K", "Q8_0", None for bf16/fp8
    size_bytes: int
    size_gb: float
    is_distilled: bool
    display_name: str         # Human-friendly, e.g. "LTX 2.3 — GGUF Q4_K (12 GB)"

class ModelFormatInfo(TypedDict):
    id: str
    name: str
    size_gb: float
    min_vram_gb: int
    quality_tier: str         # "Best" | "Excellent" | "Very Good" | "Good"
    needs_distilled_lora: bool
    download_url: str
    description: str
```

### Model Scanner Service (`backend/services/model_scanner/`)

New service with Protocol + real + fake implementations (follows codebase pattern):

**Protocol** (`model_scanner.py`):
```python
class ModelScanner(Protocol):
    def scan_video_models(self, folder: Path) -> list[DetectedModel]: ...
```

**Implementation** (`model_scanner_impl.py`):
```python
class ModelScannerImpl:
    def scan_video_models(self, folder: Path) -> list[DetectedModel]:
        """Scan folder for supported video model files."""
```

**Fake** (`tests/fakes/services.py`):
```python
class FakeModelScanner:
    def __init__(self) -> None:
        self._models: list[DetectedModel] = []
    def set_models(self, models: list[DetectedModel]) -> None:
        self._models = models
    def scan_video_models(self, folder: Path) -> list[DetectedModel]:
        return self._models
```

**Detection logic** (NO file size heuristics — uses file metadata):
- `.gguf` extension → read GGUF file header for quant type metadata (using `gguf` Python reader or raw struct parsing)
- `.safetensors` with companion `config.json` or `model_index.json` containing dtype info → BF16 or FP8
- `.safetensors` without companion config → read safetensors header metadata for dtype field
- Folder with `quantize_config.json` containing `"quant_type": "nf4"` → NF4
- **Corrupt/unreadable files**: skip with warning log, don't crash the scan

**Distilled LoRA detection**:
- Scans the same folder (and a `loras/` subfolder) for files matching known distilled LoRA filenames
- Returns `is_distilled: True` on the model if it's a known distilled checkpoint (the default BF16 and official FP8 distilled are pre-distilled)
- For GGUF/NF4 models: `is_distilled` is always `False` (they need the LoRA)

### Pipeline Changes

**No changes to `FastVideoPipeline` Protocol.** The `create()` signature stays the same:
```python
create(checkpoint_path, gemma_root, upsampler_path, device, lora_path, lora_weight)
```

Each new pipeline class handles its format-specific loading internally. The `checkpoint_path` points to the actual file (`.gguf` or `.safetensors`), and the pipeline detects the format from the file extension.

**GGUF Pipeline** (`backend/services/fast_video_pipeline/gguf_fast_video_pipeline.py`):
- Implements `FastVideoPipeline` protocol
- Uses `diffusers` GGUF quantizer infrastructure to load transformer weights
- Loads VAE and text encoder normally (small enough for any GPU)
- Handles distilled LoRA injection when `lora_path` is provided
- Implements `generate()`, `warmup()`, `compile_transformer()` matching existing interface

**NF4 Pipeline** (`backend/services/fast_video_pipeline/nf4_fast_video_pipeline.py`):
- Implements `FastVideoPipeline` protocol
- Uses BitsAndBytes `BnbQuantizationConfig` with `load_in_4bit=True, bnb_4bit_quant_type="nf4"`
- Same pattern as existing FLUX Klein pipeline (`flux_klein_pipeline.py`)
- Handles distilled LoRA injection

### PipelinesHandler Changes

`_create_video_pipeline()` currently hardcodes `self._fast_video_pipeline_class`. Updated to:

1. Read `selected_video_model` from `AppSettings`
2. If set, determine format from the selected model's file extension
3. Pick the right pipeline class: `.gguf` → `GGUFFastVideoPipeline`, NF4 folder → `NF4FastVideoPipeline`, else → existing `LTXFastVideoPipeline`
4. Call `.create()` with the selected model's path as `checkpoint_path`

The pipeline classes are injected via `ServiceBundle` (like the existing pattern), so tests can provide fakes.

### Settings Changes (`backend/state/app_settings.py`)

New fields:
```python
custom_video_model_path: str   # User-chosen folder for custom models, empty = use default models dir
selected_video_model: str      # Filename of selected model, empty = use default BF16 checkpoint
```

Also update `SettingsResponse` and `to_settings_response()` to include these new fields.

### Interaction with existing `video_model` setting

`video_model: str = "ltx-fast"` remains as-is — it selects the model *type* (fast vs pro mode).
`selected_video_model` selects which *checkpoint file* to use for that model type.
They are independent: `video_model` picks the mode, `selected_video_model` picks the weights.

### New API Endpoints

```
GET  /api/models/video/scan          → { models: DetectedModel[], distilled_lora_found: bool }
POST /api/models/video/select        → { model: str }  (filename — validated against scan results)
GET  /api/models/video/guide         → { gpu_name, vram_gb, recommended_format, formats: ModelFormatInfo[], distilled_lora: DistilledLoraInfo }
```

**Validation**: `POST /select` checks that the filename exists in the configured folder. Returns 400 if not found.

**Generation guard**: If a generation is running, `POST /select` returns 409 Conflict (uses existing `_ensure_no_running_generation()` pattern).

### Model Guide Data (`backend/services/model_scanner/model_guide_data.py`)

Static config file (not hardcoded in handler logic) containing format metadata:
```python
MODEL_FORMATS: list[ModelFormatInfo] = [
    {
        "id": "bf16",
        "name": "BF16 (Full Precision)",
        "size_gb": 43,
        "min_vram_gb": 32,
        "quality_tier": "Best",
        "needs_distilled_lora": False,
        "download_url": "https://huggingface.co/Lightricks/LTX-Video-2.3-22b-distilled/...",
        "description": "Best quality. Requires 32GB+ VRAM."
    },
    ...
]

DISTILLED_LORA_INFO = {
    "name": "LTX 2.3 Distilled LoRA",
    "size_gb": 0.5,
    "download_url": "https://huggingface.co/...",
    "description": "Required for GGUF and NF4 models to generate quickly."
}
```

This is a data file, easily updated when new models release — not buried in handler code.

### Routes (`backend/_routes/models.py`)

Thin routes delegating to handler (follows existing pattern):
```python
@router.get("/video/scan")
def scan_video_models(handler = Depends(get_state_service)):
    return handler.models.scan_video_models()

@router.post("/video/select")
def select_video_model(body: SelectModelRequest, handler = Depends(get_state_service)):
    return handler.models.select_video_model(body.model)

@router.get("/video/guide")
def video_model_guide(handler = Depends(get_state_service)):
    return handler.models.video_model_guide()
```
```

## Frontend Design

### Model Guide Popup Component

`frontend/components/ModelGuideDialog.tsx`

Visual design:
- Modal dialog (same style as SettingsModal)
- Top banner: GPU detection result with icon
- Grid of format cards (2 columns), each card shows:
  - Format name (bold)
  - File size badge
  - Quality bar (5 segments, filled proportionally)
  - Speed bar (5 segments)
  - "Recommended" pill badge (green, on best match)
  - "Download" button (opens HuggingFace URL in browser)
- Bottom section: model folder path + change button
- Distilled LoRA callout box (yellow/amber) when applicable

### Models Tab Component

Added to `SettingsModal.tsx` as new tab:

- Model dropdown (select from scanned models)
- Folder path display + Change/Open/Scan buttons
- GPU info summary
- "Open Model Guide" button
- Distilled LoRA status indicator

## Error Handling

- **Corrupt model files**: `ModelScannerImpl` catches read errors, logs warning, skips the file
- **Missing distilled LoRA**: UI shows amber warning with download link; generation still attempted (may be slow but won't crash)
- **Model selected but deleted from disk**: `PipelinesHandler` catches FileNotFoundError at load time, returns clear error to frontend; frontend shows "Model file not found — please re-scan" message
- **Selecting model during generation**: Returns 409 Conflict, frontend disables the dropdown while generating
- **GGUF package not installed**: `GGUFFastVideoPipeline.create()` catches ImportError, raises RuntimeError with "Install gguf package: pip install gguf>=0.10.0"

## Testing Strategy

### Backend Tests

- `test_model_scanner.py` — scan folder with mixed model files (real temp files with correct headers), verify format detection; test corrupt file handling; test empty folder; test distilled LoRA detection
- `test_model_guide.py` — verify GPU VRAM → recommended format mapping logic
- `test_model_selection.py` — integration test via TestClient: scan → select → verify settings updated; select nonexistent model → 400; select during generation → 409

Pipeline integration tests for GGUF/NF4 are deferred — they require actual model files and GPU. Manual testing against real quantized models will validate these.

### Frontend

No frontend tests currently exist in the project. The Model Guide and Models tab should be manually tested.

## File Map

### New Files
- `backend/services/model_scanner/model_scanner.py` — ModelScanner Protocol
- `backend/services/model_scanner/model_scanner_impl.py` — Real implementation
- `backend/services/model_scanner/__init__.py`
- `backend/services/model_scanner/model_guide_data.py` — Static format metadata config
- `backend/services/fast_video_pipeline/gguf_fast_video_pipeline.py` — GGUF pipeline
- `backend/services/fast_video_pipeline/nf4_fast_video_pipeline.py` — NF4 pipeline
- `backend/tests/test_model_scanner.py` — Model scanning tests
- `backend/tests/test_model_guide.py` — Guide recommendation tests
- `backend/tests/test_model_selection.py` — Selection integration tests
- `frontend/components/ModelGuideDialog.tsx` — Model Guide popup

### Modified Files
- `backend/api_types.py` — Add `DetectedModel`, `ModelFormatInfo`, `SelectModelRequest` types
- `backend/state/app_settings.py` — Add `custom_video_model_path`, `selected_video_model`; update `SettingsResponse` and `to_settings_response()`
- `backend/handlers/models_handler.py` — Add `scan_video_models()`, `select_video_model()`, `video_model_guide()` methods
- `backend/_routes/models.py` — Add 3 new routes under `/video/`
- `backend/app_handler.py` — Wire `ModelScannerImpl` into `ServiceBundle`, pass to `ModelsHandler`
- `backend/handlers/pipelines_handler.py` — Update `_create_video_pipeline()` to check `selected_video_model` setting and pick pipeline class by format
- `backend/services/interfaces.py` — Re-export `ModelScanner`
- `backend/tests/fakes/services.py` — Add `FakeModelScanner`
- `backend/tests/conftest.py` — Wire `FakeModelScanner` into test `ServiceBundle`
- `frontend/components/SettingsModal.tsx` — Add Models tab, Model Guide button
- `README.md` — Add Custom Video Models section

## Dependencies

- `gguf>=0.10.0` — needed to read GGUF file metadata and load quantized weights. Must be added to `backend/pyproject.toml`.
- `diffusers` GGUF quantizer — already installed, provides `GGUFQuantizer`, `GGUFLinear`, dequantization functions
- `bitsandbytes` — already installed, provides NF4 quantization (used by FLUX Klein)

## Out of Scope

- Auto-downloading quantized models (user downloads manually)
- Converting between formats
- Quantizing models locally
- Supporting non-LTX video models
- Image model quantization changes (FLUX Klein NF4 already works)
