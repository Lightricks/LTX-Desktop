# feat: Local Image Editing (img2img) via Z-Image-Turbo

**Date:** 2026-03-08
**Type:** Enhancement
**Complexity:** Medium — reuses existing model weights, GPU lifecycle, and queue system

---

## Overview

Add local image editing to Directors Desktop using `ZImageImg2ImgPipeline` from diffusers. Users select an image, write an editing prompt, adjust a strength slider, and get back an edited image. This reuses the **same Z-Image-Turbo model weights** already downloaded — no new model, no API costs.

Primary use case: edit frames before animating them into video.

```
User has an image (generated or imported)
  → Clicks "Edit" or drops an image into the editor
  → Writes a prompt: "make the sky a dramatic sunset"
  → Adjusts strength: 0.65 (moderate change)
  → Gets back the edited image
  → Animates it into video
```

## Technical Foundation

`ZImageImg2ImgPipeline` shares the same model architecture and weights as `ZImagePipeline` (text-to-image). The difference is purely in the pipeline class — img2img encodes the source image into latents, adds noise at a level determined by `strength`, then denoises from there.

**Key parameters:**
- `image`: PIL Image input
- `strength`: 0.0 (no change) to 1.0 (ignore input entirely). Default: **0.65**
- `guidance_scale`: Must be **0.0** for Turbo models
- `num_inference_steps`: **9** (yields 8 DiT forward passes)
- `torch_dtype`: `torch.bfloat16`

**Requires:** `diffusers >= 0.37.0`

**VRAM:** Same as text-to-image (~16GB). No additional memory for the source image encoding.

**LoRA:** Works identically — same `ZImageLoraLoaderMixin` base class.

---

## Phase 1: Backend Pipeline + API

### 1a. Extend the protocol

**`backend/services/image_generation_pipeline/image_generation_pipeline.py`**

Add `img2img` method to `ImageGenerationPipeline` protocol:

```python
def img2img(
    self,
    prompt: str,
    image: PILImageType,
    strength: float,
    height: int,
    width: int,
    guidance_scale: float,
    num_inference_steps: int,
    seed: int,
) -> ImagePipelineOutputLike: ...
```

### 1b. Implement img2img in ZIT pipeline

**`backend/services/image_generation_pipeline/zit_image_generation_pipeline.py`**

Create a second internal pipeline instance using `ZImageImg2ImgPipeline`. Both share the same model components (transformer, VAE, text encoder), so we don't double VRAM usage. The img2img pipeline wraps the same underlying model.

```python
from diffusers import ZImagePipeline, ZImageImg2ImgPipeline

class ZitImageGenerationPipeline:
    def __init__(self, model_path: str, device: str | None = None) -> None:
        self.pipeline = ZImagePipeline.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
        )
        # Create img2img pipeline sharing the same model components
        self._img2img_pipeline = ZImageImg2ImgPipeline(
            scheduler=self.pipeline.scheduler,
            vae=self.pipeline.vae,
            text_encoder=self.pipeline.text_encoder,
            tokenizer=self.pipeline.tokenizer,
            transformer=self.pipeline.transformer,
        )
        if device is not None:
            self.to(device)

    @torch.inference_mode()
    def img2img(
        self,
        prompt: str,
        image: PILImageType,
        strength: float,
        height: int,
        width: int,
        guidance_scale: float,
        num_inference_steps: int,
        seed: int,
    ) -> ImagePipelineOutputLike:
        generator = torch.Generator(
            device=self._resolve_generator_device()
        ).manual_seed(seed)
        output = self._img2img_pipeline(
            prompt=prompt,
            image=image,
            strength=strength,
            height=height,
            width=width,
            guidance_scale=0.0,  # Always 0.0 for Turbo
            num_inference_steps=num_inference_steps,
            generator=generator,
            output_type="pil",
            return_dict=True,
        )
        return self._normalize_output(output)
```

**Important:** The `to()` method must also move the img2img pipeline. Since they share components, calling `self.pipeline.enable_model_cpu_offload()` should cover both, but verify this. If not, call it on `self._img2img_pipeline` too.

### 1c. Extend API types

**`backend/api_types.py`**

Add fields to `GenerateImageRequest`:

```python
class GenerateImageRequest(BaseModel):
    prompt: NonEmptyPrompt
    width: int = 1024
    height: int = 1024
    numSteps: int = 4
    numImages: int = 1
    loraPath: str | None = None
    loraWeight: float = 1.0
    # New for img2img:
    sourceImagePath: str | None = None
    strength: float = 0.65
```

### 1d. Update image generation handler

**`backend/handlers/image_generation_handler.py`**

In `generate_image()`, branch on `sourceImagePath`:

```python
from PIL import Image

def generate_image(self, ..., source_image_path: str | None = None, strength: float = 0.65) -> list[str]:
    # ... existing setup (load pipeline, LoRA, etc.) ...

    source_image: PILImageType | None = None
    if source_image_path:
        source_image = Image.open(source_image_path).convert("RGB")
        # Snap source dimensions to 16-multiples if using source dimensions
        width = (source_image.width // 16) * 16
        height = (source_image.height // 16) * 16
        source_image = source_image.resize((width, height), Image.LANCZOS)

    for i in range(num_images):
        if source_image is not None:
            result = zit.img2img(
                prompt=prompt,
                image=source_image,
                strength=strength,
                height=height,
                width=width,
                guidance_scale=0.0,
                num_inference_steps=num_inference_steps,
                seed=seed + i,
            )
        else:
            result = zit.generate(
                prompt=prompt,
                height=height,
                width=width,
                guidance_scale=0.0,
                num_inference_steps=num_inference_steps,
                seed=seed + i,
            )
        # ... save output (use prefix "zit_edit_" for edits) ...
```

Update the `generate()` entry point to pass `source_image_path` and `strength` from `req`.

### 1e. Update job executor

**`backend/handlers/job_executors.py`**

In `_execute_image()`, pass `sourceImagePath` and `strength` from `job.params` into `GenerateImageRequest`.

### 1f. Update gallery handler

**`backend/handlers/gallery_handler.py`**

Add `"zit_edit_"` prefix mapping:

```python
_MODEL_PREFIXES: list[tuple[str, str]] = [
    ("zit_edit_", "zit-edit"),    # New
    ("zit_image_", "zit"),
    # ... existing ...
]
```

---

## Phase 2: Frontend — Edit Flow

### 2a. Add `editImage` to the generation hook

**`frontend/hooks/use-generation.ts`**

Add new method:

```typescript
const editImage = useCallback(async (
    prompt: string,
    sourceImagePath: string,
    settings: GenerationSettings,
    strength: number = 0.65,
) => {
    setState(prev => ({
        ...prev,
        isGenerating: true,
        progress: 0,
        statusMessage: 'Editing image...',
        videoUrl: null, videoPath: null, imageUrl: null, imageUrls: [],
        error: null,
    }))

    try {
        const backendUrl = await window.electronAPI.getBackendUrl()
        const response = await fetch(`${backendUrl}/api/queue/submit`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                type: 'image',
                model: appSettings.imageModel || 'z-image-turbo',
                params: {
                    prompt,
                    sourceImagePath,
                    strength,
                    width: 0,   // 0 = use source dimensions
                    height: 0,
                    numSteps: settings.imageSteps || 8,
                    numImages: 1,
                    ...(settings.loraPath ? { loraPath: settings.loraPath, loraWeight: settings.loraWeight ?? 1.0 } : {}),
                },
            }),
        })
        // ... same submit + poll pattern as generateImage ...
    } catch (error) { /* ... */ }
}, [appSettings.imageModel, startPolling])
```

Add `editImage` to the returned interface.

### 2b. Integrate into GenSpace

**`frontend/views/GenSpace.tsx`**

Add state for the edit source image:

```typescript
const [editSourceImage, setEditSourceImage] = useState<{ url: string; path: string } | null>(null)
```

When `editSourceImage` is set and mode is `image`:
- Show the source image preview above the prompt
- Show a strength slider (0.0–1.0, default 0.65)
- Change prompt placeholder to "Describe your edit..."
- Change the generate button label to "Edit"
- Hide aspect ratio / resolution controls (use source dimensions)
- On submit, call `editImage()` instead of `generateImage()`

Add a clear button (X) to remove the source image and return to text-to-image.

### 2c. Wire the Edit button

**`frontend/components/ImageResult.tsx`**

The Edit button already exists (line ~130) but has no handler. Add an `onEdit` prop:

```typescript
interface ImageResultProps {
    // ... existing ...
    onEdit?: (imagePath: string) => void
}
```

Wire `onClick` to call `onEdit` with the image file path. In GenSpace, pass a callback that sets `editSourceImage`.

### 2d. Add strength slider to SettingsPanel

**`frontend/components/SettingsPanel.tsx`**

When in image mode and a source image is present, show:

```typescript
{/* Strength slider — only visible during image editing */}
<div className="space-y-2">
    <label className="text-xs font-medium text-zinc-400">Edit Strength</label>
    <input
        type="range"
        min={0} max={100} value={strength * 100}
        onChange={(e) => setStrength(Number(e.target.value) / 100)}
    />
    <div className="flex justify-between text-[10px] text-zinc-500">
        <span>Subtle</span>
        <span>{Math.round(strength * 100)}%</span>
        <span>Heavy</span>
    </div>
</div>
```

Add `strength` to `GenerationSettings` interface.

### 2e. Add phase message for editing

**`frontend/hooks/use-generation.ts`**

```typescript
case 'encoding_image':
    return 'Encoding source image...'
```

---

## Phase 3: Polish

### 3a. Before/after comparison

**`frontend/components/ImageResult.tsx`**

When the result was generated from a source image, show a "Compare" toggle:
- On click/hold, swap the displayed image to the source
- Release returns to the edited result
- Simple state toggle, no complex slider needed for MVP

### 3b. Edit from video frames

**`frontend/contexts/ProjectContext.tsx`**

The `genSpaceEditImageUrl` mechanism already exists for sending frames from VideoEditor to GenSpace. Update the receiver in GenSpace to support routing to image-edit mode:

```typescript
// When receiving an edit request from VideoEditor
if (genSpaceEditImageUrl) {
    setEditSourceImage({ url: genSpaceEditImageUrl, path: fileUrlToPath(genSpaceEditImageUrl) })
    setMode('image')
}
```

### 3c. Edit from gallery

Add an "Edit" action to gallery items (both inline gallery in GenSpace and the full Gallery view). Clicking it navigates to GenSpace with the image pre-loaded as edit source.

### 3d. Sequential edits

After an edit completes, the result becomes available as a new source. The "Edit" button on the result should re-populate the source image with the current output, enabling iterative refinement.

---

## What NOT to build (yet)

- **Inpainting / masking** — requires canvas drawing tools, mask pipeline, significant UI work. Future phase.
- **ControlNet** — `ZImageControlNetPipeline` exists in diffusers but adds model complexity. Future phase.
- **API-backed img2img** — Replicate supports img2img but adds upload/download overhead. Local-only for now.
- **Batch editing** — editing multiple images with the same prompt. Low demand, adds complexity.

---

## Acceptance Criteria

- [x] User can click "Edit" on a generated image and it becomes the source for img2img
- [ ] User can drop/browse an external image file as edit source
- [x] Strength slider appears when source image is set (default 0.65)
- [x] Prompt describes the desired edit; generation uses img2img pipeline
- [x] Output image matches source dimensions (snapped to 16-multiples)
- [x] LoRA works with img2img (same load/unload behavior)
- [x] Edited images appear in gallery with "zit-edit" model tag
- [x] Progress phases report correctly (Preparing GPU → Loading model → Encoding image → Generating → Complete)
- [x] No new model download required — reuses existing ZIT weights
- [x] Backend tests pass with fake img2img method
- [x] TypeScript and Pyright typechecks pass

## Quality Gate

- [x] `pnpm typecheck` passes
- [x] `pnpm backend:test` passes (343+ tests)
- [ ] Manual test: generate image → edit it → verify output is visually modified
- [ ] Manual test: import external image → edit → verify dimensions preserved
- [ ] Manual test: edit with LoRA active → verify style applied

---

## Files to Modify

### Backend (7 files)
| File | Change |
|------|--------|
| `services/image_generation_pipeline/image_generation_pipeline.py` | Add `img2img()` to protocol |
| `services/image_generation_pipeline/zit_image_generation_pipeline.py` | Implement `img2img()` with `ZImageImg2ImgPipeline` sharing model components |
| `api_types.py` | Add `sourceImagePath`, `strength` to `GenerateImageRequest` |
| `handlers/image_generation_handler.py` | Branch on `sourceImagePath`, load/resize source image, call `img2img` |
| `handlers/job_executors.py` | Pass `sourceImagePath`, `strength` from job params |
| `handlers/gallery_handler.py` | Add `"zit_edit_"` prefix mapping |
| `tests/fakes/services.py` | Add `img2img()` to `FakeImageGenerationPipeline` |

### Frontend (5 files)
| File | Change |
|------|--------|
| `hooks/use-generation.ts` | Add `editImage()` method, `encoding_image` phase message |
| `views/GenSpace.tsx` | Add `editSourceImage` state, edit mode UI, strength slider, wire Edit button |
| `components/ImageResult.tsx` | Wire `onEdit` prop to Edit button |
| `components/SettingsPanel.tsx` | Add `strength` to `GenerationSettings`, show slider in edit mode |
| `contexts/ProjectContext.tsx` | Support image-edit routing from VideoEditor frames |

### No changes needed
- Queue system (already supports arbitrary params)
- Queue worker (delegates to executor)
- Pipeline lifecycle / GPU management (img2img reuses same model)
- Device management / CPU offload (shared components)
- LoRA loading (same mixin base class)

---

## Dependency Check

Verify diffusers version in `backend/pyproject.toml`:

```bash
grep diffusers backend/pyproject.toml
```

If < 0.37.0, update to `diffusers >= 0.37.0` and re-lock with `uv lock`.

---

## References

- HuggingFace Z-Image docs: https://huggingface.co/docs/diffusers/en/api/pipelines/z_image
- Model card: https://huggingface.co/Tongyi-MAI/Z-Image-Turbo
- Existing ZIT pipeline: `backend/services/image_generation_pipeline/zit_image_generation_pipeline.py`
- ImageResult Edit button (placeholder): `frontend/components/ImageResult.tsx:130-134`
- GenSpace edit routing: `frontend/views/GenSpace.tsx:986-993`
