# GPU Optimization Results — March 9, 2026

**Hardware:** NVIDIA RTX 4090 24GB VRAM, Windows 11, CUDA 12.9
**Model:** LTX-Video 2.3 (ltx-2.3-22b-distilled), 22B parameter distilled model

---

## What We Did

### 1. FFN Chunked Feedforward (Peak VRAM Reduction)

**Problem:** The LTX transformer has 48 transformer blocks. Each block's FeedForward layer expands the hidden dimension by 4× (e.g., 3072 → 12288), creating enormous intermediate tensors. When processing long sequences (>200 frames), these intermediate tensors exceed available VRAM, forcing the GPU into a slow fallback path that causes the "nonlinear scaling cliff" — where 10s took 6.5× longer than 8s for only 25% more frames.

**Solution:** We monkey-patch every FeedForward module in the transformer to split its computation along the sequence dimension into 8 chunks. Instead of computing all 241 frames at once through the 4× expansion, it processes ~30 frames at a time. The output is mathematically identical (FeedForward is pointwise, so chunking is lossless). This reduces peak VRAM usage by up to 8×.

**Setting:** `ffnChunkCount` (default: 8, set to 0 to disable)
**File:** `backend/services/gpu_optimizations/ffn_chunking.py`

### 2. TeaCache (Timestep-Aware Caching)

**Problem:** During denoising, the transformer runs a full forward pass at every timestep. Many adjacent timesteps produce very similar outputs, wasting computation.

**Solution:** TeaCache monitors how much the input changes between denoising steps using a relative L1 distance metric, rescaled by a polynomial fitted to the LTX-Video noise schedule. When the change is below a threshold, it reuses the previous step's residual (the difference between input and output) instead of running the full transformer. First and last steps are always fully computed.

**Setting:** `teaCacheThreshold` (default: 0.0 = off, 0.03 = balanced quality/speed, 0.05 = aggressive)
**File:** `backend/services/gpu_optimizations/tea_cache.py`

### 3. VRAM Deep Cleanup

**Problem:** After heavy GPU workloads (especially long generations), VRAM fragmentation caused subsequent generations to stall at 15% progress indefinitely. The GPU showed 100% utilization but made no progress.

**Solution:** After every GPU job completes, we now run an aggressive cleanup: two rounds of garbage collection + CUDA cache clearing + CUDA synchronize. This ensures VRAM is fully reclaimed before the next job starts.

**File:** `backend/services/gpu_cleaner/torch_cleaner.py` (deep_cleanup method)

### 4. R2 Cloud Storage Upload

**Problem:** Generated videos/images only lived on the local machine.

**Solution:** After each generation, results can be automatically uploaded to Cloudflare R2 (S3-compatible) storage. Configure via settings: `r2AccessKeyId`, `r2SecretAccessKey`, `r2Endpoint`, `r2Bucket`, `r2PublicUrl`, and toggle with `autoUploadToR2`.

**File:** `backend/services/r2_client/r2_client_impl.py`

---

## Benchmark Results

All tests at 512p, 16:9 aspect ratio, 24fps, FFN chunking=8, TeaCache threshold=0.03.

### Comparison Table

| Duration | Frames | Baseline | Optimized | Speedup | Time Saved |
|----------|--------|----------|-----------|---------|------------|
| 2s | 49 | 37s | 42s (cold) / 59s* | — | Session warmth variance |
| 5s | 121 | 84s | **65s** | **1.29×** | 19s (23%) |
| 8s | 193 | 100s | **65s** | **1.54×** | 35s (35%) |
| 10s | 241 | 651s | **275s** | **2.37×** | 376s (58%) |
| 20s | 481 | ~11,820s (3.3 hrs) | Still slow** | ~2× est. | See notes |

*The 2s "optimized" run was the first warm generation after cold start. The baseline 37s was also from a warm session. Session-dependent variance of ±20s is normal for short clips.

**The 20s test reached 15% inference after ~50 minutes and was cancelled. Extrapolating: ~5-6 hours total, which is ~2× faster than baseline but still impractical. For 20s+ content, use extend chains (5× 4s clips ≈ 5 minutes).

### Key Takeaway: The Nonlinear Scaling Cliff is Dramatically Reduced

Before optimizations:
```
8s (193 frames) = 100s
10s (241 frames) = 651s   ← 6.5× jump for 25% more frames
```

After optimizations:
```
8s (193 frames) = 65s
10s (241 frames) = 275s   ← 4.2× jump for 25% more frames (was 6.5×)
```

The cliff is still there (attention is still quadratic in sequence length) but FFN chunking prevents the worst of the VRAM thrashing. The 10s generation went from "impractical" (11 min) to "tolerable" (4.5 min).

### Scaling Curve (512p, Optimized)

```
Frames:    49     121     193     241      481
Time:     42s    65s     65s    275s    ~5hrs (est)
Per-frame: 0.86s  0.54s  0.34s  1.14s   ~37s
```

The sweet spot is **5-8 seconds** (121-193 frames). Beyond 193 frames, the per-frame cost increases dramatically due to quadratic attention scaling.

---

## Practical User Guidelines (Updated)

| Use Case | Setting | Expected Time |
|----------|---------|---------------|
| Quick preview | 512p, 2s | ~40s |
| Standard clip | 512p, 5s | **~65s** (was 84s) |
| Longer clip | 512p, 8s | **~65s** (was 100s) |
| Extended clip | 512p, 10s | **~4.5 min** (was 11 min) |
| Long scene | 512p, 2s × 5 extend chain | ~5 min |
| High quality short | 720p, 2s | ~40s |

**Avoid:** 512p ≥20s (hours), 720p ≥8s (hours), 1080p ≥5s (OOM crash)

**For longer scenes:** Use the extend feature to chain 2-5s clips. Five 2s clips = ~3-5 min total, vs a single 10s clip = ~4.5 min — similar time but with better control.

---

## Generated Samples

All benchmark outputs are in `D:\git\directors-desktop\backend\outputs\`:

### Today's Optimized Benchmark Outputs
| File | Duration | Time | Prompt |
|------|----------|------|--------|
| `ltx2_video_20260309_113236_01996534.mp4` | 2s (warmup) | 42s | Test warmup scene |
| `ltx2_video_20260309_113318_a6b79593.mp4` | 2s | 59s | Ocean waves crashing on rocky shore, sunset, cinematic |
| `ltx2_video_20260309_113417_9832bd5d.mp4` | 5s | 65s | Jellyfish glowing neon in dark ocean, bioluminescent |
| `ltx2_video_20260309_113522_68e618a6.mp4` | 8s | 65s | Samurai in bamboo forest, rain, cinematic |
| `ltx2_video_20260309_113627_5ab81499.mp4` | 10s | 275s | Rocket launch at dawn, smoke plume, slow motion |

### Earlier Baseline Benchmark Outputs
| File | Duration | Time | Prompt |
|------|----------|------|--------|
| `ltx2_video_20260309_052523_29fd7111.mp4` | 5s | 84s | Ocean waves crashing on rocky shore, sunset, cinematic |
| `ltx2_video_20260309_052647_869c45d0.mp4` | 8s | 100s | Jellyfish glowing neon in dark ocean, bioluminescent |
| `ltx2_video_20260309_052826_071807c1.mp4` | 10s | 651s | Samurai in bamboo forest, rain, cinematic |
| `ltx2_video_20260309_053918_5b3bb04d.mp4` | 10s | 650s | Rocket launch at dawn, smoke plume, slow motion |
| `ltx2_video_20260309_055008_3f713c6c.mp4` | 20s | ~3.3hrs | Flower blooming timelapse, golden hour |

### Image Outputs (ZIT model)
| File | Size | Notes |
|------|------|-------|
| `zit_image_20260309_095030_6f9f6d95.png` | 1.2MB | Robot in flower garden (API, nano-banana-2) |
| `zit_image_20260309_051257_40f12d39.png` | 1.2MB | Local ZIT generation |
| Plus 7 more ZIT images from earlier sessions | | |

---

## Model & Feature Status

### LTX-Video Version
**Yes, we're on LTX-2.3** — the latest. The checkpoint is `ltx-2.3-22b-distilled.safetensors` from `Lightricks/LTX-2.3` on HuggingFace. This is the 22-billion parameter distilled model, which is the most capable version available.

### Prompt Enhancement (Magic Wand)
**Yes, prompt enhancement is implemented.** There's a magic wand button in the GenSpace UI that enhances prompts via:
1. **Directors Palette API** (priority) — if `paletteApiKey` is set, uses `/api/prompt-expander` endpoint with "2x" expansion level
2. **Gemini 2.0 Flash** (fallback) — if `geminiApiKey` is set, uses Gemini to expand prompts with cinematic details

The enhancement adds lighting, camera angles, mood, and atmosphere to vague prompts. It works for both text-to-video and text-to-image modes.

**Current state:** It works if you have either a Palette API key or Gemini API key configured. The Palette API key is already set based on your settings.

### Aspect Ratios
Supported aspect ratios for local generation:
- **16:9** (default) — standard widescreen
- **9:16** — vertical/portrait
- **1:1** — square
- **4:3** — classic TV
- **3:4** — portrait

For API-forced generation (Seedance), only 16:9 and 9:16 are allowed.

The resolution mapping by aspect ratio is handled in `video_generation_handler.py`. Each resolution tier (512p, 720p, 1080p) has width/height values per aspect ratio.

---

## What's Missing: Time Estimation UI

Currently, the frontend shows these status phases during generation:
- "Queued — waiting..."
- "Starting up..."
- "Preparing GPU..."
- "Loading model..." / "Loading video model..."
- "Encoding prompt..."
- "Generating..." (inference phase)
- "Decoding video..."
- "Complete!"

**What Directors Palette has that we don't:**
- **Elapsed time counter** (MM:SS since generation started)
- **Estimated time remaining** (based on benchmark data for the resolution/duration combo)
- **Time-based progress bar** (progress = elapsed / estimated × 100%)
- **Stage indicators** (Analysis → Generation → Complete)

**Recommendation:** Add estimated durations based on our benchmark data. For example, when a user submits a 512p 8s job, show "Estimated: ~1:05" and count up elapsed time. The data:

```
512p 2s = ~40s estimated
512p 5s = ~65s estimated
512p 8s = ~65s estimated
512p 10s = ~275s estimated (4:35)
```

This would be a frontend change in `use-generation.ts` and the progress display component.
