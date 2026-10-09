import type { GenSpaceMode } from './genspace-multi-keyframe'
import { toKeyframeInputs, type KeyframeItem } from './multi-keyframe.ts'
import type { PromptProvenance } from './prompt-provenance.ts'

/** The GenerationSettings fields a video request is built from, structurally. */
export interface GenerateVideoBodySettings {
  model: string
  duration: number | null
  videoResolution: string
  fps: number
  audio: boolean
  cameraMotion: string
  negativePrompt?: string
  aspectRatio?: string
  loras?: Array<{ ref: string; scale: number; catalogId?: string }>
}

/**
 * The whole /api/generate body for a video request.
 *
 * `promptProvenance` is the part the backend cannot infer: it enhances every typed prompt, so a
 * prompt the user already enhanced by hand has to say so or it gets rewritten a second time.
 * A selected LoRA's `catalogId` is the other one: without it the backend cannot tell a library
 * LoRA from a custom file, and enhancement would paraphrase away a trigger phrase the adapter
 * needs. It carries only display-irrelevant identity, so a custom LoRA simply omits it.
 */
export function buildGenerateVideoBody({
  prompt,
  promptProvenance,
  settings,
  imagePath,
  lastImagePath,
  audioPath,
  imageInputs,
}: {
  prompt: string
  promptProvenance: PromptProvenance
  settings: GenerateVideoBodySettings
  imagePath: string | null | undefined
  lastImagePath: string | null | undefined
  audioPath?: string | null
  imageInputs?: { mode: GenSpaceMode; keyframes: KeyframeItem[] }
}): Record<string, unknown> {
  return {
    prompt,
    promptProvenance,
    model: settings.model,
    duration: settings.duration,
    resolution: settings.videoResolution,
    fps: settings.fps,
    audio: settings.audio,
    cameraMotion: settings.cameraMotion,
    negativePrompt: settings.negativePrompt ?? '',
    aspectRatio: settings.aspectRatio || '16:9',
    ...buildGenerateVideoImageInputs({
      mode: imageInputs?.mode ?? 'video',
      imagePath,
      lastImagePath,
      keyframes: imageInputs?.keyframes ?? [],
    }),
    ...(audioPath ? { audioPath } : {}),
    ...(settings.loras?.length
      ? {
          loras: settings.loras.map(l => ({
            ref: l.ref,
            scale: l.scale,
            ...(l.catalogId ? { catalogId: l.catalogId } : {}),
          })),
        }
      : {}),
  }
}

interface GenerateVideoImageInputs {
  imagePath?: string
  lastImagePath?: string
  keyframes?: Array<{
    imagePath: string
    frameIndex: number
    strength: number
  }>
}

export function buildGenerateVideoImageInputs({
  mode,
  imagePath,
  lastImagePath,
  keyframes,
}: {
  mode: GenSpaceMode
  imagePath: string | null | undefined
  lastImagePath: string | null | undefined
  keyframes: KeyframeItem[]
}): GenerateVideoImageInputs {
  if (mode === 'multi-keyframe' && keyframes.length > 0) {
    return { keyframes: toKeyframeInputs(keyframes) }
  }

  return {
    ...(imagePath ? { imagePath } : {}),
    ...(lastImagePath ? { lastImagePath } : {}),
  }
}
