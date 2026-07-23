import { addVisualAssetToProject } from './asset-copy'
import { toModelsDirRelativeRef } from './lora-library'
import { logger } from './logger'
import type { RecoveryGenType, RecoveryImporter } from './generation-recovery'

// Mirrors GenSpace's own live-generation completion effect (see the `videoPath` effect in
// GenSpace.tsx) but operates on the recovery marker's captured context instead of live component
// state, so it can run in the background while that project's GenSpace isn't mounted. ic-lora/
// retake/extend all write a marker with no `settings` (see GenerationRecoveryContext) and recover
// as a standalone video asset here too, same as GenSpace's own mount-recovery effect does.
const importVideo: RecoveryImporter = async (ctx, result, { addAsset, modelsDir }) => {
  const videoPath = typeof result === 'string' ? result : result[0]
  if (!videoPath) return

  const copied = await addVisualAssetToProject(videoPath, ctx.projectId, 'video')
  if (!copied) throw new Error('Could not persist generated video to project storage')

  const s = ctx.settings
  const genMode = ctx.inputAudioUrl
    ? 'audio-to-video'
    : ctx.inputImageUrl ? 'image-to-video' : 'text-to-video'

  addAsset(ctx.projectId, {
    type: 'video',
    path: copied.path,
    bigThumbnailPath: copied.bigThumbnailPath,
    smallThumbnailPath: copied.smallThumbnailPath,
    width: copied.width,
    height: copied.height,
    prompt: ctx.prompt,
    resolution: s?.videoResolution ?? '',
    duration: s?.duration,
    generationParams: {
      mode: genMode,
      prompt: ctx.prompt,
      model: s?.model ?? 'fast',
      duration: s?.duration ?? 0,
      resolution: s?.videoResolution ?? '',
      fps: s?.fps ?? 24,
      audio: s?.audio ?? false,
      cameraMotion: 'none',
      imageAspectRatio: s?.aspectRatio,
      imageSteps: 4,
      inputImageUrl: ctx.inputImageUrl,
      inputAudioUrl: ctx.inputAudioUrl,
      loras: s?.loras && s.loras.length > 0
        ? s.loras.map(l => ({ ref: toModelsDirRelativeRef(l.ref, modelsDir), name: l.name, scale: l.scale }))
        : undefined,
    },
    takes: [{
      path: copied.path,
      bigThumbnailPath: copied.bigThumbnailPath,
      smallThumbnailPath: copied.smallThumbnailPath,
      width: copied.width,
      height: copied.height,
      createdAt: Date.now(),
    }],
    activeTakeIndex: 0,
  })
}

// Mirrors GenSpace's `imagePaths` completion effect. A failed copy for one image in a multi-image
// batch is logged and skipped rather than thrown, same as the live effect — one bad file must not
// drop the rest of the batch. But if *every* copy in the batch fails, this must throw rather than
// return normally: checkAndConsumeRecovery treats a clean return as "imported, delete the
// marker" — silently succeeding on a total failure would drop the result for good.
const importImage: RecoveryImporter = async (ctx, result, { addAsset }) => {
  const paths = Array.isArray(result) ? result : [result]
  const s = ctx.settings
  const genMode = ctx.inputImageUrl ? 'image-edit' : 'text-to-image'
  let importedAny = false

  for (const imgPath of paths) {
    const copied = await addVisualAssetToProject(imgPath, ctx.projectId, 'image')
    if (!copied) {
      logger.error(`Could not persist generated image to project storage: ${imgPath}`)
      continue
    }
    importedAny = true
    addAsset(ctx.projectId, {
      type: 'image',
      path: copied.path,
      bigThumbnailPath: copied.bigThumbnailPath,
      smallThumbnailPath: copied.smallThumbnailPath,
      width: copied.width,
      height: copied.height,
      prompt: ctx.prompt,
      resolution: s?.imageResolution ?? '',
      generationParams: {
        mode: genMode,
        prompt: ctx.prompt,
        model: 'fast',
        duration: 5,
        resolution: s?.imageResolution ?? '',
        fps: 24,
        audio: false,
        cameraMotion: 'none',
        imageAspectRatio: s?.aspectRatio,
        imageSteps: s?.imageSteps ?? 4,
        ...(ctx.inputImageUrl ? { inputImageUrl: ctx.inputImageUrl, imageEditStrength: s?.imageEditStrength } : {}),
      },
      takes: [{
        path: copied.path,
        bigThumbnailPath: copied.bigThumbnailPath,
        smallThumbnailPath: copied.smallThumbnailPath,
        width: copied.width,
        height: copied.height,
        createdAt: Date.now(),
      }],
      activeTakeIndex: 0,
    })
  }

  if (!importedAny && paths.length > 0) {
    throw new Error('Could not persist any generated image to project storage')
  }
}

export const builtinRecoveryImporters: Partial<Record<RecoveryGenType, RecoveryImporter>> = {
  video: importVideo,
  image: importImage,
}
