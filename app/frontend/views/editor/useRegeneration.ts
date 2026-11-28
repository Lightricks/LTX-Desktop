import { useCallback, useEffect, useRef } from 'react'
import type { Asset, TimelineClip } from '../../types/project-model'
import type { GenerationSettings } from '../../components/SettingsPanel'
import type { GenerationError } from '../../lib/generation-errors'
import type { VideoGenerationPipeline } from '../../lib/video-generation-model-specs'
import type { GenSpaceMode } from '../../lib/genspace-multi-keyframe'
import { fromPersistedKeyframes, type KeyframeItem } from '../../lib/multi-keyframe'
import { addVisualAssetToProject } from '../../lib/asset-copy'
import { ApiClient } from '../../lib/api-client'
import { logger } from '../../lib/logger'
import {
  selectAssets,
  selectClips,
  selectRegenerationPreError,
  selectRegeneratingAssetId,
  selectRegeneratingClipId,
} from './editor-selectors'
import { useEditorActions, useEditorStore } from './editor-store'

const LEGACY_VIDEO_RESOLUTION_MAP: Record<string, string> = {
  '768x512': '540p',
  '1024x576': '540p',
  '1280x720': '720p',
  '1920x1080': '1080p',
}

const LEGACY_IMAGE_RESOLUTION_MAP: Record<string, string> = {
  '768x512': '1080p',
  '1024x576': '1080p',
  '1280x720': '1080p',
  '1920x1080': '1080p',
  '2160p': '2048p',
}

function normalizeVideoResolution(value: string | undefined): string {
  if (!value) return '540p'
  if (value === '540p' || value === '720p' || value === '1080p' || value === '1440p' || value === '2160p') {
    return value
  }
  return LEGACY_VIDEO_RESOLUTION_MAP[value] || '540p'
}

function normalizeImageResolution(value: string | undefined): string {
  if (!value) return '1080p'
  if (value === '1080p' || value === '1440p' || value === '2048p') {
    return value
  }
  return LEGACY_IMAGE_RESOLUTION_MAP[value] || '1080p'
}

function resolveLiveAssetForClip(assets: Asset[], clip: TimelineClip): Asset {
  if (!clip.assetId) return clip.asset!
  return assets.find(asset => asset.id === clip.assetId) || clip.asset!
}

function resolveClipPath(assets: Asset[], clip: TimelineClip): string {
  const liveAsset = resolveLiveAssetForClip(assets, clip)
  if (liveAsset.takes && liveAsset.takes.length > 0 && clip.takeIndex !== undefined) {
    const idx = Math.max(0, Math.min(clip.takeIndex, liveAsset.takes.length - 1))
    return liveAsset.takes[idx].path || ''
  }
  return liveAsset.path || ''
}

function resolveAssetPreviewPath(assets: Asset[], clips: TimelineClip[], asset: Asset, clipId?: string): string {
  const clip = clipId ? clips.find(candidate => candidate.id === clipId) : undefined
  if (clip) {
    const clipPath = resolveClipPath(assets, clip)
    if (clipPath) return clipPath
  }

  if (asset.takes && asset.takes.length > 0) {
    const activeIndex = Math.max(0, Math.min(asset.activeTakeIndex ?? (asset.takes.length - 1), asset.takes.length - 1))
    return asset.takes[activeIndex]?.path || asset.path
  }

  return asset.path
}

export interface UseRegenerationParams {
  projectId: string
  // Generation hook values
  regenGenerate: (
    prompt: string,
    imagePath: string | null,
    settings: GenerationSettings,
    audioPath?: string | null,
    lastImagePath?: string | null,
    imageInputs?: { mode: GenSpaceMode; keyframes: KeyframeItem[] },
  ) => Promise<void>
  regenGenerateImage: (prompt: string, settings: GenerationSettings, editSource?: string | null) => Promise<void>
  regenVideoPath: string | null
  regenImagePath: string | null
  isRegenerating: boolean
  regenCancel: () => void
  regenReset: () => void
  regenError: GenerationError | null
  canCancelInFlight: boolean
  shouldVideoGenerateWithLtxApi: boolean
}

export function useRegeneration(params: UseRegenerationParams) {
  const {
    projectId,
    regenGenerate,
    regenGenerateImage,
    regenVideoPath,
    regenImagePath,
    isRegenerating,
    regenCancel,
    regenReset,
    regenError,
    canCancelInFlight,
    shouldVideoGenerateWithLtxApi,
  } = params
  const {
    applyGeneratedTake,
    cancelClipRegeneration,
    failClipRegeneration,
    setRegenerationPreError,
    startClipRegeneration,
    updateAsset,
  } = useEditorActions()
  const assets = useEditorStore(selectAssets)
  const clips = useEditorStore(selectClips)
  const regeneratingAssetId = useEditorStore(selectRegeneratingAssetId)
  const regeneratingClipId = useEditorStore(selectRegeneratingClipId)
  const regenerationPreError = useEditorStore(selectRegenerationPreError)
  const cancelRequestedRef = useRef(false)

  const dismissRegenerationPreError = useCallback(() => {
    setRegenerationPreError(null)
  }, [setRegenerationPreError])

  const handleRegenerate = useCallback(async (assetId: string, clipId?: string) => {
    if (!projectId || isRegenerating) return
    const asset = assets.find(candidate => candidate.id === assetId)
    if (!asset) return

    startClipRegeneration(assetId, clipId)

    let generationParams = asset.generationParams
    if (!generationParams) {
      try {
        const clipPath = resolveAssetPreviewPath(assets, clips, asset, clipId)
        let framePath = ''

        if (asset.type === 'video' && clipPath) {
          const frame = await window.electronAPI.extractVideoFrame({
            videoPath: clipPath,
            seekTime: 0.1,
            width: 512,
            quality: 3,
          })
          framePath = frame.path
        } else if (asset.type === 'image' && clipPath) {
          framePath = clipPath
        }

        if (framePath) {
          const result = await ApiClient.suggestGapPrompt({
            gapDuration: asset.duration || 5,
            mode: asset.type === 'image' ? 'text-to-image' : 'text-to-video',
            beforePrompt: '',
            afterPrompt: '',
            beforeFrame: framePath,
            afterFrame: '',
          })
          if (!result.ok) {
            throw new Error(result.error.message)
          }

          const promptSuggestion = result.data
          if (promptSuggestion.suggested_prompt) {
            generationParams = {
              mode: asset.type === 'image' ? 'text-to-image' : 'text-to-video',
              prompt: promptSuggestion.suggested_prompt,
              model: 'fast',
              duration: asset.duration || 5,
              resolution: asset.type === 'image'
                ? normalizeImageResolution(asset.resolution)
                : normalizeVideoResolution(asset.resolution),
              fps: 24,
              audio: false,
              cameraMotion: 'none',
            }
            updateAsset(asset.id, { generationParams })
          }
        }
      } catch (error) {
        logger.warn(`Failed to auto-generate prompt for imported asset: ${error}`)
      }

      if (!generationParams) {
        failClipRegeneration(
          'Could not auto-generate a prompt for this clip. Try using "Send to GenSpace" instead.',
        )
        return
      }
    }

    if (generationParams.mode === 'retake') {
      failClipRegeneration(
        'Retake assets cannot be regenerated yet. Try using Retake from the clip menu instead.',
      )
      return
    }

    if (generationParams.mode === 'ic-lora') {
      failClipRegeneration(
        'IC-LoRA assets cannot be regenerated yet. Try using IC-LoRA from the clip menu instead.',
      )
      return
    }

    if (generationParams.mode === 'text-to-image') {
      void regenGenerateImage(generationParams.prompt, {
        model: generationParams.model as VideoGenerationPipeline,
        duration: generationParams.duration,
        videoResolution: '540p',
        fps: generationParams.fps,
        audio: generationParams.audio,
        cameraMotion: generationParams.cameraMotion,
        imageResolution: normalizeImageResolution(generationParams.resolution),
        imageAspectRatio: generationParams.imageAspectRatio || '16:9',
        imageSteps: generationParams.imageSteps || 4,
        variations: 1,
      })
      return
    }

    if (generationParams.mode === 'image-edit') {
      if (!generationParams.inputImageUrl) {
        failClipRegeneration(
          'This edited asset is missing its source image and cannot be regenerated.',
        )
        return
      }
      void regenGenerateImage(generationParams.prompt, {
        model: generationParams.model as VideoGenerationPipeline,
        duration: generationParams.duration,
        videoResolution: '540p',
        fps: generationParams.fps,
        audio: generationParams.audio,
        cameraMotion: generationParams.cameraMotion,
        imageResolution: normalizeImageResolution(generationParams.resolution),
        imageAspectRatio: generationParams.imageAspectRatio || '16:9',
        imageSteps: generationParams.imageSteps || 8,
        imageEditStrength: generationParams.imageEditStrength,
        variations: 1,
      }, generationParams.inputImageUrl)
      return
    }

    const imagePath = generationParams.inputImageUrl
      && (generationParams.mode === 'image-to-video' || generationParams.mode === 'audio-to-video')
      ? generationParams.inputImageUrl
      : null
    const rawVideoSettings: GenerationSettings = {
      model: generationParams.model as VideoGenerationPipeline,
      duration: generationParams.duration,
      videoResolution: normalizeVideoResolution(generationParams.resolution),
      fps: generationParams.fps,
      audio: generationParams.audio,
      cameraMotion: generationParams.cameraMotion,
      imageResolution: '1080p',
      imageAspectRatio: generationParams.imageAspectRatio || '16:9',
      imageSteps: generationParams.imageSteps || 4,
      // Local LoRA refs are filesystem paths the cloud API can't resolve.
      loras: !shouldVideoGenerateWithLtxApi ? generationParams.loras : undefined,
    }
    void regenGenerate(
      generationParams.prompt,
      imagePath,
      rawVideoSettings,
      generationParams.mode === 'audio-to-video' ? generationParams.inputAudioUrl : undefined,
      imagePath && generationParams.duration != null ? generationParams.inputLastImageUrl : undefined,
      generationParams.mode === 'multi-keyframe'
        ? {
            mode: 'multi-keyframe',
            keyframes: fromPersistedKeyframes(generationParams.keyframes ?? []),
          }
        : undefined,
    )
  }, [
    isRegenerating,
    projectId,
    regenGenerate,
    regenGenerateImage,
    startClipRegeneration,
    failClipRegeneration,
    assets,
    clips,
    updateAsset,
    shouldVideoGenerateWithLtxApi,
  ])

  const handleCancelRegeneration = useCallback(() => {
    if (!canCancelInFlight) return
    cancelRequestedRef.current = true
    regenCancel()
  }, [canCancelInFlight, regenCancel])

  const persistGeneratedTake = useCallback(async (
    generatedPath: string,
    type: 'video' | 'image',
    assetId: string,
    clipId: string | null,
  ) => {
    if (!projectId) return

    const copied = await addVisualAssetToProject(generatedPath, projectId, type)
    if (!copied) {
      logger.error(`Failed to persist regenerated ${type}: ${generatedPath}`)
      cancelClipRegeneration()
      regenReset()
      return
    }

    applyGeneratedTake(assetId, {
      path: copied.path,
      bigThumbnailPath: copied.bigThumbnailPath,
      smallThumbnailPath: copied.smallThumbnailPath,
      width: copied.width,
      height: copied.height,
      createdAt: Date.now(),
    }, clipId ?? undefined)
    regenReset()
  }, [applyGeneratedTake, cancelClipRegeneration, projectId, regenReset])

  useEffect(() => {
    if (cancelRequestedRef.current) return
    if (!regenVideoPath || !regeneratingAssetId || !projectId || isRegenerating) return
    void persistGeneratedTake(regenVideoPath, 'video', regeneratingAssetId, regeneratingClipId)
  }, [
    isRegenerating,
    persistGeneratedTake,
    projectId,
    regeneratingAssetId,
    regeneratingClipId,
    regenVideoPath,
  ])

  useEffect(() => {
    if (cancelRequestedRef.current) return
    if (!regenImagePath || !regeneratingAssetId || !projectId || isRegenerating) return
    void persistGeneratedTake(regenImagePath, 'image', regeneratingAssetId, regeneratingClipId)
  }, [
    isRegenerating,
    persistGeneratedTake,
    projectId,
    regeneratingAssetId,
    regeneratingClipId,
    regenImagePath,
  ])

  // Keep UI clip state in sync if generation fails.
  // Do not reset generation error here; dialog owns that lifecycle.
  useEffect(() => {
    if (cancelRequestedRef.current) return
    if (!regeneratingAssetId || isRegenerating || !regenError) return
    cancelClipRegeneration()
  }, [cancelClipRegeneration, isRegenerating, regeneratingAssetId, regenError])

  useEffect(() => {
    if (!cancelRequestedRef.current || isRegenerating) return
    cancelRequestedRef.current = false
    cancelClipRegeneration()
    regenReset()
  }, [cancelClipRegeneration, isRegenerating, regenReset])

  return {
    regeneratingAssetId,
    regenerationPreError,
    handleRegenerate,
    handleCancelRegeneration,
    dismissRegenerationPreError,
  }
}
