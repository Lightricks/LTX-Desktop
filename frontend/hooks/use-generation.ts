import { useState, useCallback, useRef, useEffect } from 'react'
import type { GenerationSettings } from '../components/SettingsPanel'
import { ApiClient, type ApiRequestBodyOf, type ApiSuccessOf } from '../lib/api-client'
import { createLocalGenerationError, type GenerationError } from '../lib/generation-errors'
import { canCancelLocalJob, withGenerationActive } from '../lib/generation-active'
import { useAppSettings } from '../contexts/AppSettingsContext'

const POLLING_INTERVAL_MS = 2000

export const GENERATION_RECOVERY_KEY = 'ltx-generation-recovery'

export interface GenerationRecoveryContext {
  projectId: string
  prompt: string
  // Absent for ic-lora/retake: those recover as standalone video assets (Phase 1),
  // so there are no video/image settings to restore.
  settings?: GenerationSettings
  // Retake/extend write this instead of a full `settings` blob — the recovery importer
  // prefers it over `settings.model` (which defaults to 'fast' when absent).
  model?: string
  inputImageUrl?: string
  inputAudioUrl?: string
  genType?: 'image' | 'enhance'
  // Frozen at marker write (job start) — same rule as hook canCancel. Lets Stop survive a
  // UI refresh before the first progress poll returns (local GPU can starve that poll).
  // Absent on older markers: treat as not cancellable until the poll reports it.
  canCancel?: boolean
  // Whatever generation id the backend reported at the moment this marker was written — i.e.
  // immediately BEFORE this generation started. The handler that starts a generation loads its
  // pipeline (can take many seconds — worse for image models loading checkpoint shards) before
  // it ever reports a new id, so a poll can otherwise be looking at a stale, unrelated id/result
  // that predates this marker entirely. Once a later poll observes a DIFFERENT id, that's proof
  // (single global generation slot) that this marker's own generation has started — see
  // checkAndConsumeRecovery in lib/generation-recovery.ts.
  baselineId: string | null
  // Set once a poll observes an id different from baselineId — i.e. once this marker's own
  // generation is confirmed to exist. Distinct from baselineId: a LATER id change past this point
  // means a DIFFERENT generation superseded ours (not that ours just started), which must NOT be
  // imported under this marker.
  generationId?: string
}

export function readRecoveryMarkerCanCancel(): boolean {
  const saved = localStorage.getItem(GENERATION_RECOVERY_KEY)
  if (!saved) return false
  try {
    return (JSON.parse(saved) as GenerationRecoveryContext).canCancel === true
  } catch {
    return false
  }
}

interface GenerationState {
  isGenerating: boolean
  isCancelling: boolean
  /** Frozen at job start — Stop stays hidden if this POST was an LTX/FAL cloud job. */
  canCancel: boolean
  progress: number
  statusMessage: string
  videoPath: string | null
  imagePath: string | null
  imagePaths: string[]
  error: GenerationError | null
}

type GenerateVideoRequest = ApiRequestBodyOf<'generateVideo'>
type GenerateImageRequest = ApiRequestBodyOf<'generateImage'>

interface UseGenerationReturn extends GenerationState {
  generate: (prompt: string, imagePath: string | null, settings: GenerationSettings, audioPath?: string | null) => Promise<void>
  generateImage: (prompt: string, settings: GenerationSettings, editSource?: string | null) => Promise<void>
  cancel: () => void
  reset: () => void
  resumeIfRunning: () => Promise<'running' | 'complete' | 'none'>
}

const IMAGE_SHORT_SIDE_BY_RESOLUTION: Record<string, number> = {
  '1080p': 1080,
  '1440p': 1440,
  '2048p': 2048,
}

const IMAGE_ASPECT_RATIO_VALUE: Record<string, number> = {
  '1:1': 1,
  '16:9': 16 / 9,
  '9:16': 9 / 16,
  '4:3': 4 / 3,
  '3:4': 3 / 4,
  '21:9': 21 / 9,
}

function getImageDimensions(settings: GenerationSettings): { width: number; height: number } {
  const shortSide = IMAGE_SHORT_SIDE_BY_RESOLUTION[settings.imageResolution]
  if (!shortSide) {
    throw new Error(`Unsupported image resolution mapping: ${settings.imageResolution}`)
  }

  const ratio = IMAGE_ASPECT_RATIO_VALUE[settings.imageAspectRatio]
  if (!ratio) {
    throw new Error(`Unsupported image aspect ratio mapping: ${settings.imageAspectRatio}`)
  }

  if (ratio >= 1) {
    return { width: Math.round(shortSide * ratio), height: shortSide }
  }
  return { width: shortSide, height: Math.round(shortSide / ratio) }
}

// Map phase to user-friendly message
function getPhaseMessage(phase: string): string {
  switch (phase) {
    case 'validating_request':
      return 'Validating request...'
    case 'uploading_image':
      return 'Uploading image...'
    case 'uploading_audio':
      return 'Uploading audio...'
    case 'loading_model':
      return 'Loading model...'
    case 'encoding_text':
      return 'Encoding prompt...'
    case 'inference':
      return 'Generating...'
    case 'downloading_output':
      return 'Downloading output...'
    case 'decoding':
      return 'Decoding video...'
    case 'complete':
      return 'Complete!'
    case 'cancelled':
      return 'Cancelling…'
    default:
      return 'Generating...'
  }
}

export function useGeneration(): UseGenerationReturn {
  const { settings: appSettings, shouldImageGenerateWithFalApi, shouldVideoGenerateWithLtxApi, refreshSettings } = useAppSettings()
  const [state, setState] = useState<GenerationState>({
    isGenerating: false,
    isCancelling: false,
    canCancel: false,
    progress: 0,
    statusMessage: '',
    videoPath: null,
    imagePath: null,
    imagePaths: [],
    error: null,
  })

  const recoveryIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const clearRecoveryPolling = () => {
    if (recoveryIntervalRef.current) {
      clearInterval(recoveryIntervalRef.current)
      recoveryIntervalRef.current = null
    }
  }

  useEffect(() => clearRecoveryPolling, [])

  // Re-attach to a generation that was running OR finished while the frontend was
  // unmounted. Polls the backend progress endpoint; localStorage recovery context
  // (inputs, settings incl. loras) is owned by the caller (GenSpace). Returns the
  // recovered status so the caller can restore context for 'running' AND 'complete'
  // (a generation that finished during the unmount window still needs its metadata).
  const resumeIfRunning = useCallback(async (): Promise<'running' | 'complete' | 'none'> => {
    const apply = (data: ApiSuccessOf<'getGenerationProgress'>): 'running' | 'complete' | 'other' => {
      if (data.status === 'complete' && data.result != null) {
        const vp = typeof data.result === 'string' ? data.result : null
        const ips = Array.isArray(data.result) ? data.result : []
        setState({
          isGenerating: false, isCancelling: false, canCancel: false, progress: 100, statusMessage: 'Complete!',
          videoPath: vp, imagePath: ips[0] ?? null, imagePaths: ips, error: null,
        })
        return 'complete'
      }
      if (data.status === 'running') {
        setState(prev => ({
          ...prev,
          isGenerating: true,
          isCancelling: data.phase === 'cancelled',
          canCancel: data.cancellable,
          progress: data.progress,
          statusMessage: getPhaseMessage(data.phase),
        }))
        return 'running'
      }
      setState(prev => ({ ...prev, isGenerating: false, isCancelling: false, canCancel: false, statusMessage: '' }))
      return 'other'
    }

    const initial = await ApiClient.getGenerationProgress()
    if (!initial.ok) return 'none'
    const status = apply(initial.data)
    if (status === 'complete') return 'complete'
    if (status !== 'running') return 'none'

    clearRecoveryPolling()
    recoveryIntervalRef.current = setInterval(async () => {
      const r = await ApiClient.getGenerationProgress()
      if (!r.ok) return
      if (apply(r.data) !== 'running') clearRecoveryPolling()
    }, POLLING_INTERVAL_MS)
    return 'running'
  }, [])

  const generate = useCallback(async (
    prompt: string,
    imagePath: string | null,
    settings: GenerationSettings,
    audioPath?: string | null,
  ) => {
    const statusMsg = settings.model.startsWith('pro')
      ? 'Loading Pro model & generating...'
      : 'Generating video...'

    setState({
      isGenerating: true,
      isCancelling: false,
      canCancel: canCancelLocalJob('video', shouldVideoGenerateWithLtxApi, shouldImageGenerateWithFalApi),
      progress: 0,
      statusMessage: statusMsg,
      videoPath: null,
      imagePath: null,
      imagePaths: [],
      error: null,
    })

    let progressInterval: ReturnType<typeof setInterval> | null = null
    let shouldApplyPollingUpdates = true

    await withGenerationActive(async () => {
      try {
        // Prepare JSON body
        const body: Record<string, unknown> = {
          prompt,
          model: settings.model,
          duration: settings.duration,
          resolution: settings.videoResolution,
          fps: settings.fps,
          audio: settings.audio,
          cameraMotion: settings.cameraMotion,
          negativePrompt: (settings as { negativePrompt?: string }).negativePrompt ?? '',
          aspectRatio: settings.aspectRatio || '16:9',
        }
        if (imagePath) {
          body.imagePath = imagePath
        }
        if (audioPath) {
          body.audioPath = audioPath
        }
        if (settings.loras?.length) {
          body.loras = settings.loras.map(l => ({ ref: l.ref, scale: l.scale }))
        }

        // Poll for real progress from backend with time-based interpolation
        let lastPhase = ''
        let inferenceStartTime = 0
        // Estimated inference time in seconds based on model
        const estimatedInferenceTime = settings.model.startsWith('pro') ? 120 : 45

        const pollProgress = async () => {
          if (!shouldApplyPollingUpdates) return
          const result = await ApiClient.getGenerationProgress()
          if (!result.ok || !shouldApplyPollingUpdates) return

          const data = result.data
          let displayProgress = data.progress
          let statusMessage = getPhaseMessage(data.phase)

          // Time-based interpolation during inference phase
          if (data.phase === 'inference') {
            if (lastPhase !== 'inference') {
              inferenceStartTime = Date.now()
            }
            const elapsed = (Date.now() - inferenceStartTime) / 1000
            // Interpolate from 15% to 95% based on estimated time
            const inferenceProgress = Math.min(elapsed / estimatedInferenceTime, 0.95)
            displayProgress = 15 + Math.floor(inferenceProgress * 80)
          }

          // Keep API/local completion as a terminal response state, not polling state.
          // Polling complete means backend state is finalized, but request can still be in-flight.
          if (data.phase === 'complete' || data.status === 'complete') {
            displayProgress = 95
            statusMessage = 'Finalizing...'
          }

          lastPhase = data.phase

          setState(prev => {
            if (prev.isCancelling) {
              return { ...prev, statusMessage: 'Cancelling…' }
            }
            return {
              ...prev,
              progress: displayProgress,
              statusMessage,
            }
          })
        }

        progressInterval = setInterval(pollProgress, 500)

        // Start generation (HTTP POST - synchronous, returns when done)
        // Do not abort this POST: liveness suppression stays up until the
        // backend returns {status: "cancelled"} and the GPU job unwinds.
        const result = await ApiClient.generateVideo(body as unknown as GenerateVideoRequest)
        shouldApplyPollingUpdates = false
        if (!result.ok) {
          setState(prev => ({
            ...prev,
            isGenerating: false,
            isCancelling: false,
            canCancel: false,
            error: result,
          }))
          return
        }

        const payload = result.data
        if (payload.status === 'complete') {
          setState({
            isGenerating: false,
            isCancelling: false,
            canCancel: false,
            progress: 100,
            statusMessage: 'Complete!',
            videoPath: payload.video_path,
            imagePath: null,
            imagePaths: [],
            error: null,
          })
        } else if (payload.status === 'cancelled') {
          setState(prev => ({
            ...prev,
            isGenerating: false,
            isCancelling: false,
            canCancel: false,
            statusMessage: 'Cancelled',
          }))
        } else {
          throw new Error('Unexpected response from /api/generate')
        }

      } catch (error) {
        setState(prev => ({
          ...prev,
          isGenerating: false,
          isCancelling: false,
          canCancel: false,
          error: createLocalGenerationError(error instanceof Error ? error.message : 'Unknown error'),
        }))
      } finally {
        shouldApplyPollingUpdates = false
        if (progressInterval) {
          clearInterval(progressInterval)
        }
      }
    })
  }, [shouldImageGenerateWithFalApi, shouldVideoGenerateWithLtxApi])

  const cancel = useCallback(() => {
    let claimedCancelling = false
    setState(prev => {
      if (!prev.isGenerating || prev.isCancelling) return prev
      claimedCancelling = true
      return {
        ...prev,
        isCancelling: true,
        statusMessage: 'Cancelling…',
      }
    })
    // Always POST — retake/extend/IC-LoRA Stop reuse this while this hook is idle.
    void (async () => {
      const result = await ApiClient.cancelGeneration()
      const accepted = result.ok && result.data.status === 'cancelling'
      if (accepted || !claimedCancelling) return
      setState(prev => {
        if (!prev.isCancelling) return prev
        return { ...prev, isCancelling: false }
      })
    })()
  }, [])

  const generateImage = useCallback(async (
    prompt: string,
    settings: GenerationSettings,
    editSource?: string | null,
  ) => {
    const isEditing = !!editSource

    const openFalConnectDialog = () => {
      window.dispatchEvent(new CustomEvent('open-api-gateway', {
        detail: {
          requiredKeys: ['fal'],
          title: 'Connect FAL AI',
          description: `FAL AI is required for ${isEditing ? 'editing' : 'generating'} images with Z Image Turbo when API generations are enabled.`,
          blocking: false,
        },
      }))
    }

    if (shouldImageGenerateWithFalApi) {
      const settingsResult = await ApiClient.getSettings()
      const hasFalApiKey = settingsResult.ok ? settingsResult.data.hasFalApiKey : appSettings.hasFalApiKey
      if (!hasFalApiKey) {
        if (settingsResult.ok) void refreshSettings()
        openFalConnectDialog()
        return
      }
    }

    const numImages = settings.variations || 1

    setState({
      isGenerating: true,
      isCancelling: false,
      canCancel: canCancelLocalJob('image', shouldVideoGenerateWithLtxApi, shouldImageGenerateWithFalApi),
      progress: 0,
      statusMessage: isEditing
        ? 'Editing image...'
        : numImages > 1 ? `Generating ${numImages} images...` : 'Generating image...',
      videoPath: null,
      imagePath: null,
      imagePaths: [],
      error: null,
    })

    await withGenerationActive(async () => {
      let progressInterval: ReturnType<typeof setInterval> | null = null
      try {
        // Skip prompt enhancement for T2I - use original prompt directly
        const finalPrompt = prompt

        // Edit runs at the source image's resolution; width/height are ignored server-side.
        const dims = isEditing ? { width: 1024, height: 1024 } : getImageDimensions(settings)
        const numSteps = settings.imageSteps || (isEditing ? 8 : 4)

        // Poll for progress
        const pollProgress = async () => {
          const result = await ApiClient.getGenerationProgress()
          if (!result.ok) return

          const data = result.data
          const currentImage = data.currentStep || 0
          const totalImages = data.totalSteps || numImages
          setState(prev => {
            if (prev.isCancelling) {
              return { ...prev, statusMessage: 'Cancelling…' }
            }
            return {
              ...prev,
              progress: data.progress,
              statusMessage: data.phase === 'loading_model'
                ? 'Loading Z-Image Turbo model...'
                : data.phase === 'inference'
                  ? isEditing
                    ? 'Editing image...'
                    : numImages > 1
                      ? `Generating image ${currentImage + 1}/${totalImages}...`
                      : 'Generating image...'
                  : data.phase === 'complete'
                    ? 'Complete!'
                    : 'Generating...',
            }
          })
        }

        progressInterval = setInterval(pollProgress, 500)

        const imageRequest: GenerateImageRequest = {
          prompt: finalPrompt,
          width: dims.width,
          height: dims.height,
          numSteps,
          numImages,
          // strength is ignored server-side unless imagePath is set, but the request type
          // requires it — send the default rather than the edit-only setting when not editing.
          strength: isEditing ? (settings.imageEditStrength ?? 0.6) : 0.6,
          ...(isEditing ? { imagePath: editSource } : {}),
        }
        const result = await ApiClient.generateImage(imageRequest)

        if (!result.ok) {
          setState(prev => ({
            ...prev,
            isGenerating: false,
            isCancelling: false,
            canCancel: false,
            error: result,
          }))
          return
        }

        const payload = result.data
        if (payload.status === 'complete') {
          const rawPaths = payload.image_paths
          if (rawPaths.length === 0) {
            throw new Error('Image generation completed without output images')
          }

          setState({
            isGenerating: false,
            isCancelling: false,
            canCancel: false,
            progress: 100,
            statusMessage: 'Complete!',
            videoPath: null,
            imagePath: rawPaths[0],
            imagePaths: rawPaths,
            error: null,
          })
        } else if (payload.status === 'cancelled') {
          setState(prev => ({
            ...prev,
            isGenerating: false,
            isCancelling: false,
            canCancel: false,
            statusMessage: 'Cancelled',
          }))
        } else {
          throw new Error('Unexpected response from /api/generate-image')
        }

      } catch (error) {
        setState(prev => ({
          ...prev,
          isGenerating: false,
          isCancelling: false,
          canCancel: false,
          error: createLocalGenerationError(error instanceof Error ? error.message : 'Unknown error'),
        }))
      } finally {
        if (progressInterval) {
          clearInterval(progressInterval)
        }
      }
    })
  }, [appSettings.hasFalApiKey, shouldImageGenerateWithFalApi, shouldVideoGenerateWithLtxApi, refreshSettings])

  const reset = useCallback(() => {
    clearRecoveryPolling()
    localStorage.removeItem(GENERATION_RECOVERY_KEY)
    setState({
      isGenerating: false,
      isCancelling: false,
      canCancel: false,
      progress: 0,
      statusMessage: '',
      videoPath: null,
      imagePath: null,
      imagePaths: [],
      error: null,
    })
  }, [])

  return {
    ...state,
    generate,
    generateImage,
    cancel,
    reset,
    resumeIfRunning,
  }
}
