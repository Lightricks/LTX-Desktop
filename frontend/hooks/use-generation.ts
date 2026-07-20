import { useState, useCallback, useRef } from 'react'
import type { GenerationSettings } from '../components/SettingsPanel'
import { ApiClient, type ApiRequestBodyOf, type ApiSuccessOf } from '../lib/api-client'
import { createLocalGenerationError, type GenerationError } from '../lib/generation-errors'
import { useAppSettings } from '../contexts/AppSettingsContext'
import { getBackendCredentials } from '../lib/backend'
import {
  materializeBackendOutput,
  materializeBackendOutputs,
  prepareBackendMedia,
} from '../lib/backend-media'

interface GenerationState {
  isGenerating: boolean
  progress: number
  statusMessage: string
  videoPath: string | null
  imagePath: string | null
  imagePaths: string[]
  error: GenerationError | null
}

type GenerateVideoRequest = ApiRequestBodyOf<'generateVideo'>
type GenerateImageRequest = ApiRequestBodyOf<'generateImage'>
type GenerationProgressPayload = ApiSuccessOf<'getGenerationProgress'>

const REMOTE_RECOVERY_TIMEOUT_MS = 30 * 60 * 1000
const REMOTE_RECOVERY_POLL_MS = 2_000

function isTransportFailure(value: unknown): boolean {
  if (!value || typeof value !== 'object') return false
  const result = value as { ok?: unknown; error?: { code?: unknown } }
  return result.ok === false
    && (result.error?.code === 'NETWORK_ERROR' || result.error?.code === 'RESPONSE_READ_FAILED')
}

function waitForRemoteRetry(signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal.aborted) {
      reject(new DOMException('Generation cancelled', 'AbortError'))
      return
    }
    const onAbort = () => {
      window.clearTimeout(timeout)
      reject(new DOMException('Generation cancelled', 'AbortError'))
    }
    const timeout = window.setTimeout(() => {
      signal.removeEventListener('abort', onAbort)
      resolve()
    }, REMOTE_RECOVERY_POLL_MS)
    signal.addEventListener('abort', onAbort, { once: true })
  })
}

async function recoverRemoteGeneration(
  generationId: string,
  signal: AbortSignal,
  onProgress: (progress: GenerationProgressPayload) => void,
): Promise<GenerationProgressPayload> {
  const deadline = Date.now() + REMOTE_RECOVERY_TIMEOUT_MS

  while (Date.now() < deadline) {
    const progressResult = await ApiClient.getGenerationProgress(undefined, { signal })
    if (progressResult.ok) {
      const progress = progressResult.data
      if (progress.generationId === generationId) {
        onProgress(progress)
        if (progress.status === 'complete') return progress
        if (progress.status === 'cancelled') {
          throw new DOMException('Generation cancelled', 'AbortError')
        }
        if (progress.status === 'error') {
          throw new Error(progress.error || 'The remote generation failed')
        }
      }
    }
    await waitForRemoteRetry(signal)
  }

  throw new Error('Timed out while waiting to recover the remote generation')
}

interface UseGenerationReturn extends GenerationState {
  generate: (prompt: string, imagePath: string | null, settings: GenerationSettings, audioPath?: string | null) => Promise<void>
  generateImage: (prompt: string, settings: GenerationSettings) => Promise<void>
  cancel: () => void
  reset: () => void
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
    default:
      return 'Generating...'
  }
}

export function useGeneration(): UseGenerationReturn {
  const { settings: appSettings, forceApiGenerations, refreshSettings } = useAppSettings()
  const [state, setState] = useState<GenerationState>({
    isGenerating: false,
    progress: 0,
    statusMessage: '',
    videoPath: null,
    imagePath: null,
    imagePaths: [],
    error: null,
  })

  const abortControllerRef = useRef<AbortController | null>(null)

  const generate = useCallback(async (
    prompt: string,
    imagePath: string | null,
    settings: GenerationSettings,
    audioPath?: string | null,
  ) => {
    const statusMsg = settings.model === 'pro'
      ? 'Loading Pro model & generating...'
      : 'Generating video...'

    setState({
      isGenerating: true,
      progress: 0,
      statusMessage: statusMsg,
      videoPath: null,
      imagePath: null,
      imagePaths: [],
      error: null,
    })

    abortControllerRef.current = new AbortController()
    const generationId = crypto.randomUUID()
    let progressInterval: ReturnType<typeof setInterval> | null = null
    let shouldApplyPollingUpdates = true

    try {
      if (imagePath || audioPath) {
        setState(prev => ({
          ...prev,
          statusMessage: imagePath && audioPath
            ? 'Uploading image and audio...'
            : imagePath ? 'Uploading image...' : 'Uploading audio...',
        }))
      }
      const [preparedImage, preparedAudio] = await Promise.all([
        prepareBackendMedia(imagePath, 'image'),
        prepareBackendMedia(audioPath, 'audio'),
      ])

      // Prepare JSON body
      const body: Record<string, unknown> = {
        generationId,
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
      if (preparedImage?.path) {
        body.imagePath = preparedImage.path
      } else if (preparedImage?.mediaId) {
        body.imageMediaId = preparedImage.mediaId
      }
      if (preparedAudio?.path) {
        body.audioPath = preparedAudio.path
      } else if (preparedAudio?.mediaId) {
        body.audioMediaId = preparedAudio.mediaId
      }

      // Poll for real progress from backend with time-based interpolation
      let lastPhase = ''
      let inferenceStartTime = 0
      // Estimated inference time in seconds based on model
      const estimatedInferenceTime = settings.model === 'pro' ? 120 : 45
      
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

        setState(prev => ({
          ...prev,
          progress: displayProgress,
          statusMessage,
        }))
      }
      
      progressInterval = setInterval(pollProgress, 500)

      // Start generation (HTTP POST - synchronous, returns when done)
      const backend = await getBackendCredentials()
      const result = await ApiClient.generateVideo(body as unknown as GenerateVideoRequest, {
        signal: abortControllerRef.current.signal,
      })
      shouldApplyPollingUpdates = false
      let payload: ApiSuccessOf<'generateVideo'>
      if (!result.ok) {
        if (backend.mode === 'external' && isTransportFailure(result)) {
          setState(prev => ({
            ...prev,
            statusMessage: 'Connection lost. Waiting for the remote generation...',
          }))
          const recovered = await recoverRemoteGeneration(
            generationId,
            abortControllerRef.current.signal,
            (progress) => {
              setState(prev => ({
                ...prev,
                progress: progress.progress,
                statusMessage: progress.status === 'running'
                  ? getPhaseMessage(progress.phase)
                  : 'Downloading recovered output...',
              }))
            },
          )
          const recoveredArtifact = recovered.artifact ?? recovered.artifacts?.[0]
          if (!recoveredArtifact) {
            throw new Error('The recovered remote generation has no downloadable artifact')
          }
          payload = {
            status: 'complete',
            video_path: '',
            artifact: recoveredArtifact,
          }
        } else {
          setState(prev => ({
            ...prev,
            isGenerating: false,
            error: result,
          }))
          return
        }
      } else {
        payload = result.data
      }

      if (payload.status === 'complete') {
        setState(prev => ({ ...prev, statusMessage: 'Downloading output...', progress: 97 }))
        const outputPath = await materializeBackendOutput(payload.video_path, payload.artifact)
        setState({
          isGenerating: false,
          progress: 100,
          statusMessage: 'Complete!',
          videoPath: outputPath,
          imagePath: null,
          imagePaths: [],
          error: null,
        })
      } else if (payload.status === 'cancelled') {
        setState(prev => ({
          ...prev,
          isGenerating: false,
          statusMessage: 'Cancelled',
        }))
      } else {
        throw new Error('Unexpected response from /api/generate')
      }

    } catch (error) {
      if (error instanceof Error && error.name === 'AbortError') {
        setState(prev => ({
          ...prev,
          isGenerating: false,
          statusMessage: 'Cancelled',
        }))
      } else {
        setState(prev => ({
          ...prev,
          isGenerating: false,
          error: createLocalGenerationError(error instanceof Error ? error.message : 'Unknown error'),
        }))
      }
    } finally {
      shouldApplyPollingUpdates = false
      if (progressInterval) {
        clearInterval(progressInterval)
      }
    }
  }, [])

  const cancel = useCallback(async () => {
    // Abort the fetch request
    abortControllerRef.current?.abort()
    
    // Also tell the backend to cancel
    void ApiClient.cancelGeneration()
    
    setState(prev => ({
      ...prev,
      isGenerating: false,
      statusMessage: 'Cancelled',
    }))
  }, [])

  const generateImage = useCallback(async (
    prompt: string,
    settings: GenerationSettings
  ) => {
    if (forceApiGenerations) {
      const settingsResult = await ApiClient.getSettings()
      if (settingsResult.ok) {
        if (!settingsResult.data.hasFalApiKey) {
          void refreshSettings()
          window.dispatchEvent(new CustomEvent('open-api-gateway', {
            detail: {
              requiredKeys: ['fal'],
              title: 'Connect FAL AI',
              description: 'FAL AI is required for generating images with Z Image Turbo when API generations are enabled.',
              blocking: false,
            },
          }))
          return
        }
      } else {
        if (!appSettings.hasFalApiKey) {
          window.dispatchEvent(new CustomEvent('open-api-gateway', {
            detail: {
              requiredKeys: ['fal'],
              title: 'Connect FAL AI',
              description: 'FAL AI is required for generating images with Z Image Turbo when API generations are enabled.',
              blocking: false,
            },
          }))
          return
        }
      }
    }

    const numImages = settings.variations || 1
    
    setState({
      isGenerating: true,
      progress: 0,
      statusMessage: numImages > 1 ? `Generating ${numImages} images...` : 'Generating image...',
      videoPath: null,
      imagePath: null,
      imagePaths: [],
      error: null,
    })

    abortControllerRef.current = new AbortController()
    const generationId = crypto.randomUUID()
    let progressInterval: ReturnType<typeof setInterval> | null = null
    let shouldApplyPollingUpdates = true

    try {
      // Skip prompt enhancement for T2I - use original prompt directly
      const finalPrompt = prompt

      const dims = getImageDimensions(settings)
      const numSteps = settings.imageSteps || 4

      // Poll for progress
      const pollProgress = async () => {
        if (!shouldApplyPollingUpdates) return
        const result = await ApiClient.getGenerationProgress()
        if (!result.ok || !shouldApplyPollingUpdates) return

        const data = result.data
        const currentImage = data.currentStep || 0
        const totalImages = data.totalSteps || numImages
        setState(prev => ({
          ...prev,
          progress: data.progress,
          statusMessage: data.phase === 'loading_model'
            ? 'Loading Z-Image Turbo model...'
            : data.phase === 'inference'
              ? numImages > 1
                ? `Generating image ${currentImage + 1}/${totalImages}...`
                : 'Generating image...'
              : data.phase === 'complete'
                ? 'Complete!'
                : 'Generating...',
        }))
      }
      
      progressInterval = setInterval(pollProgress, 500)

      const imageRequest: GenerateImageRequest = {
        generationId,
        prompt: finalPrompt,
        width: dims.width,
        height: dims.height,
        numSteps,
        numImages,
      }
      const backend = await getBackendCredentials()
      const result = await ApiClient.generateImage(imageRequest, {
        signal: abortControllerRef.current.signal,
      })

      shouldApplyPollingUpdates = false
      let payload: ApiSuccessOf<'generateImage'>
      if (!result.ok) {
        if (backend.mode === 'external' && isTransportFailure(result)) {
          setState(prev => ({
            ...prev,
            statusMessage: 'Connection lost. Waiting for the remote generation...',
          }))
          const recovered = await recoverRemoteGeneration(
            generationId,
            abortControllerRef.current.signal,
            (progress) => {
              setState(prev => ({
                ...prev,
                progress: progress.progress,
                statusMessage: progress.status === 'running'
                  ? getPhaseMessage(progress.phase)
                  : 'Downloading recovered output...',
              }))
            },
          )
          payload = {
            status: 'complete',
            image_paths: [],
            artifacts: recovered.artifacts ?? [],
          }
        } else {
          setState(prev => ({
            ...prev,
            isGenerating: false,
            error: result,
          }))
          return
        }
      } else {
        payload = result.data
      }

      if (payload.status === 'complete') {
        setState(prev => ({ ...prev, statusMessage: 'Downloading output...', progress: 97 }))
        const rawPaths = await materializeBackendOutputs(payload.image_paths, payload.artifacts)
        if (rawPaths.length === 0) {
          throw new Error('Image generation completed without output images')
        }

        setState({
          isGenerating: false,
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
          statusMessage: 'Cancelled',
        }))
      } else {
        throw new Error('Unexpected response from /api/generate-image')
      }

    } catch (error) {
      if (error instanceof Error && error.name === 'AbortError') {
        setState(prev => ({
          ...prev,
          isGenerating: false,
          statusMessage: 'Cancelled',
        }))
      } else {
        setState(prev => ({
          ...prev,
          isGenerating: false,
          error: createLocalGenerationError(error instanceof Error ? error.message : 'Unknown error'),
        }))
      }
    } finally {
      shouldApplyPollingUpdates = false
      if (progressInterval) clearInterval(progressInterval)
    }
  }, [appSettings.hasFalApiKey, forceApiGenerations, refreshSettings])

  const reset = useCallback(() => {
    setState({
      isGenerating: false,
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
  }
}
