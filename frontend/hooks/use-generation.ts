import { useState, useCallback, useRef } from 'react'
import type { GenerationSettings } from '../components/SettingsPanel'
import { backendFetch } from '../lib/backend'
import { useAppSettings } from '../contexts/AppSettingsContext'

interface GenerationState {
  isGenerating: boolean
  progress: number
  statusMessage: string
  videoUrl: string | null
  videoPath: string | null  // Original file path for upscaling
  imageUrl: string | null
  imageUrls: string[]  // For multiple image variations
  error: string | null
}

interface GenerationProgress {
  status: string
  phase: string
  progress: number
  currentStep: number | null
  totalSteps: number | null
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
    videoUrl: null,
    videoPath: null,
    imageUrl: null,
    imageUrls: [],
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
      videoUrl: null,
      videoPath: null,
      imageUrl: null,
      imageUrls: [],
      error: null,
    })

    abortControllerRef.current = new AbortController()
    let progressInterval: ReturnType<typeof setInterval> | null = null
    let shouldApplyPollingUpdates = true

    try {
      // Prepare JSON body
      const body: Record<string, unknown> = {
        prompt,
        model: settings.model,
        duration: String(settings.duration),
        resolution: settings.videoResolution,
        fps: String(settings.fps),
        audio: String(settings.audio),
        cameraMotion: settings.cameraMotion,
        aspectRatio: settings.aspectRatio || '16:9',
      }
      if (imagePath) {
        body.imagePath = imagePath
      }
      if (audioPath) {
        body.audioPath = audioPath
      }

      // Poll for real progress from backend with time-based interpolation
      let lastPhase = ''
      let inferenceStartTime = 0
      // Estimated inference time in seconds based on model
      const estimatedInferenceTime = settings.model === 'pro' ? 120 : 45
      
      const pollProgress = async () => {
        if (!shouldApplyPollingUpdates) return
        try {
          const res = await backendFetch('/api/generation/progress')
          if (res.ok) {
            const data: GenerationProgress = await res.json()
            if (!shouldApplyPollingUpdates) return

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
        } catch {
          // Ignore polling errors
        }
      }
      
      progressInterval = setInterval(pollProgress, 500)

      // Start generation (HTTP POST - synchronous, returns when done)
      const response = await backendFetch('/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        signal: abortControllerRef.current.signal,
      })
      shouldApplyPollingUpdates = false

      if (!response.ok) {
        const errorText = await response.text()
        throw new Error(errorText || 'Generation failed')
      }

      const result = await response.json()
      
      if (result.status === 'complete' && result.video_path) {
        // Convert Windows path to proper file:// URL
        const videoPathNormalized = result.video_path.replace(/\\/g, '/')
        const fileUrl = videoPathNormalized.startsWith('/') ? `file://${videoPathNormalized}` : `file:///${videoPathNormalized}`
        
        setState({
          isGenerating: false,
          progress: 100,
          statusMessage: 'Complete!',
          videoUrl: fileUrl,
          videoPath: result.video_path,  // Keep original path for API calls
          imageUrl: null,
          imageUrls: [],
          error: null,
        })
      } else if (result.status === 'cancelled') {
        setState(prev => ({
          ...prev,
          isGenerating: false,
          statusMessage: 'Cancelled',
        }))
      } else if (result.error) {
        throw new Error(result.error)
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
          error: error instanceof Error ? error.message : 'Unknown error',
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
    try {
      await backendFetch('/api/generate/cancel', { method: 'POST' })
    } catch {
      // Ignore errors from cancel request
    }
    
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
      try {
        const response = await backendFetch('/api/settings')
        if (response.ok) {
          const payload = await response.json()
          if (!payload?.hasFalApiKey) {
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
        }
      } catch {
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
      videoUrl: null,
      videoPath: null,
      imageUrl: null,
      imageUrls: [],
      error: null,
    })

    abortControllerRef.current = new AbortController()

    try {
      // Skip prompt enhancement for T2I - use original prompt directly
      const finalPrompt = prompt

      const dims = getImageDimensions(settings)
      const numSteps = settings.imageSteps || 4

      // Poll for progress
      const pollProgress = async () => {
        try {
          const res = await backendFetch('/api/generation/progress')
          if (res.ok) {
            const data = await res.json()
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
        } catch {
          // Ignore polling errors
        }
      }
      
      const progressInterval = setInterval(pollProgress, 500)

      const response = await backendFetch('/api/generate-image', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: finalPrompt,
          width: dims.width,
          height: dims.height,
          numSteps,
          numImages,
        }),
        signal: abortControllerRef.current.signal,
      })

      clearInterval(progressInterval)

      if (!response.ok) {
        const errorText = await response.text()
        throw new Error(errorText || 'Image generation failed')
      }

      const result = await response.json()
      
      if (result.status === 'complete') {
        // Handle both new format (image_paths array) and old format (single image_path)
        let rawPaths: string[] = []
        if (result.image_paths && Array.isArray(result.image_paths)) {
          rawPaths = result.image_paths
        } else if (result.image_path) {
          rawPaths = [result.image_path]
        }
        
        if (rawPaths.length > 0) {
          // Convert all paths to file URLs
          const fileUrls = rawPaths.map((path: string) => {
            const imagePath = path.replace(/\\/g, '/')
            return imagePath.startsWith('/') ? `file://${imagePath}` : `file:///${imagePath}`
          })
          
          setState({
            isGenerating: false,
            progress: 100,
            statusMessage: 'Complete!',
            videoUrl: null,
            videoPath: null,
            imageUrl: fileUrls[0],  // First image for backwards compatibility
            imageUrls: fileUrls,    // All images
            error: null,
          })
        }
      } else if (result.status === 'cancelled') {
        setState(prev => ({
          ...prev,
          isGenerating: false,
          statusMessage: 'Cancelled',
        }))
      } else if (result.error) {
        throw new Error(result.error)
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
          error: error instanceof Error ? error.message : 'Unknown error',
        }))
      }
    }
  }, [appSettings.hasFalApiKey, forceApiGenerations, refreshSettings])

  const reset = useCallback(() => {
    setState({
      isGenerating: false,
      progress: 0,
      statusMessage: '',
      videoUrl: null,
      videoPath: null,
      imageUrl: null,
      imageUrls: [],
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
