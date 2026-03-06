import { useState, useCallback, useRef, useEffect } from 'react'
import type { GenerationSettings } from '../components/SettingsPanel'
import { useAppSettings } from '../contexts/AppSettingsContext'

export interface QueueJob {
  id: string
  type: string
  model: string
  params: Record<string, unknown>
  status: string
  slot: string
  progress: number
  phase: string
  result_paths: string[]
  error: string | null
  created_at: string
}

interface GenerationState {
  isGenerating: boolean
  progress: number
  statusMessage: string
  videoUrl: string | null
  videoPath: string | null  // Original file path for upscaling
  imageUrl: string | null
  imageUrls: string[]  // For multiple image variations
  error: string | null
  jobs: QueueJob[]
  lastModel: string | null
}

interface UseGenerationReturn extends GenerationState {
  generate: (prompt: string, imagePath: string | null, settings: GenerationSettings, audioPath?: string | null, lastFramePath?: string | null) => Promise<void>
  generateImage: (prompt: string, settings: GenerationSettings) => Promise<void>
  cancel: () => void
  reset: () => void
  clearQueue: () => void
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
  '4:5': 4 / 5,
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

// Convert a file system path to a file:// URL
function pathToFileUrl(filePath: string): string {
  const normalized = filePath.replace(/\\/g, '/')
  return normalized.startsWith('/') ? `file://${normalized}` : `file:///${normalized}`
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
    jobs: [],
    lastModel: null,
  })

  // Track the most recently submitted job ID for cancel
  const activeJobIdRef = useRef<string | null>(null)
  const pollIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null)

  // Start polling the queue status. Cleans up automatically when no active jobs remain.
  const startPolling = useCallback(() => {
    if (pollIntervalRef.current) return // already polling

    const poll = async () => {
      try {
        const backendUrl = await window.electronAPI.getBackendUrl()
        const res = await fetch(`${backendUrl}/api/queue/status`)
        if (!res.ok) return
        const data: { jobs: QueueJob[] } = await res.json()
        const jobs: QueueJob[] = data.jobs

        // Derive progress from the most-recently active job
        const activeId = activeJobIdRef.current
        const activeJob = activeId ? jobs.find(j => j.id === activeId) : null

        // Only consider *our* active job as running — stale/other jobs shouldn't block the UI
        const hasRunning = activeJob
          ? (activeJob.status === 'queued' || activeJob.status === 'running')
          : false

        setState(prev => {
          const next = { ...prev, jobs }

          if (activeJob) {
            next.progress = activeJob.progress
            next.statusMessage = getPhaseMessage(activeJob.phase)

            if (activeJob.status === 'complete') {
              next.isGenerating = hasRunning
              next.progress = 100
              next.statusMessage = 'Complete!'

              next.lastModel = activeJob.model

              if (activeJob.type === 'video' && activeJob.result_paths.length > 0) {
                const rawPath = activeJob.result_paths[0]
                next.videoUrl = pathToFileUrl(rawPath)
                next.videoPath = rawPath
              } else if (activeJob.type === 'image' && activeJob.result_paths.length > 0) {
                const fileUrls = activeJob.result_paths.map(pathToFileUrl)
                next.imageUrl = fileUrls[0]
                next.imageUrls = fileUrls
              }

              // Clear active job so we don't keep overwriting state
              activeJobIdRef.current = null
            } else if (activeJob.status === 'error') {
              next.isGenerating = hasRunning
              next.error = activeJob.error || 'Generation failed'
              activeJobIdRef.current = null
            } else if (activeJob.status === 'cancelled') {
              next.isGenerating = hasRunning
              next.statusMessage = 'Cancelled'
              activeJobIdRef.current = null
            }
          } else {
            next.isGenerating = hasRunning
          }

          return next
        })

        // Stop polling when nothing is active
        if (!hasRunning) {
          if (pollIntervalRef.current) {
            clearInterval(pollIntervalRef.current)
            pollIntervalRef.current = null
          }
        }
      } catch {
        // Ignore polling errors
      }
    }

    // Fire immediately, then every 500ms
    void poll()
    pollIntervalRef.current = setInterval(poll, 500)
  }, [])

  // Clean up polling on unmount
  useEffect(() => {
    return () => {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current)
        pollIntervalRef.current = null
      }
    }
  }, [])

  const generate = useCallback(async (
    prompt: string,
    imagePath: string | null,
    settings: GenerationSettings,
    audioPath?: string | null,
    lastFramePath?: string | null,
  ) => {
    const statusMsg = settings.model === 'pro'
      ? 'Loading Pro model & generating...'
      : 'Generating video...'

    setState(prev => ({
      ...prev,
      isGenerating: true,
      progress: 0,
      statusMessage: statusMsg,
      videoUrl: null,
      videoPath: null,
      imageUrl: null,
      imageUrls: [],
      error: null,
    }))

    try {
      const backendUrl = await window.electronAPI.getBackendUrl()

      const params: Record<string, unknown> = {
        prompt,
        duration: String(settings.duration),
        resolution: settings.videoResolution,
        fps: String(settings.fps),
        audio: String(settings.audio),
        cameraMotion: settings.cameraMotion,
        aspectRatio: settings.aspectRatio || '16:9',
      }
      if (imagePath) {
        params.imagePath = imagePath
      }
      if (audioPath) {
        params.audioPath = audioPath
      }
      if (lastFramePath) {
        params.lastFramePath = lastFramePath
      }

      const response = await fetch(`${backendUrl}/api/queue/submit`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          type: 'video',
          model: settings.model,
          params,
        }),
      })

      if (!response.ok) {
        const errorText = await response.text()
        throw new Error(errorText || 'Failed to submit video generation job')
      }

      const result: { id: string; status: string } = await response.json()
      activeJobIdRef.current = result.id
      startPolling()
    } catch (error) {
      setState(prev => ({
        ...prev,
        isGenerating: false,
        error: error instanceof Error ? error.message : 'Unknown error',
      }))
    }
  }, [startPolling])

  const cancel = useCallback(async () => {
    const jobId = activeJobIdRef.current
    if (!jobId) return

    try {
      const backendUrl = await window.electronAPI.getBackendUrl()
      await fetch(`${backendUrl}/api/queue/cancel/${jobId}`, {
        method: 'POST',
      })
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
        const backendUrl = await window.electronAPI.getBackendUrl()
        const response = await fetch(`${backendUrl}/api/settings`)
        if (response.ok) {
          const payload = await response.json()
          if (!payload?.hasReplicateApiKey) {
            void refreshSettings()
            window.dispatchEvent(new CustomEvent('open-api-gateway', {
              detail: {
                requiredKeys: ['replicate'],
                title: 'Connect Replicate',
                description: 'Replicate is required for generating images when API generations are enabled.',
                blocking: false,
              },
            }))
            return
          }
        }
      } catch {
        if (!appSettings.hasReplicateApiKey) {
          window.dispatchEvent(new CustomEvent('open-api-gateway', {
            detail: {
              requiredKeys: ['replicate'],
              title: 'Connect Replicate',
              description: 'Replicate is required for generating images when API generations are enabled.',
              blocking: false,
            },
          }))
          return
        }
      }
    }

    const numImages = settings.variations || 1

    setState(prev => ({
      ...prev,
      isGenerating: true,
      progress: 0,
      statusMessage: numImages > 1 ? `Generating ${numImages} images...` : 'Generating image...',
      videoUrl: null,
      videoPath: null,
      imageUrl: null,
      imageUrls: [],
      error: null,
    }))

    try {
      const backendUrl = await window.electronAPI.getBackendUrl()

      const dims = getImageDimensions(settings)
      const numSteps = settings.imageSteps || 4

      const response = await fetch(`${backendUrl}/api/queue/submit`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          type: 'image',
          model: appSettings.imageModel || 'z-image-turbo',
          params: {
            prompt,
            width: dims.width,
            height: dims.height,
            numSteps,
            numImages,
          },
        }),
      })

      if (!response.ok) {
        const errorText = await response.text()
        throw new Error(errorText || 'Failed to submit image generation job')
      }

      const result: { id: string; status: string } = await response.json()
      activeJobIdRef.current = result.id
      startPolling()
    } catch (error) {
      setState(prev => ({
        ...prev,
        isGenerating: false,
        error: error instanceof Error ? error.message : 'Unknown error',
      }))
    }
  }, [appSettings.hasReplicateApiKey, appSettings.imageModel, forceApiGenerations, refreshSettings, startPolling])

  const clearQueue = useCallback(async () => {
    try {
      const backendUrl = await window.electronAPI.getBackendUrl()
      const res = await fetch(`${backendUrl}/api/queue/clear`, { method: 'POST' })
      if (res.ok) {
        const data: { jobs: QueueJob[] } = await res.json()
        setState(prev => ({ ...prev, jobs: data.jobs }))
      }
    } catch {
      // Ignore errors
    }
  }, [])

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
      jobs: [],
      lastModel: null,
    })
  }, [])

  return {
    ...state,
    generate,
    generateImage,
    cancel,
    reset,
    clearQueue,
  }
}
