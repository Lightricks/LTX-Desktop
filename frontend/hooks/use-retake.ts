import { useCallback, useState } from 'react'
import { backendFetch } from '../lib/backend'
import { logger } from '../lib/logger'

export type RetakeMode = 'replace_audio_and_video' | 'replace_video' | 'replace_audio'

export interface RetakeSubmitParams {
  videoPath: string
  startTime: number
  duration: number
  prompt: string
  mode: RetakeMode
}

export interface RetakeResult {
  videoPath: string
  videoUrl: string
}

interface UseRetakeState {
  isRetaking: boolean
  retakeStatus: string
  retakeError: string | null
  result: RetakeResult | null
}

export function useRetake() {
  const [state, setState] = useState<UseRetakeState>({
    isRetaking: false,
    retakeStatus: '',
    retakeError: null,
    result: null,
  })

  const submitRetake = useCallback(async (params: RetakeSubmitParams) => {
    if (!params.videoPath) return

    setState({
      isRetaking: true,
      retakeStatus: 'Generating',
      retakeError: null,
      result: null,
    })

    try {
      const response = await backendFetch('/api/retake', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          video_path: params.videoPath,
          start_time: params.startTime,
          duration: params.duration,
          prompt: params.prompt,
          mode: params.mode,
        }),
      })

      const data = await response.json()

      if (response.ok && data.status === 'complete' && data.video_path) {
        const pathNormalized = data.video_path.replace(/\\/g, '/')
        const videoUrl = pathNormalized.startsWith('/') ? `file://${pathNormalized}` : `file:///${pathNormalized}`

        setState({
          isRetaking: false,
          retakeStatus: 'Retake complete!',
          retakeError: null,
          result: {
            videoPath: data.video_path,
            videoUrl,
          },
        })
        return
      }

      const errorMsg = data.error || 'Unknown error'
      setState({
        isRetaking: false,
        retakeStatus: '',
        retakeError: errorMsg,
        result: null,
      })
      logger.error(`Retake failed: ${errorMsg}`)
    } catch (error) {
      const message = (error as Error).message || 'Unknown error'
      logger.error(`Retake error: ${message}`)
      setState({
        isRetaking: false,
        retakeStatus: '',
        retakeError: message,
        result: null,
      })
    }
  }, [])

  const resetRetake = useCallback(() => {
    setState({
      isRetaking: false,
      retakeStatus: '',
      retakeError: null,
      result: null,
    })
  }, [])

  return {
    submitRetake,
    resetRetake,
    isRetaking: state.isRetaking,
    retakeStatus: state.retakeStatus,
    retakeError: state.retakeError,
    retakeResult: state.result,
  }
}
