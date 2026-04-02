import { useCallback, useState } from 'react'
import { ApiClient } from '../lib/api-client'
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
      const payload = await ApiClient.retake({
        video_path: params.videoPath,
        start_time: params.startTime,
        duration: params.duration,
        prompt: params.prompt,
        mode: params.mode,
      })

      if (payload.status === 'cancelled') {
        setState({
          isRetaking: false,
          retakeStatus: 'Cancelled',
          retakeError: null,
          result: null,
        })
        return
      }

      if ('video_path' in payload) {
        setState({
          isRetaking: false,
          retakeStatus: 'Retake complete!',
          retakeError: null,
          result: {
            videoPath: payload.video_path,
          },
        })
        return
      }

      logger.error(`Retake completed without local video payload: ${JSON.stringify(payload.result)}`)
      const errorMsg = 'Retake completed but no local video file was returned'
      setState({
        isRetaking: false,
        retakeStatus: '',
        retakeError: errorMsg,
        result: null,
      })
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
