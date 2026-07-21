import { useCallback, useState } from 'react'
import { ApiClient, type ApiRequestBodyOf } from '../lib/api-client'
import { logger } from '../lib/logger'
import { materializeBackendOutput, prepareBackendMedia } from '../lib/backend-media'

export type RetakeMode = 'replace_audio_and_video' | 'replace_video' | 'replace_audio'
type RetakeBody = NonNullable<ApiRequestBodyOf<'retake'>>

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

    let preparedVideo
    try {
      setState(prev => ({ ...prev, retakeStatus: 'Uploading source video' }))
      preparedVideo = await prepareBackendMedia(params.videoPath, 'video')
    } catch (error) {
      setState({
        isRetaking: false,
        retakeStatus: '',
        retakeError: error instanceof Error ? error.message : String(error),
        result: null,
      })
      return
    }
    const request: RetakeBody = {
      start_time: params.startTime,
      duration: params.duration,
      prompt: params.prompt,
      mode: params.mode,
    }
    if (preparedVideo?.path) request.video_path = preparedVideo.path
    if (preparedVideo?.mediaId) request.video_media_id = preparedVideo.mediaId
    setState(prev => ({ ...prev, retakeStatus: 'Generating' }))

    const result = await ApiClient.retake(request)

    if (!result.ok) {
      logger.error(`Retake error: ${result.error.message}`)
      setState({
        isRetaking: false,
        retakeStatus: '',
        retakeError: result.error.message,
        result: null,
      })
      return
    }

    const payload = result.data

    if (payload.status === 'cancelled') {
      setState({
        isRetaking: false,
        retakeStatus: 'Cancelled',
        retakeError: null,
        result: null,
      })
      return
    }

    if (payload.status === 'complete' && 'video_path' in payload) {
      try {
        const outputPath = await materializeBackendOutput(payload.video_path, payload.artifact)
        setState({
          isRetaking: false,
          retakeStatus: 'Retake complete!',
          retakeError: null,
          result: {
            videoPath: outputPath,
          },
        })
        return
      } catch (error) {
        const errorMsg = error instanceof Error ? error.message : String(error)
        logger.error(`Retake output materialization failed: ${errorMsg}`)
        setState({
          isRetaking: false,
          retakeStatus: '',
          retakeError: errorMsg,
          result: null,
        })
        return
      }
    }

    const errorMsg = 'Retake completed without a downloadable video result'
    logger.error(`${errorMsg}: ${JSON.stringify(payload)}`)
    setState({
      isRetaking: false,
      retakeStatus: '',
      retakeError: errorMsg,
      result: null,
    })

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
