import { useCallback, useState } from 'react'
import type { components } from '../generated/backend-openapi'
import { ApiClient } from '../lib/api-client'
import { canCancelLocalJob, withGenerationActive } from '../lib/generation-active'
import { logger } from '../lib/logger'
import { useAppSettings } from '../contexts/AppSettingsContext'

export type RetakeMode = 'replace_audio_and_video' | 'replace_video' | 'replace_audio'

// ltxv-api /v1/retake and /v2/extend accept ltx-2-pro / ltx-2-3-pro.
// Desktop maps those to pipeline "pro".
export type RetakeExtendModel = components['schemas']['RetakeRequest']['model']

// Runtime options for the retake/extend MODEL dropdown. Checked against the OpenAPI union
// so a schema change that adds/removes a value fails typecheck until this list is updated.
export const RETAKE_EXTEND_MODELS = ['pro'] as const satisfies ReadonlyArray<RetakeExtendModel>

/** Map a persisted video pipeline id onto the nearest retake/extend model. */
export function retakeExtendModelFromPipeline(
  _model: string | undefined | null,
): RetakeExtendModel {
  return 'pro'
}

export interface RetakeSubmitParams {
  videoPath: string
  startTime: number
  duration: number
  prompt: string
  mode: RetakeMode
  resolution?: { width: number; height: number }
  model: RetakeExtendModel
}

export interface RetakeResult {
  videoPath: string
}

interface UseRetakeState {
  isRetaking: boolean
  canCancel: boolean
  retakeStatus: string
  retakeError: string | null
  result: RetakeResult | null
}

export function useRetake() {
  const { shouldVideoGenerateWithLtxApi, shouldImageGenerateWithFalApi } = useAppSettings()
  const [state, setState] = useState<UseRetakeState>({
    isRetaking: false,
    canCancel: false,
    retakeStatus: '',
    retakeError: null,
    result: null,
  })

  const submitRetake = useCallback(async (params: RetakeSubmitParams) => {
    if (!params.videoPath) return

    setState({
      isRetaking: true,
      canCancel: canCancelLocalJob('video', shouldVideoGenerateWithLtxApi, shouldImageGenerateWithFalApi),
      retakeStatus: 'Generating',
      retakeError: null,
      result: null,
    })

    await withGenerationActive(async () => {
      const result = await ApiClient.retake({
        video_path: params.videoPath,
        start_time: params.startTime,
        duration: params.duration,
        prompt: params.prompt,
        mode: params.mode,
        resolution: params.resolution,
        model: params.model,
      })

      if (!result.ok) {
        logger.error(`Retake error: ${result.error.message}`)
        setState({
          isRetaking: false,
          canCancel: false,
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
          canCancel: false,
          retakeStatus: 'Cancelled',
          retakeError: null,
          result: null,
        })
        return
      }

      if ('video_path' in payload) {
        setState({
          isRetaking: false,
          canCancel: false,
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
        canCancel: false,
        retakeStatus: '',
        retakeError: errorMsg,
        result: null,
      })
    })
  }, [shouldImageGenerateWithFalApi, shouldVideoGenerateWithLtxApi])

  const resetRetake = useCallback(() => {
    setState({
      isRetaking: false,
      canCancel: false,
      retakeStatus: '',
      retakeError: null,
      result: null,
    })
  }, [])

  return {
    submitRetake,
    resetRetake,
    isRetaking: state.isRetaking,
    canCancel: state.canCancel,
    retakeStatus: state.retakeStatus,
    retakeError: state.retakeError,
    retakeResult: state.result,
  }
}
