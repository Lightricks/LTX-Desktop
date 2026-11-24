import { useCallback, useState } from 'react'
import type { components } from '../generated/backend-openapi'
import { ApiClient } from '../lib/api-client'
import { canCancelLocalJob, withGenerationActive } from '../lib/generation-active'
import type { ApiFailure } from '../lib/generation-errors'
import { logger } from '../lib/logger'
import { useAppSettings } from '../contexts/AppSettingsContext'

export type RetakeMode = 'replace_audio_and_video' | 'replace_video' | 'replace_audio'

// ltxv-api /v2/retake and /v2/extend accept ltx-2-pro / ltx-2-3-pro.
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
  promptProvenance?: 'typed' | 'enhanced'
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
  retakeError: ApiFailure | null
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
        prompt_provenance: params.promptProvenance ?? 'typed',
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
          retakeError: { code: result.error.code, message: result.error.message },
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
      setState({
        isRetaking: false,
        canCancel: false,
        retakeStatus: '',
        retakeError: {
          code: 'RETAKE_NO_LOCAL_VIDEO',
          message: 'Retake completed but no local video file was returned',
        },
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
