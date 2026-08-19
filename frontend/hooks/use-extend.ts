import { useCallback, useState } from 'react'
import { ApiClient } from '../lib/api-client'
import { canCancelLocalJob, withGenerationActive } from '../lib/generation-active'
import { logger } from '../lib/logger'
import { useAppSettings } from '../contexts/AppSettingsContext'
import type { RetakeExtendModel } from './use-retake'

export type ExtendDirection = 'start' | 'end'

// Seconds-to-add presets (matches LTX Studio). API allows 2–20s.
export const EXTEND_SECONDS = [4, 6, 8, 10, 12] as const
export const DEFAULT_EXTEND_SECONDS = 4

export interface ExtendSubmitParams {
  videoPath: string
  duration: number
  prompt: string
  mode: ExtendDirection
  resolution?: { width: number; height: number }
  model: RetakeExtendModel
}

export interface ExtendResult {
  videoPath: string
}

interface UseExtendState {
  isExtending: boolean
  canCancel: boolean
  extendStatus: string
  extendError: string | null
  result: ExtendResult | null
}

export function useExtend() {
  const { shouldVideoGenerateWithLtxApi, shouldImageGenerateWithFalApi } = useAppSettings()
  const [state, setState] = useState<UseExtendState>({
    isExtending: false,
    canCancel: false,
    extendStatus: '',
    extendError: null,
    result: null,
  })

  const submitExtend = useCallback(async (params: ExtendSubmitParams) => {
    if (!params.videoPath) return

    setState({
      isExtending: true,
      canCancel: canCancelLocalJob('video', shouldVideoGenerateWithLtxApi, shouldImageGenerateWithFalApi),
      extendStatus: 'Generating',
      extendError: null,
      result: null,
    })

    await withGenerationActive(async () => {
      const result = await ApiClient.extend({
        video_path: params.videoPath,
        duration: params.duration,
        prompt: params.prompt,
        mode: params.mode,
        resolution: params.resolution,
        model: params.model,
      })

      if (!result.ok) {
        logger.error(`Extend error: ${result.error.message}`)
        setState({ isExtending: false, canCancel: false, extendStatus: '', extendError: result.error.message, result: null })
        return
      }

      const payload = result.data

      if (payload.status === 'cancelled') {
        setState({ isExtending: false, canCancel: false, extendStatus: 'Cancelled', extendError: null, result: null })
        return
      }

      if ('video_path' in payload) {
        setState({
          isExtending: false,
          canCancel: false,
          extendStatus: 'Extend complete!',
          extendError: null,
          result: { videoPath: payload.video_path },
        })
        return
      }

      // A 200 with a remote payload and no local file is a legitimate success (the backend
      // completed the generation); there's just no local artifact to import. Don't surface
      // it as an error.
      logger.warn(`Extend completed with a remote payload and no local file: ${JSON.stringify(payload.result)}`)
      setState({
        isExtending: false,
        canCancel: false,
        extendStatus: 'Extend complete!',
        extendError: null,
        result: null,
      })
    })
  }, [shouldImageGenerateWithFalApi, shouldVideoGenerateWithLtxApi])

  const resetExtend = useCallback(() => {
    setState({ isExtending: false, canCancel: false, extendStatus: '', extendError: null, result: null })
  }, [])

  return {
    submitExtend,
    resetExtend,
    isExtending: state.isExtending,
    canCancel: state.canCancel,
    extendStatus: state.extendStatus,
    extendError: state.extendError,
    extendResult: state.result,
  }
}
