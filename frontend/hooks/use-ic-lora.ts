import { useCallback, useState } from 'react'
import { ApiClient } from '../lib/api-client'
import { logger } from '../lib/logger'

export type IcLoraConditioningType = 'canny' | 'depth'

export interface IcLoraSubmitParams {
  videoPath: string
  conditioningType: IcLoraConditioningType
  conditioningStrength: number
  prompt: string
}

export interface IcLoraResult {
  videoPath: string
}

interface UseIcLoraState {
  isGenerating: boolean
  status: string
  error: string | null
  result: IcLoraResult | null
}

export function useIcLora() {
  const [state, setState] = useState<UseIcLoraState>({
    isGenerating: false,
    status: '',
    error: null,
    result: null,
  })

  const submitIcLora = useCallback(async (params: IcLoraSubmitParams) => {
    if (!params.videoPath || !params.prompt.trim()) return

    setState({
      isGenerating: true,
      status: 'Generating',
      error: null,
      result: null,
    })

    try {
      const payload = await ApiClient.generateIcLora({
        video_path: params.videoPath,
        conditioning_type: params.conditioningType,
        conditioning_strength: params.conditioningStrength,
        prompt: params.prompt,
      } as unknown as Parameters<typeof ApiClient.generateIcLora>[0])
      if (payload.status === 'cancelled') {
        setState({
          isGenerating: false,
          status: 'Cancelled',
          error: null,
          result: null,
        })
        return
      }

      if (payload.status === 'complete') {
        setState({
          isGenerating: false,
          status: 'Generation complete!',
          error: null,
          result: {
            videoPath: payload.video_path,
          },
        })
        return
      }
    } catch (error) {
      const message = (error as Error).message || 'Unknown error'
      logger.error(`IC-LoRA error: ${message}`)
      setState({
        isGenerating: false,
        status: '',
        error: message,
        result: null,
      })
    }
  }, [])

  const reset = useCallback(() => {
    setState({
      isGenerating: false,
      status: '',
      error: null,
      result: null,
    })
  }, [])

  return {
    submitIcLora,
    resetIcLora: reset,
    isIcLoraGenerating: state.isGenerating,
    icLoraStatus: state.status,
    icLoraError: state.error,
    icLoraResult: state.result,
  }
}
