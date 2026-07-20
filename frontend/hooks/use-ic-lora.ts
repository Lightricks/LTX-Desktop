import { useCallback, useState } from 'react'
import { ApiClient, type ApiRequestBodyOf } from '../lib/api-client'
import { logger } from '../lib/logger'
import { materializeBackendOutput, prepareBackendMedia } from '../lib/backend-media'

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

type GenerateIcLoraBody = NonNullable<ApiRequestBodyOf<'generateIcLora'>>

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

    let preparedVideo
    try {
      setState(prev => ({ ...prev, status: 'Uploading source video' }))
      preparedVideo = await prepareBackendMedia(params.videoPath, 'video')
    } catch (error) {
      setState({
        isGenerating: false,
        status: '',
        error: error instanceof Error ? error.message : String(error),
        result: null,
      })
      return
    }
    const request: GenerateIcLoraBody = {
      conditioning_type: params.conditioningType,
      conditioning_strength: params.conditioningStrength,
      prompt: params.prompt,
      num_inference_steps: 30,
      cfg_guidance_scale: 1,
      negative_prompt: '',
    }
    if (preparedVideo?.path) request.video_path = preparedVideo.path
    if (preparedVideo?.mediaId) request.video_media_id = preparedVideo.mediaId
    setState(prev => ({ ...prev, status: 'Generating' }))

    const result = await ApiClient.generateIcLora(request)
    if (!result.ok) {
      logger.error(`IC-LoRA error: ${result.error.message}`)
      setState({
        isGenerating: false,
        status: '',
        error: result.error.message,
        result: null,
      })
      return
    }

    const payload = result.data
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
      let outputPath: string
      try {
        outputPath = await materializeBackendOutput(payload.video_path, payload.artifact)
      } catch (error) {
        setState({
          isGenerating: false,
          status: '',
          error: error instanceof Error ? error.message : String(error),
          result: null,
        })
        return
      }
      setState({
        isGenerating: false,
        status: 'Generation complete!',
        error: null,
        result: {
          videoPath: outputPath,
        },
      })
      return
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
