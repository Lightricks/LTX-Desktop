import { useCallback, useState } from 'react'
import { ApiClient, type ApiRequestBodyOf } from '../lib/api-client'
import { withGenerationActive } from '../lib/generation-active'
import { logger } from '../lib/logger'

export type IcLoraConditioningType = 'canny' | 'depth' | 'custom'
export type IcLoraAudioMode = 'source' | 'generated' | 'off'

export interface IcLoraSubmitParams {
  videoPath: string
  conditioningType: IcLoraConditioningType
  conditioningStrength: number
  prompt: string
  // "custom": the user's own IC-LoRA weights + a pre-rendered control video.
  customLoraRef?: string
  controlVideoPath?: string
  // Skip Stage 2 refine — transformation IC-LoRAs need this; default off.
  skipStage2?: boolean
  // Keep the IC-LoRA active during Stage 2 refine (only when Stage 2 runs). Default off.
  // On: the transformation survives refinement and the output lands at the chosen resolution.
  useLoraInStage2?: boolean
  // Target output resolution for the useLoraInStage2 path (undefined = source). Backend
  // snaps it down to a valid size and never upscales above source.
  resolution?: { width: number; height: number }
  // Stage-1-only canvas multiplier: 2.0 = native, 1.0 = half. Only used when skipStage2.
  resolutionFactor?: number
  // Soundtrack: source (input clip) / generated (prompt) / off.
  audioMode?: IcLoraAudioMode
  // LoRA adapter merge weight (load-time). Default 1.0.
  loraStrength?: number
  // Override control-video fps (decimate to fewer frames). undefined = source fps.
  fpsOverride?: number
  // IC-LoRA mode: catalog IC-LoRA id + the user's input media path + output duration.
  // When icLoraId is set, the backend resolves catalog weights and builds the
  // control video via the IC-LoRA's preprocessing pipeline.
  icLoraId?: string
  // Catalog download.variants[].id when the entry has multiple checkpoints.
  variantId?: string
  inputPath?: string
  // Values for the IC-LoRA's declared controls, keyed by control id (e.g. duration).
  // Sent verbatim; the backend validates + maps them to behaviour.
  controlValues?: Record<string, number | string>
  // Outpainting (position_canvas control): per-edge pixels to add around the source.
  outpaintPads?: { left: number; right: number; top: number; bottom: number }
  // When true, an empty prompt is allowed (set from the IC-LoRA's allows_empty_prompt).
  allowEmptyPrompt?: boolean
  // Optional reference image for catalog IC-LoRAs that opt in (allows_reference_image).
  // Sent as image conditioning at frame 0, strength 1.0.
  referenceImagePath?: string
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

type GenerateIcLoraBody = ApiRequestBodyOf<'generateIcLora'>

export function useIcLora() {
  const [state, setState] = useState<UseIcLoraState>({
    isGenerating: false,
    status: '',
    error: null,
    result: null,
  })

  const submitIcLora = useCallback(async (params: IcLoraSubmitParams) => {
    if (!params.prompt.trim() && !params.allowEmptyPrompt) return
    // IC-LoRA mode needs inputPath; canny/depth/custom need videoPath.
    if (!params.icLoraId && !params.videoPath) return
    if (params.icLoraId && !params.inputPath) return

    setState({
      isGenerating: true,
      status: 'Generating',
      error: null,
      result: null,
    })

    await withGenerationActive(async () => {
      const result = await ApiClient.generateIcLora({
        video_path: params.videoPath,
        conditioning_type: params.conditioningType,
        conditioning_strength: params.conditioningStrength,
        prompt: params.prompt,
        custom_lora_ref: params.customLoraRef,
        control_video_path: params.controlVideoPath,
        skip_stage_2: params.skipStage2,
        use_lora_in_stage_2: params.useLoraInStage2,
        resolution: params.resolution,
        resolution_factor: params.resolutionFactor,
        audio_mode: params.audioMode,
        lora_strength: params.loraStrength,
        fps_override: params.fpsOverride,
        ic_lora_id: params.icLoraId,
        variant_id: params.variantId,
        input_path: params.inputPath,
        control_values: params.controlValues,
        outpaint_pads: params.outpaintPads,
        images: params.referenceImagePath
          ? [{ path: params.referenceImagePath, frame: 0, strength: 1.0 }]
          : [],
      } as GenerateIcLoraBody)
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
    })
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
