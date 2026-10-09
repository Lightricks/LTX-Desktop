import type { ApiErrorsOf } from './api-client'

export const IC_LORA_SOURCE_TOO_LARGE = 'IC_LORA_SOURCE_TOO_LARGE'
export const VIDEO_JOB_TOO_LARGE = 'VIDEO_JOB_TOO_LARGE'
export const LOCAL_GENERATION_UNSUPPORTED = 'LOCAL_GENERATION_UNSUPPORTED'
export const LTX_INVALID_API_KEY = 'LTX_INVALID_API_KEY'
export const LTX_API_KEY_MISSING = 'LTX_API_KEY_MISSING'

type ActionableLocalCode =
  | typeof IC_LORA_SOURCE_TOO_LARGE
  | typeof VIDEO_JOB_TOO_LARGE
  | typeof LOCAL_GENERATION_UNSUPPORTED

function isActionableLocalCode(code: string): code is ActionableLocalCode {
  return (
    code === IC_LORA_SOURCE_TOO_LARGE ||
    code === VIDEO_JOB_TOO_LARGE ||
    code === LOCAL_GENERATION_UNSUPPORTED
  )
}

export type LocalGenerationError = {
  status: 'default'
  error: {
    code: 'LOCAL_GENERATION_ERROR' | ActionableLocalCode
    message: string
  }
}

export type GenerationError = ApiErrorsOf<'generateVideo'> | ApiErrorsOf<'generateImage'> | LocalGenerationError

export function createLocalGenerationError(
  message: string,
  code: string = 'LOCAL_GENERATION_ERROR',
): LocalGenerationError {
  return {
    status: 'default',
    error: {
      code: isActionableLocalCode(code) ? code : 'LOCAL_GENERATION_ERROR',
      message,
    },
  }
}

export type ApiFailure = {
  code: string
  message: string
}

function isLtxApiKeyFailure(code: string): boolean {
  return code === LTX_INVALID_API_KEY || code === LTX_API_KEY_MISSING
}

/** Keep a rejected or missing LTX key as a typed error so the dialog can open Settings. */
export function generationErrorFromApiFailure(failure: ApiFailure): GenerationError {
  if (isLtxApiKeyFailure(failure.code)) {
    return { status: '4XX', error: failure }
  }
  return createLocalGenerationError(failure.message, failure.code)
}
