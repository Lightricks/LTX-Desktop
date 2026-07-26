import { backendFetch } from './backend'
import type { components, paths } from '../generated/backend-openapi'

type HttpMethod = 'get' | 'post' | 'put' | 'patch' | 'delete'

type OperationFor<
  TPath extends keyof paths,
  TMethod extends HttpMethod,
> = NonNullable<paths[TPath][TMethod]>

type ResponsesFor<
  TPath extends keyof paths,
  TMethod extends HttpMethod,
> = OperationFor<TPath, TMethod>['responses']

type JsonBodyOf<TResponse> = TResponse extends {
  content: infer TContent
}
  ? TContent extends { 'application/json': infer TJson }
    ? TJson
    : never
  : never

type JsonResponseFor<
  TPath extends keyof paths,
  TMethod extends HttpMethod,
> = OperationFor<TPath, TMethod> extends {
  responses: { 200: infer TResponse }
}
  ? JsonBodyOf<TResponse>
  : never

type JsonBodyFor<
  TPath extends keyof paths,
  TMethod extends HttpMethod,
> = OperationFor<TPath, TMethod> extends {
  requestBody?: { content: { 'application/json': infer TBody } }
}
  ? TBody
  : never

type QueryFor<
  TPath extends keyof paths,
  TMethod extends HttpMethod,
> = OperationFor<TPath, TMethod> extends {
  parameters: { query?: infer TQuery }
}
  ? TQuery
  : never

type HTTPErrorResponse = components["schemas"]["HTTPErrorResponse"]

type ExactErrorResponseFor<
  TPath extends keyof paths,
  TMethod extends HttpMethod,
  TStatus extends number,
> = TStatus extends keyof ResponsesFor<TPath, TMethod>
  ? JsonBodyOf<ResponsesFor<TPath, TMethod>[TStatus]>
  : never

type Fallback4xxErrorFor<
  TPath extends keyof paths,
  TMethod extends HttpMethod,
> = '4XX' extends keyof ResponsesFor<TPath, TMethod>
  ? JsonBodyOf<ResponsesFor<TPath, TMethod>['4XX']>
  : HTTPErrorResponse

type Fallback5xxErrorFor<
  TPath extends keyof paths,
  TMethod extends HttpMethod,
> = '5XX' extends keyof ResponsesFor<TPath, TMethod>
  ? JsonBodyOf<ResponsesFor<TPath, TMethod>['5XX']>
  : HTTPErrorResponse

type DefaultErrorFor<
  TPath extends keyof paths,
  TMethod extends HttpMethod,
> = 'default' extends keyof ResponsesFor<TPath, TMethod>
  ? JsonBodyOf<ResponsesFor<TPath, TMethod>['default']>
  : HTTPErrorResponse

type ExactErrorMembers<
  TPath extends keyof paths,
  TMethod extends HttpMethod,
  TExactStatuses extends readonly number[],
> = {
  [TStatus in TExactStatuses[number]]: {
    ok: false
    status: TStatus
    error: ExactErrorResponseFor<TPath, TMethod, TStatus>
  }
}[TExactStatuses[number]]

type FallbackErrorMembers<
  TPath extends keyof paths,
  TMethod extends HttpMethod,
> =
  | {
      ok: false
      status: '4XX'
      error: Fallback4xxErrorFor<TPath, TMethod>
    }
  | {
      ok: false
      status: '5XX'
      error: Fallback5xxErrorFor<TPath, TMethod>
    }
  | {
      ok: false
      status: 'default'
      error: DefaultErrorFor<TPath, TMethod>
    }

export type EndpointResult<
  TPath extends keyof paths,
  TMethod extends HttpMethod,
  TExactStatuses extends readonly number[] = [],
> =
  | {
      ok: true
      data: JsonResponseFor<TPath, TMethod>
    }
  | ExactErrorMembers<TPath, TMethod, TExactStatuses>
  | FallbackErrorMembers<TPath, TMethod>

type SyntheticErrorStatus = '4XX' | '5XX' | 'default'

export type ApiSuccess<TValue> = TValue extends { ok: true; data: infer TData }
  ? TData
  : never

export type ApiErrors<TValue> = TValue extends { ok: false; status: infer TStatus; error: infer TError }
  ? { status: TStatus; error: TError }
  : never

function buildQueryString(query: Record<string, unknown> | undefined): string {
  if (!query) return ''
  const params = new URLSearchParams()
  for (const [key, value] of Object.entries(query)) {
    if (value === null || value === undefined) continue
    params.set(key, String(value))
  }
  const serialized = params.toString()
  return serialized ? `?${serialized}` : ''
}

function buildJsonRequestInit(body: unknown, init?: RequestInit): RequestInit {
  const headers = new Headers(init?.headers)
  headers.set('Content-Type', 'application/json')
  return {
    ...init,
    headers,
    body: JSON.stringify(body),
  }
}

function buildSyntheticError(code: string, message: string): HTTPErrorResponse {
  return { code, message }
}

function resolveFallbackStatus(httpStatus: number): SyntheticErrorStatus {
  if (httpStatus >= 400 && httpStatus < 500) return '4XX'
  if (httpStatus >= 500 && httpStatus < 600) return '5XX'
  return 'default'
}

function resolveErrorStatus<TExactStatuses extends readonly number[]>(
  httpStatus: number,
  exactErrorStatuses: TExactStatuses,
): TExactStatuses[number] | SyntheticErrorStatus {
  if ((exactErrorStatuses as readonly number[]).includes(httpStatus)) {
    return httpStatus as TExactStatuses[number]
  }
  return resolveFallbackStatus(httpStatus)
}

function buildParsedErrorResult<
  TPath extends keyof paths,
  TMethod extends HttpMethod,
  TExactStatuses extends readonly number[],
>(
  status: TExactStatuses[number] | SyntheticErrorStatus,
  payload: unknown,
): EndpointResult<TPath, TMethod, TExactStatuses> {
  return {
    ok: false,
    status,
    error: payload as ExactErrorResponseFor<TPath, TMethod, TExactStatuses[number]>
      | Fallback4xxErrorFor<TPath, TMethod>
      | Fallback5xxErrorFor<TPath, TMethod>
      | DefaultErrorFor<TPath, TMethod>,
  } as EndpointResult<TPath, TMethod, TExactStatuses>
}

function buildSyntheticErrorResult<
  TPath extends keyof paths,
  TMethod extends HttpMethod,
  TExactStatuses extends readonly number[],
>(
  status: SyntheticErrorStatus,
  code: string,
  message: string,
): EndpointResult<TPath, TMethod, TExactStatuses> {
  return {
    ok: false,
    status,
    error: buildSyntheticError(code, message) as Fallback4xxErrorFor<TPath, TMethod>
      | Fallback5xxErrorFor<TPath, TMethod>
      | DefaultErrorFor<TPath, TMethod>,
  } as EndpointResult<TPath, TMethod, TExactStatuses>
}

async function requestEndpointResult<
  TPath extends keyof paths,
  TMethod extends HttpMethod,
  TExactStatuses extends readonly number[],
>(
  endpoint: TPath,
  method: TMethod,
  exactErrorStatuses: TExactStatuses,
  init?: RequestInit,
  requestPath?: string,
): Promise<EndpointResult<TPath, TMethod, TExactStatuses>> {
  const path = requestPath ?? String(endpoint)

  let response: Response
  try {
    response = await backendFetch(path, {
      method: method.toUpperCase(),
      ...init,
    })
  } catch (error) {
    return buildSyntheticErrorResult<TPath, TMethod, TExactStatuses>(
      'default',
      'NETWORK_ERROR',
      error instanceof Error ? error.message : 'Request failed before the server responded.',
    )
  }

  let text = ''
  try {
    text = await response.text()
  } catch (error) {
    return buildSyntheticErrorResult<TPath, TMethod, TExactStatuses>(
      resolveFallbackStatus(response.status),
      'RESPONSE_READ_FAILED',
      error instanceof Error ? error.message : 'Failed to read response body.',
    )
  }

  if (response.ok) {
    if (!text) {
      return buildSyntheticErrorResult<TPath, TMethod, TExactStatuses>(
        'default',
        'EMPTY_SUCCESS_RESPONSE',
        `${path} returned an empty response body.`,
      )
    }

    try {
      return {
        ok: true,
        data: JSON.parse(text) as JsonResponseFor<TPath, TMethod>,
      }
    } catch (error) {
      return buildSyntheticErrorResult<TPath, TMethod, TExactStatuses>(
        'default',
        'INVALID_SUCCESS_RESPONSE',
        error instanceof Error ? error.message : 'Server returned invalid JSON.',
      )
    }
  }

  if (!text) {
    return buildSyntheticErrorResult<TPath, TMethod, TExactStatuses>(
      resolveFallbackStatus(response.status),
      `HTTP_${response.status}`,
      `${response.status} ${response.statusText || 'Request failed'}`,
    )
  }

  try {
    const payload = JSON.parse(text) as unknown
    return buildParsedErrorResult<TPath, TMethod, TExactStatuses>(
      resolveErrorStatus(response.status, exactErrorStatuses),
      payload,
    )
  } catch {
    return buildSyntheticErrorResult<TPath, TMethod, TExactStatuses>(
      resolveFallbackStatus(response.status),
      `HTTP_${response.status}`,
      text,
    )
  }
}

export function makeEndpointClient<
  TPath extends keyof paths,
  TMethod extends HttpMethod,
  TExactStatuses extends readonly number[] = [],
>(
  endpoint: TPath,
  method: TMethod,
  config?: {
    exactErrorStatuses?: TExactStatuses
  },
) {
  const exactErrorStatuses = (config?.exactErrorStatuses ?? []) as TExactStatuses

  return (
    body?: JsonBodyFor<TPath, TMethod>,
    init?: RequestInit,
    requestPath?: string,
  ): Promise<EndpointResult<TPath, TMethod, TExactStatuses>> => {
    const requestInit = body === undefined
      ? init
      : buildJsonRequestInit(body, init)
    return requestEndpointResult(endpoint, method, exactErrorStatuses, requestInit, requestPath)
  }
}

export class ApiClient {
  static getHealth = makeEndpointClient('/health', 'get')

  static getModelDownloadProgress(
    query: QueryFor<'/api/models/download/progress', 'get'>,
  ): Promise<EndpointResult<'/api/models/download/progress', 'get'>> {
    const path = `/api/models/download/progress${buildQueryString(query as Record<string, unknown>)}`
    return requestEndpointResult('/api/models/download/progress', 'get', [] as const, undefined, path)
  }

  static listModels(
    query?: QueryFor<'/api/models', 'get'>,
  ): Promise<EndpointResult<'/api/models', 'get'>> {
    const path = `/api/models${buildQueryString(query as Record<string, unknown>)}`
    return requestEndpointResult('/api/models', 'get', [] as const, undefined, path)
  }

  
  static getLtxRecommendation = makeEndpointClient('/api/models/ltx-recommendation', 'get')

  static getImgGenRecommendation = makeEndpointClient('/api/models/img-gen-recommendation', 'get')

  static getLtxIcLoraRecommendation = makeEndpointClient('/api/models/ltx-ic-lora-recommendation', 'get')

  static getTextEncoderRecommendation = makeEndpointClient('/api/models/text-encoder-recommendation', 'get')

  static describeCheckpoints = makeEndpointClient('/api/models/describe', 'post')

  static getActiveDownload = makeEndpointClient('/api/models/download/active', 'get')

  static getLtxVersions = makeEndpointClient('/api/models/ltx-versions', 'get')

  static setActiveLtxModel = makeEndpointClient('/api/models/active-ltx-model', 'post')

  static startModelDownload = makeEndpointClient('/api/models/download', 'post')

  static deleteModels = makeEndpointClient('/api/models/delete', 'delete')

  static getRuntimePolicy = makeEndpointClient('/api/runtime-policy', 'get')

  static getRuntimeTelemetry = makeEndpointClient('/api/runtime-telemetry', 'get')

  static getGpuInfo = makeEndpointClient('/api/gpu-info', 'get')

  static getSettings = makeEndpointClient('/api/settings', 'get')

  static updateSettings = makeEndpointClient('/api/settings', 'post')

  static suggestGapPrompt = makeEndpointClient('/api/suggest-gap-prompt', 'post', {
    exactErrorStatuses: [401, 403] as const,
  })

  static generateVideo = makeEndpointClient('/api/generate', 'post', {
    exactErrorStatuses: [402] as const,
  })

  static getGenerateVideoModelSpecs = makeEndpointClient('/api/generate/models-specs', 'get')

  static cancelGeneration = makeEndpointClient('/api/generate/cancel', 'post')

  static getGenerationProgress = makeEndpointClient('/api/generation/progress', 'get')

  static generateImage = makeEndpointClient('/api/generate-image', 'post')

  static enhancePrompt = makeEndpointClient('/api/enhance-prompt', 'post', {
    exactErrorStatuses: [404, 409] as const,
  })

  static retake = makeEndpointClient('/api/retake', 'post')

  static extend = makeEndpointClient('/api/extend', 'post')

  static startHuggingFaceLogin = makeEndpointClient('/api/auth/huggingface/login', 'post')

  static getHuggingFaceAuthStatus = makeEndpointClient('/api/auth/huggingface/status', 'get')

  static huggingFaceLogout = makeEndpointClient('/api/auth/huggingface/logout', 'post')

  static checkModelAccess = makeEndpointClient('/api/models/check-access', 'post')

  static generateIcLora = makeEndpointClient('/api/ic-lora/generate', 'post')

  static extractIcLoraConditioning = makeEndpointClient('/api/ic-lora/extract-conditioning', 'post')

  static listIcLoras = makeEndpointClient('/api/ic-loras', 'get')

  static startIcLoraDownload = makeEndpointClient('/api/ic-loras/download', 'post')

  static getIcLoraDownloadProgress(
    query: QueryFor<'/api/ic-loras/download/progress', 'get'>,
  ): Promise<EndpointResult<'/api/ic-loras/download/progress', 'get'>> {
    const path = `/api/ic-loras/download/progress${buildQueryString(query as Record<string, unknown>)}`
    return requestEndpointResult('/api/ic-loras/download/progress', 'get', [] as const, undefined, path)
  }

  static listLoras = makeEndpointClient('/api/loras', 'get')

  static startLoraDownload = makeEndpointClient('/api/loras/download', 'post')

  static getLoraDownloadProgress(
    query: QueryFor<'/api/loras/download/progress', 'get'>,
  ): Promise<EndpointResult<'/api/loras/download/progress', 'get'>> {
    const path = `/api/loras/download/progress${buildQueryString(query as Record<string, unknown>)}`
    return requestEndpointResult('/api/loras/download/progress', 'get', [] as const, undefined, path)
  }
}

type ApiClientMethodName = keyof typeof ApiClient

export type ApiRequestBodyOf<TMethod extends ApiClientMethodName> = (typeof ApiClient)[TMethod] extends (
  body?: infer TBody,
  ...args: any[]
) => Promise<any>
  ? TBody
  : never

export type ApiSuccessOf<TMethod extends ApiClientMethodName> = (typeof ApiClient)[TMethod] extends (...args: any[]) => Promise<any>
  ? ApiSuccess<Awaited<ReturnType<(typeof ApiClient)[TMethod]>>>
  : never

export type ApiErrorsOf<TMethod extends ApiClientMethodName> = (typeof ApiClient)[TMethod] extends (...args: any[]) => Promise<any>
  ? ApiErrors<Awaited<ReturnType<(typeof ApiClient)[TMethod]>>>
  : never
