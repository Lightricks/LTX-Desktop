import { backendFetch, type BackendFetch } from './backend.ts'
import type { components, paths } from '../generated/backend-openapi.ts'
import type { ExploreAsset, ExploreAssetListResponse } from './explore-contract.ts'

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

type StandardHttpErrorResult =
  | {
      ok: false
      status: '4XX'
      error: HTTPErrorResponse
    }
  | {
      ok: false
      status: '5XX'
      error: HTTPErrorResponse
    }
  | {
      ok: false
      status: 'default'
      error: HTTPErrorResponse
    }

type ExactParsedErrorMembers<TExactStatuses extends readonly number[]> = {
  [TStatus in TExactStatuses[number]]: {
    ok: false
    status: TStatus
    error: HTTPErrorResponse
  }
}[TExactStatuses[number]]

type ParsedJsonResult<TData, TExactStatuses extends readonly number[] = []> =
  | {
      ok: true
      data: TData
    }
  | ExactParsedErrorMembers<TExactStatuses>
  | StandardHttpErrorResult

/** Remote-only multipart `POST /api/assets/upload`. Success is the path-optional Explore Asset. */
export type RemoteUploadAssetResult = ParsedJsonResult<ExploreAsset>

/** Shared list payload: Desktop path fields optional; Remote may set `has_thumbnail`. */
export type ExploreListAssetsResult = ParsedJsonResult<ExploreAssetListResponse>

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

function buildParsedErrorResult<TExactStatuses extends readonly number[]>(
  status: TExactStatuses[number] | SyntheticErrorStatus,
  payload: unknown,
): Exclude<ParsedJsonResult<never, TExactStatuses>, { ok: true }> {
  return {
    ok: false,
    status,
    error: payload as HTTPErrorResponse,
  } as Exclude<ParsedJsonResult<never, TExactStatuses>, { ok: true }>
}

function buildSyntheticErrorResult(
  status: SyntheticErrorStatus,
  code: string,
  message: string,
): StandardHttpErrorResult {
  return {
    ok: false,
    status,
    error: buildSyntheticError(code, message),
  }
}

// Shared JSON parser for OpenAPI endpoints and the remote-only multipart upload
// route, which is intentionally absent from the desktop schema.
async function requestParsedJson<
  TData,
  TExactStatuses extends readonly number[],
>(
  fetchImpl: BackendFetch,
  path: string,
  method: string,
  exactErrorStatuses: TExactStatuses,
  init?: RequestInit,
): Promise<ParsedJsonResult<TData, TExactStatuses>> {
  let response: Response
  try {
    response = await fetchImpl(path, {
      method: method.toUpperCase(),
      ...init,
    })
  } catch (error) {
    return buildSyntheticErrorResult(
      'default',
      'NETWORK_ERROR',
      error instanceof Error ? error.message : 'Request failed before the server responded.',
    )
  }

  let text = ''
  try {
    text = await response.text()
  } catch (error) {
    return buildSyntheticErrorResult(
      resolveFallbackStatus(response.status),
      'RESPONSE_READ_FAILED',
      error instanceof Error ? error.message : 'Failed to read response body.',
    )
  }

  if (response.ok) {
    if (!text) {
      return buildSyntheticErrorResult(
        'default',
        'EMPTY_SUCCESS_RESPONSE',
        `${path} returned an empty response body.`,
      )
    }

    try {
      // Untyped JSON boundary: callers pin TData (ExploreAsset for remote upload, OpenAPI body otherwise).
      return {
        ok: true,
        data: JSON.parse(text) as TData,
      }
    } catch (error) {
      return buildSyntheticErrorResult(
        'default',
        'INVALID_SUCCESS_RESPONSE',
        error instanceof Error ? error.message : 'Server returned invalid JSON.',
      )
    }
  }

  if (!text) {
    return buildSyntheticErrorResult(
      resolveFallbackStatus(response.status),
      `HTTP_${response.status}`,
      `${response.status} ${response.statusText || 'Request failed'}`,
    )
  }

  try {
    const payload: unknown = JSON.parse(text)
    return buildParsedErrorResult(
      resolveErrorStatus(response.status, exactErrorStatuses),
      payload,
    )
  } catch {
    return buildSyntheticErrorResult(
      resolveFallbackStatus(response.status),
      `HTTP_${response.status}`,
      text,
    )
  }
}

function requestEndpointResult<
  TPath extends keyof paths,
  TMethod extends HttpMethod,
  TExactStatuses extends readonly number[],
>(
  fetchImpl: BackendFetch,
  endpoint: TPath,
  method: TMethod,
  exactErrorStatuses: TExactStatuses,
  init?: RequestInit,
  requestPath?: string,
): Promise<EndpointResult<TPath, TMethod, TExactStatuses>> {
  return requestParsedJson<JsonResponseFor<TPath, TMethod>, TExactStatuses>(
    fetchImpl,
    requestPath ?? String(endpoint),
    method,
    exactErrorStatuses,
    init,
  ) as Promise<EndpointResult<TPath, TMethod, TExactStatuses>>
}

function makeEndpointClient<
  TPath extends keyof paths,
  TMethod extends HttpMethod,
  TExactStatuses extends readonly number[] = [],
>(
  fetchImpl: BackendFetch,
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
    return requestEndpointResult(fetchImpl, endpoint, method, exactErrorStatuses, requestInit, requestPath)
  }
}

export function createApiClient(fetchImpl: BackendFetch) {
  return {
    getHealth: makeEndpointClient(fetchImpl, '/health', 'get'),

    getModelDownloadProgress(
      query: QueryFor<'/api/models/download/progress', 'get'>,
    ): Promise<EndpointResult<'/api/models/download/progress', 'get'>> {
      const path = `/api/models/download/progress${buildQueryString(query as Record<string, unknown>)}`
      return requestEndpointResult(fetchImpl, '/api/models/download/progress', 'get', [] as const, undefined, path)
    },

    listModels(
      query?: QueryFor<'/api/models', 'get'>,
    ): Promise<EndpointResult<'/api/models', 'get'>> {
      const path = `/api/models${buildQueryString(query as Record<string, unknown>)}`
      return requestEndpointResult(fetchImpl, '/api/models', 'get', [] as const, undefined, path)
    },

    getLtxRecommendation(
      query?: QueryFor<'/api/models/ltx-recommendation', 'get'>,
    ): Promise<EndpointResult<'/api/models/ltx-recommendation', 'get'>> {
      const path = `/api/models/ltx-recommendation${buildQueryString(query as Record<string, unknown>)}`
      return requestEndpointResult(fetchImpl, '/api/models/ltx-recommendation', 'get', [] as const, undefined, path)
    },

    getImgGenRecommendation(
      query?: QueryFor<'/api/models/img-gen-recommendation', 'get'>,
    ): Promise<EndpointResult<'/api/models/img-gen-recommendation', 'get'>> {
      const path = `/api/models/img-gen-recommendation${buildQueryString(query as Record<string, unknown>)}`
      return requestEndpointResult(
        fetchImpl,
        '/api/models/img-gen-recommendation',
        'get',
        [] as const,
        undefined,
        path,
      )
    },

    getLtxIcLoraRecommendation: makeEndpointClient(fetchImpl, '/api/models/ltx-ic-lora-recommendation', 'get'),

    getTextEncoderRecommendation: makeEndpointClient(fetchImpl, '/api/models/text-encoder-recommendation', 'get'),

    describeCheckpoints: makeEndpointClient(fetchImpl, '/api/models/describe', 'post'),

    getActiveDownload: makeEndpointClient(fetchImpl, '/api/models/download/active', 'get'),

    getLtxVersions: makeEndpointClient(fetchImpl, '/api/models/ltx-versions', 'get'),

    setActiveLtxModel: makeEndpointClient(fetchImpl, '/api/models/active-ltx-model', 'post'),

    startModelDownload: makeEndpointClient(fetchImpl, '/api/models/download', 'post'),

    deleteModels: makeEndpointClient(fetchImpl, '/api/models/delete', 'delete'),

    getRuntimePolicy: makeEndpointClient(fetchImpl, '/api/runtime-policy', 'get'),

    getFeatureFlags: makeEndpointClient(fetchImpl, '/api/feature-flags', 'get'),

    updateFeatureFlags: makeEndpointClient(fetchImpl, '/api/feature-flags', 'patch'),

    getGpuInfo: makeEndpointClient(fetchImpl, '/api/gpu-info', 'get'),

    getSettings: makeEndpointClient(fetchImpl, '/api/settings', 'get'),

    getPromptEnhancer: makeEndpointClient(fetchImpl, '/api/prompt-enhancer', 'get'),

    listGeminiModels: makeEndpointClient(fetchImpl, '/api/settings/gemini-models', 'get'),

    updateSettings: makeEndpointClient(fetchImpl, '/api/settings', 'post'),

    getRemoteStatus: makeEndpointClient(fetchImpl, '/api/remote/status', 'get'),

    listRemoteDevices: makeEndpointClient(fetchImpl, '/api/remote/devices', 'get'),

    revokeRemoteDevice(
      deviceId: string,
    ): Promise<EndpointResult<'/api/remote/devices/{device_id}/revoke', 'post', [404]>> {
      return requestEndpointResult(
        fetchImpl,
        '/api/remote/devices/{device_id}/revoke',
        'post',
        [404] as const,
        undefined,
        `/api/remote/devices/${encodeURIComponent(deviceId)}/revoke`,
      )
    },

    suggestGapPrompt: makeEndpointClient(fetchImpl, '/api/suggest-gap-prompt', 'post', {
      exactErrorStatuses: [401, 403] as const,
    }),

    generateVideo: makeEndpointClient(fetchImpl, '/api/generate', 'post', {
      exactErrorStatuses: [402] as const,
    }),

    getGenerateVideoModelSpecs: makeEndpointClient(fetchImpl, '/api/generate/models-specs', 'get'),

    cancelGeneration: makeEndpointClient(fetchImpl, '/api/generate/cancel', 'post'),

    getGenerationProgress: makeEndpointClient(fetchImpl, '/api/generation/progress', 'get'),

    generateImage: makeEndpointClient(fetchImpl, '/api/generate-image', 'post'),

    enhancePrompt: makeEndpointClient(fetchImpl, '/api/enhance-prompt', 'post', {
      exactErrorStatuses: [404, 409] as const,
    }),

    retake: makeEndpointClient(fetchImpl, '/api/retake', 'post'),

    extend: makeEndpointClient(fetchImpl, '/api/extend', 'post'),

    startHuggingFaceLogin: makeEndpointClient(fetchImpl, '/api/auth/huggingface/login', 'post'),

    getHuggingFaceAuthStatus: makeEndpointClient(fetchImpl, '/api/auth/huggingface/status', 'get'),

    huggingFaceLogout: makeEndpointClient(fetchImpl, '/api/auth/huggingface/logout', 'post'),

    checkModelAccess: makeEndpointClient(fetchImpl, '/api/models/check-access', 'post'),

    generateIcLora: makeEndpointClient(fetchImpl, '/api/ic-lora/generate', 'post'),

    extractIcLoraConditioning: makeEndpointClient(fetchImpl, '/api/ic-lora/extract-conditioning', 'post'),

    listIcLoras(
      query?: QueryFor<'/api/ic-loras', 'get'>,
    ): Promise<EndpointResult<'/api/ic-loras', 'get'>> {
      const path = `/api/ic-loras${buildQueryString(query as Record<string, unknown>)}`
      return requestEndpointResult(fetchImpl, '/api/ic-loras', 'get', [] as const, undefined, path)
    },

    startIcLoraDownload: makeEndpointClient(fetchImpl, '/api/ic-loras/download', 'post'),
    getIcLoraDownloadActive: makeEndpointClient(fetchImpl, '/api/ic-loras/download/active', 'get'),

    getIcLoraDownloadProgress(
      query: QueryFor<'/api/ic-loras/download/progress', 'get'>,
    ): Promise<EndpointResult<'/api/ic-loras/download/progress', 'get'>> {
      const path = `/api/ic-loras/download/progress${buildQueryString(query as Record<string, unknown>)}`
      return requestEndpointResult(fetchImpl, '/api/ic-loras/download/progress', 'get', [] as const, undefined, path)
    },

    deleteIcLoraInstallation: makeEndpointClient(fetchImpl, '/api/ic-loras/installation', 'delete'),

    ingestAsset: makeEndpointClient(fetchImpl, '/api/assets', 'post'),

    listAssets(
      query: QueryFor<'/api/assets', 'get'>,
    ): Promise<ExploreListAssetsResult> {
      const path = `/api/assets${buildQueryString(query as Record<string, unknown>)}`
      return requestParsedJson<ExploreAssetListResponse, []>(
        fetchImpl,
        path,
        'get',
        [] as const,
      )
    },

    uploadAsset(file: File): Promise<RemoteUploadAssetResult> {
      const body = new FormData()
      body.append('file', file)
      return requestParsedJson<ExploreAsset, []>(
        fetchImpl,
        '/api/assets/upload',
        'post',
        [] as const,
        { body },
      )
    },

    getAsset(
      assetId: string,
    ): Promise<EndpointResult<'/api/assets/{asset_id}', 'get', [404]>> {
      return requestEndpointResult(
        fetchImpl,
        '/api/assets/{asset_id}',
        'get',
        [404] as const,
        undefined,
        `/api/assets/${assetId}`,
      )
    },

    deleteAsset(
      assetId: string,
    ): Promise<EndpointResult<'/api/assets/{asset_id}', 'delete', [404, 409]>> {
      return requestEndpointResult(
        fetchImpl,
        '/api/assets/{asset_id}',
        'delete',
        [404, 409] as const,
        undefined,
        `/api/assets/${assetId}`,
      )
    },

    trimAudio(
      assetId: string,
      body: JsonBodyFor<'/api/assets/{asset_id}/trim-audio', 'post'>,
    ): Promise<
      EndpointResult<'/api/assets/{asset_id}/trim-audio', 'post', [404, 422]>
    > {
      return requestEndpointResult(
        fetchImpl,
        '/api/assets/{asset_id}/trim-audio',
        'post',
        [404, 422] as const,
        buildJsonRequestInit(body),
        `/api/assets/${assetId}/trim-audio`,
      )
    },

    trimVideo(
      assetId: string,
      body: JsonBodyFor<'/api/assets/{asset_id}/trim-video', 'post'>,
    ): Promise<
      EndpointResult<'/api/assets/{asset_id}/trim-video', 'post', [404, 422]>
    > {
      return requestEndpointResult(
        fetchImpl,
        '/api/assets/{asset_id}/trim-video',
        'post',
        [404, 422] as const,
        buildJsonRequestInit(body),
        `/api/assets/${assetId}/trim-video`,
      )
    },

    extractAudio(
      assetId: string,
    ): Promise<
      EndpointResult<'/api/assets/{asset_id}/extract-audio', 'post', [404, 422]>
    > {
      return requestEndpointResult(
        fetchImpl,
        '/api/assets/{asset_id}/extract-audio',
        'post',
        [404, 422] as const,
        undefined,
        `/api/assets/${assetId}/extract-audio`,
      )
    },

    getGenerationQueue: makeEndpointClient(fetchImpl, '/api/generation-queue', 'get'),

    reorderGenerationQueue: makeEndpointClient(fetchImpl, '/api/generation-queue/reorder', 'post'),

    clearGenerationQueueDone: makeEndpointClient(
      fetchImpl,
      '/api/generation-queue/done/clear',
      'post',
    ),

    clearGenerationQueueFailed: makeEndpointClient(
      fetchImpl,
      '/api/generation-queue/failed/clear',
      'post',
    ),

    markGenerationQueueDoneSeen(
      generationId: string,
    ): Promise<EndpointResult<'/api/generation-queue/done/{generation_id}/seen', 'post'>> {
      return requestEndpointResult(
        fetchImpl,
        '/api/generation-queue/done/{generation_id}/seen',
        'post',
        [] as const,
        undefined,
        `/api/generation-queue/done/${generationId}/seen`,
      )
    },

    dismissGenerationQueueDone(
      generationId: string,
    ): Promise<EndpointResult<'/api/generation-queue/done/{generation_id}/dismiss', 'post'>> {
      return requestEndpointResult(
        fetchImpl,
        '/api/generation-queue/done/{generation_id}/dismiss',
        'post',
        [] as const,
        undefined,
        `/api/generation-queue/done/${generationId}/dismiss`,
      )
    },

    listGenerations(
      query: QueryFor<'/api/generations', 'get'>,
    ): Promise<EndpointResult<'/api/generations', 'get'>> {
      const path = `/api/generations${buildQueryString(query as Record<string, unknown>)}`
      return requestEndpointResult(fetchImpl, '/api/generations', 'get', [] as const, undefined, path)
    },

    listRecentFeatures(
      query: QueryFor<'/api/generations/recent-features', 'get'>,
    ): Promise<EndpointResult<'/api/generations/recent-features', 'get'>> {
      const path = `/api/generations/recent-features${buildQueryString(query as Record<string, unknown>)}`
      return requestEndpointResult(
        fetchImpl,
        '/api/generations/recent-features',
        'get',
        [] as const,
        undefined,
        path,
      )
    },

    getDashboardSelection: makeEndpointClient(fetchImpl, '/api/stats/activity-dashboard-selections', 'get'),

    updateDashboardSelection: makeEndpointClient(fetchImpl, '/api/stats/activity-dashboard-selections', 'post'),

    getGenerationSeed: makeEndpointClient(fetchImpl, '/api/generation-seed', 'get'),

    updateGenerationSeed: makeEndpointClient(fetchImpl, '/api/generation-seed', 'post'),

    getDashboard(
      query: QueryFor<'/api/stats/dashboard', 'get'>,
    ): Promise<EndpointResult<'/api/stats/dashboard', 'get'>> {
      const path = `/api/stats/dashboard${buildQueryString(query as Record<string, unknown>)}`
      return requestEndpointResult(
        fetchImpl,
        '/api/stats/dashboard',
        'get',
        [] as const,
        undefined,
        path,
      )
    },

    createTextToVideo: makeEndpointClient(fetchImpl, '/api/generations/text-to-video', 'post'),

    createImageToVideo: makeEndpointClient(fetchImpl, '/api/generations/image-to-video', 'post'),

    createLoraRecipe(
      recipeId: string,
      body: JsonBodyFor<'/api/generations/recipes/{recipe_id}', 'post'>,
    ): Promise<EndpointResult<'/api/generations/recipes/{recipe_id}', 'post'>> {
      return requestEndpointResult(
        fetchImpl,
        '/api/generations/recipes/{recipe_id}',
        'post',
        [] as const,
        buildJsonRequestInit(body),
        `/api/generations/recipes/${encodeURIComponent(recipeId)}`,
      )
    },
    createAudioToVideo: makeEndpointClient(fetchImpl, '/api/generations/audio-to-video', 'post'),
    createRetake: makeEndpointClient(fetchImpl, '/api/generations/retake', 'post'),
    createIcLoraRecipe(
      recipeId: string,
      body: JsonBodyFor<'/api/generations/ic-lora-recipes/{recipe_id}', 'post'>,
    ): Promise<EndpointResult<'/api/generations/ic-lora-recipes/{recipe_id}', 'post'>> {
      return requestEndpointResult(
        fetchImpl,
        '/api/generations/ic-lora-recipes/{recipe_id}',
        'post',
        [] as const,
        buildJsonRequestInit(body),
        `/api/generations/ic-lora-recipes/${encodeURIComponent(recipeId)}`,
      )
    },
    createExtend: makeEndpointClient(fetchImpl, '/api/generations/extend', 'post'),

    deleteGeneration(
      generationId: string,
    ): Promise<EndpointResult<'/api/generations/{generation_id}', 'delete'>> {
      return requestEndpointResult(
        fetchImpl,
        '/api/generations/{generation_id}',
        'delete',
        [] as const,
        undefined,
        `/api/generations/${generationId}`,
      )
    },

    cancelQueuedGeneration(
      generationId: string,
    ): Promise<EndpointResult<'/api/generations/{generation_id}/cancel', 'post'>> {
      return requestEndpointResult(
        fetchImpl,
        '/api/generations/{generation_id}/cancel',
        'post',
        [] as const,
        undefined,
        `/api/generations/${generationId}/cancel`,
      )
    },

    retryGeneration(
      generationId: string,
    ): Promise<EndpointResult<'/api/generations/{generation_id}/retry', 'post'>> {
      return requestEndpointResult(
        fetchImpl,
        '/api/generations/{generation_id}/retry',
        'post',
        [] as const,
        undefined,
        `/api/generations/${generationId}/retry`,
      )
    },

    listLoras(
      query?: QueryFor<'/api/loras', 'get'>,
    ): Promise<EndpointResult<'/api/loras', 'get'>> {
      const path = `/api/loras${buildQueryString(query as Record<string, unknown>)}`
      return requestEndpointResult(fetchImpl, '/api/loras', 'get', [] as const, undefined, path)
    },

    startLoraDownload: makeEndpointClient(fetchImpl, '/api/loras/download', 'post'),

    getLoraDownloadProgress(
      query: QueryFor<'/api/loras/download/progress', 'get'>,
    ): Promise<EndpointResult<'/api/loras/download/progress', 'get'>> {
      const path = `/api/loras/download/progress${buildQueryString(query as Record<string, unknown>)}`
      return requestEndpointResult(fetchImpl, '/api/loras/download/progress', 'get', [] as const, undefined, path)
    },

    getLoraDownloadActive: makeEndpointClient(fetchImpl, '/api/loras/download/active', 'get'),

    deleteLoraInstallation: makeEndpointClient(fetchImpl, '/api/loras/installation', 'delete'),
  }
}

export type BoundApiClient = ReturnType<typeof createApiClient>

/** Electron-bound default for non-Explore Desktop callers. */
export const ApiClient: BoundApiClient = createApiClient(backendFetch)

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
