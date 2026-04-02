import { backendFetch } from './backend'
import type { paths } from '../generated/backend-openapi'

type HttpMethod = 'get' | 'post' | 'put' | 'patch' | 'delete'

type OperationFor<
  TPath extends keyof paths,
  TMethod extends HttpMethod,
> = NonNullable<paths[TPath][TMethod]>

type JsonResponseFor<
  TPath extends keyof paths,
  TMethod extends HttpMethod,
> = OperationFor<TPath, TMethod> extends {
  responses: { 200: { content: { 'application/json': infer TResponse } } }
}
  ? TResponse
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

export class ApiClientError extends Error {
  status: number
  endpoint: string
  payload: unknown

  constructor(message: string, status: number, endpoint: string, payload: unknown) {
    super(message)
    this.name = 'ApiClientError'
    this.status = status
    this.endpoint = endpoint
    this.payload = payload
  }
}

function toErrorMessage(payload: unknown, fallback: string): string {
  if (!payload || typeof payload !== 'object') {
    return fallback
  }
  const record = payload as Record<string, unknown>
  if (typeof record.error === 'string' && record.error.trim()) return record.error
  if (typeof record.message === 'string' && record.message.trim()) return record.message
  if (typeof record.detail === 'string' && record.detail.trim()) return record.detail
  return fallback
}

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

export class ApiClient {
  private static buildJsonRequestInit(body: unknown, init?: RequestInit): RequestInit {
    const headers = new Headers(init?.headers)
    headers.set('Content-Type', 'application/json')
    return {
      ...init,
      headers,
      body: JSON.stringify(body),
    }
  }

  private static async requestJson<
    TPath extends keyof paths,
    TMethod extends HttpMethod,
  >(
    endpoint: TPath,
    method: TMethod,
    init?: RequestInit,
    requestPath?: string,
  ): Promise<JsonResponseFor<TPath, TMethod>> {
    const path = requestPath ?? String(endpoint)
    const response = await backendFetch(path, {
      method: method.toUpperCase(),
      ...init,
    })

    if (!response.ok) {
      let payload: unknown = null
      let fallback = `${response.status} ${response.statusText || 'Request failed'}`
      try {
        const text = await response.text()
        if (text) {
          fallback = text
          payload = JSON.parse(text) as unknown
          fallback = toErrorMessage(payload, fallback)
        }
      } catch {
        // Keep fallback.
      }
      throw new ApiClientError(fallback, response.status, path, payload)
    }

    return (await response.json()) as JsonResponseFor<TPath, TMethod>
  }

  static getHealth(): Promise<JsonResponseFor<'/health', 'get'>> {
    return this.requestJson('/health', 'get')
  }

  static listModels(): Promise<JsonResponseFor<'/api/models', 'get'>> {
    return this.requestJson('/api/models', 'get')
  }

  static getModelsStatus(): Promise<JsonResponseFor<'/api/models/status', 'get'>> {
    return this.requestJson('/api/models/status', 'get')
  }

  static getModelDownloadProgress(
    query: QueryFor<'/api/models/download/progress', 'get'>,
  ): Promise<JsonResponseFor<'/api/models/download/progress', 'get'>> {
    const path = `/api/models/download/progress${buildQueryString(query as Record<string, unknown>)}`
    return this.requestJson('/api/models/download/progress', 'get', undefined, path)
  }

  static getRequiredModels(
    query: QueryFor<'/api/models/required-models', 'get'> = {},
  ): Promise<JsonResponseFor<'/api/models/required-models', 'get'>> {
    const path = `/api/models/required-models${buildQueryString(query as Record<string, unknown>)}`
    return this.requestJson('/api/models/required-models', 'get', undefined, path)
  }

  static startModelDownload(
    body: JsonBodyFor<'/api/models/download', 'post'>,
  ): Promise<JsonResponseFor<'/api/models/download', 'post'>> {
    return this.requestJson('/api/models/download', 'post', this.buildJsonRequestInit(body))
  }

  static startTextEncoderDownload(): Promise<JsonResponseFor<'/api/text-encoder/download', 'post'>> {
    return this.requestJson('/api/text-encoder/download', 'post')
  }

  static getRuntimePolicy(): Promise<JsonResponseFor<'/api/runtime-policy', 'get'>> {
    return this.requestJson('/api/runtime-policy', 'get')
  }

  static getSettings(): Promise<JsonResponseFor<'/api/settings', 'get'>> {
    return this.requestJson('/api/settings', 'get')
  }

  static updateSettings(
    body: JsonBodyFor<'/api/settings', 'post'>,
  ): Promise<JsonResponseFor<'/api/settings', 'post'>> {
    return this.requestJson('/api/settings', 'post', this.buildJsonRequestInit(body))
  }

  static suggestGapPrompt(
    body: JsonBodyFor<'/api/suggest-gap-prompt', 'post'>,
    init?: RequestInit,
  ): Promise<JsonResponseFor<'/api/suggest-gap-prompt', 'post'>> {
    return this.requestJson('/api/suggest-gap-prompt', 'post', this.buildJsonRequestInit(body, init))
  }

  static generateVideo(
    body: JsonBodyFor<'/api/generate', 'post'>,
    init?: RequestInit,
  ): Promise<JsonResponseFor<'/api/generate', 'post'>> {
    return this.requestJson('/api/generate', 'post', this.buildJsonRequestInit(body, init))
  }

  static cancelGeneration(): Promise<JsonResponseFor<'/api/generate/cancel', 'post'>> {
    return this.requestJson('/api/generate/cancel', 'post')
  }

  static getGenerationProgress(): Promise<JsonResponseFor<'/api/generation/progress', 'get'>> {
    return this.requestJson('/api/generation/progress', 'get')
  }

  static generateImage(
    body: JsonBodyFor<'/api/generate-image', 'post'>,
    init?: RequestInit,
  ): Promise<JsonResponseFor<'/api/generate-image', 'post'>> {
    return this.requestJson('/api/generate-image', 'post', this.buildJsonRequestInit(body, init))
  }

  static retake(
    body: JsonBodyFor<'/api/retake', 'post'>,
  ): Promise<JsonResponseFor<'/api/retake', 'post'>> {
    return this.requestJson('/api/retake', 'post', this.buildJsonRequestInit(body))
  }

  static generateIcLora(
    body: JsonBodyFor<'/api/ic-lora/generate', 'post'>,
  ): Promise<JsonResponseFor<'/api/ic-lora/generate', 'post'>> {
    return this.requestJson('/api/ic-lora/generate', 'post', this.buildJsonRequestInit(body))
  }

  static extractIcLoraConditioning(
    body: JsonBodyFor<'/api/ic-lora/extract-conditioning', 'post'>,
  ): Promise<JsonResponseFor<'/api/ic-lora/extract-conditioning', 'post'>> {
    return this.requestJson('/api/ic-lora/extract-conditioning', 'post', this.buildJsonRequestInit(body))
  }
}
