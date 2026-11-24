import type { components } from '../generated/backend-openapi.ts'
import {
  ApiClient,
  type ApiErrorsOf,
  type ApiRequestBodyOf,
  type ApiSuccessOf,
  type ExploreListAssetsResult,
  type RemoteUploadAssetResult,
} from './api-client.ts'
import type { ExploreAsset, ExploreListedAsset } from './explore-contract.ts'

type Equal<A, B> =
  (<T>() => T extends A ? 1 : 2) extends <T>() => T extends B ? 1 : 2
    ? true
    : false
type Expect<T extends true> = T

type Asset = components['schemas']['Asset']
type HTTPErrorResponse = components['schemas']['HTTPErrorResponse']
type IngestAssetRequest = components['schemas']['IngestAssetRequest']
type RemoteStatusResponse = components['schemas']['RemoteStatusResponse']

const uploadSuccessIsExploreAsset: Expect<
  Equal<ApiSuccessOf<'uploadAsset'>, ExploreAsset>
> = true
const desktopAssetFitsUploadSuccess: Expect<
  Asset extends ApiSuccessOf<'uploadAsset'> ? true : false
> = true
const uploadTakesOnlyTheFile: Expect<
  Equal<Parameters<typeof ApiClient.uploadAsset>, [file: File]>
> = true
const ingestBodyStaysPathJson: Expect<
  Equal<ApiRequestBodyOf<'ingestAsset'>, IngestAssetRequest>
> = true
const ingestBodyIsNotUploadArg: Expect<
  IngestAssetRequest extends Parameters<typeof ApiClient.uploadAsset>[0] ? false : true
> = true
const uploadResultIsLocalContract: Expect<
  Equal<Awaited<ReturnType<typeof ApiClient.uploadAsset>>, RemoteUploadAssetResult>
> = true
const uploadErrorsAreStandardHttp: Expect<
  Equal<
    ApiErrorsOf<'uploadAsset'>,
    | { status: '4XX'; error: HTTPErrorResponse }
    | { status: '5XX'; error: HTTPErrorResponse }
    | { status: 'default'; error: HTTPErrorResponse }
  >
> = true
const unwrapApiResultCompatible: Expect<
  RemoteUploadAssetResult extends
    | { ok: true; data: ExploreAsset }
    | { ok: false; status?: unknown; error: { message: string; code?: string } }
    ? true
    : false
> = true
const pathFreeUploadSuccess: RemoteUploadAssetResult = {
  ok: true,
  data: {
    created_at: 1,
    id: 'asset-1',
    media_kind: 'image',
    metadata: {
      mediaType: 'image',
      metadata: { width: 8, height: 8 },
    },
    mime_type: 'image/png',
    name: 'photo.png',
    origin: 'uploaded',
  },
}
const getRemoteStatusSuccess: Expect<
  Equal<ApiSuccessOf<'getRemoteStatus'>, RemoteStatusResponse>
> = true
const listSuccessItemsAreExploreListed: Expect<
  Equal<ApiSuccessOf<'listAssets'>['items'][number], ExploreListedAsset>
> = true
const listResultIsLocalContract: Expect<
  Equal<Awaited<ReturnType<typeof ApiClient.listAssets>>, ExploreListAssetsResult>
> = true
const listQueryPreserved: Expect<
  Equal<
    Parameters<typeof ApiClient.listAssets>[0],
    {
      media_kind?: ('image' | 'video' | 'audio') | null
      sort?: 'created_at-desc' | 'created_at-asc'
      q?: string | null
      cursor?: string | null
      limit?: number
    }
  >
> = true
const listErrorsAreStandardHttp: Expect<
  Equal<
    ApiErrorsOf<'listAssets'>,
    | { status: '4XX'; error: HTTPErrorResponse }
    | { status: '5XX'; error: HTTPErrorResponse }
    | { status: 'default'; error: HTTPErrorResponse }
  >
> = true
const pathFreeListSuccess: Awaited<ReturnType<typeof ApiClient.listAssets>> = {
  ok: true,
  data: {
    items: [
      {
        created_at: 1,
        id: 'asset-1',
        in_use: false,
        has_thumbnail: true,
        media_kind: 'image',
        metadata: {
          mediaType: 'image',
          metadata: { width: 8, height: 8 },
        },
        mime_type: 'image/png',
        name: 'photo.png',
        origin: 'generated',
      },
    ],
    next_cursor: null,
  },
}

void [
  uploadSuccessIsExploreAsset,
  desktopAssetFitsUploadSuccess,
  uploadTakesOnlyTheFile,
  ingestBodyStaysPathJson,
  ingestBodyIsNotUploadArg,
  uploadResultIsLocalContract,
  uploadErrorsAreStandardHttp,
  unwrapApiResultCompatible,
  pathFreeUploadSuccess,
  getRemoteStatusSuccess,
  listSuccessItemsAreExploreListed,
  listResultIsLocalContract,
  listQueryPreserved,
  listErrorsAreStandardHttp,
  pathFreeListSuccess,
]
