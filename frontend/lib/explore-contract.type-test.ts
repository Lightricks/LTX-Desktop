import type { components } from '../generated/backend-openapi.ts'
import type {
  ExploreAsset,
  ExploreGeneration,
  ExploreListedAsset,
  ExploreAssetListResponse,
} from './explore-contract.ts'

type Equal<A, B> =
  (<T>() => T extends A ? 1 : 2) extends <T>() => T extends B ? 1 : 2
    ? true
    : false
type Expect<T extends true> = T

type DesktopAsset = components['schemas']['Asset']
type DesktopAssetListItem = components['schemas']['AssetListItem']
type DesktopGeneration = components['schemas']['Generation']

const desktopAssetAssignable: Expect<
  DesktopAsset extends ExploreAsset ? true : false
> = true
const desktopGenerationAssignable: Expect<
  DesktopGeneration extends ExploreGeneration ? true : false
> = true
const exploreAssetIsNotDesktopAsset: Expect<
  Equal<DesktopAsset, ExploreAsset> extends false ? true : false
> = true
const exploreGenerationIsNotDesktopGeneration: Expect<
  Equal<DesktopGeneration, ExploreGeneration> extends false ? true : false
> = true
const pathIsNotRequired: Expect<
  ExploreAsset extends { path: string } ? false : true
> = true
const listedPathIsNotRequired: Expect<
  ExploreListedAsset extends { path: string } ? false : true
> = true
const desktopListItemAssignable: Expect<
  DesktopAssetListItem extends ExploreListedAsset ? true : false
> = true
const desktopListResponseAssignable: Expect<
  components['schemas']['AssetListResponse'] extends ExploreAssetListResponse
    ? true
    : false
> = true

const pathFreeRemoteListedAsset: ExploreListedAsset = {
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
}

const pathFreeRemoteAsset: ExploreAsset = {
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
}

const pathFreeRemoteGeneration: ExploreGeneration = {
  attempt_count: 1,
  contract_version: 1,
  created_at: 1,
  feature: 'text-to-video',
  id: 'gen-1',
  outputs: [pathFreeRemoteAsset],
  queued_at: 1,
  spec: {
    params: { prompt: 'a dog on a beach', model: 'fast' },
  },
  status: 'succeeded',
}

const desktopAsset: DesktopAsset = {
  ...pathFreeRemoteAsset,
  path: '/tmp/photo.png',
  thumbnail_path: '/tmp/photo-thumb.png',
}
const exploreFromDesktopAsset: ExploreAsset = desktopAsset

const desktopGeneration: DesktopGeneration = {
  ...pathFreeRemoteGeneration,
  outputs: [desktopAsset],
}
const exploreFromDesktopGeneration: ExploreGeneration = desktopGeneration

// @ts-expect-error Remote path-free assets are not Desktop Assets
const remoteAssetIsNotDesktop: DesktopAsset = pathFreeRemoteAsset
// @ts-expect-error Remote path-free listed assets are not Desktop AssetListItems
const remoteListedIsNotDesktop: DesktopAssetListItem = pathFreeRemoteListedAsset
// @ts-expect-error Remote path-free generations are not Desktop Generations
const remoteGenerationIsNotDesktop: DesktopGeneration = pathFreeRemoteGeneration

void [
  desktopAssetAssignable,
  desktopGenerationAssignable,
  exploreAssetIsNotDesktopAsset,
  exploreGenerationIsNotDesktopGeneration,
  pathIsNotRequired,
  listedPathIsNotRequired,
  desktopListItemAssignable,
  desktopListResponseAssignable,
  pathFreeRemoteListedAsset,
  pathFreeRemoteAsset,
  pathFreeRemoteGeneration,
  exploreFromDesktopAsset,
  exploreFromDesktopGeneration,
  remoteAssetIsNotDesktop,
  remoteListedIsNotDesktop,
  remoteGenerationIsNotDesktop,
]
