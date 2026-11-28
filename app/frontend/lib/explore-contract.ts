import type { components } from '../generated/backend-openapi.ts'

type DesktopAsset = components['schemas']['Asset']
type DesktopGeneration = components['schemas']['Generation']

/**
 * Shared Explore Asset: Desktop's required `path` / optional `thumbnail_path`
 * plus Remote's path-free projection. Consumers must treat both path fields as
 * optional because Remote omits them.
 */
export type ExploreAsset = Omit<DesktopAsset, 'path' | 'thumbnail_path'> & {
  path?: DesktopAsset['path']
  thumbnail_path?: DesktopAsset['thumbnail_path']
  bytes_url?: string
  thumbnail_url?: string | null
}

/**
 * Shared Explore Generation: Desktop Generation with ExploreAsset outputs.
 * `spec` stays the generic JSON map that form/result readers already consume.
 */
export type ExploreGeneration = Omit<DesktopGeneration, 'outputs'> & {
  outputs: ExploreAsset[]
}

export type ExploreListedAsset = ExploreAsset & {
  in_use: boolean
  has_thumbnail?: boolean
}

export type ExploreAssetListResponse = {
  items: ExploreListedAsset[]
  next_cursor?: string | null
}
