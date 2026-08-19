import type { Asset } from '../types/project-model'

export type GenSpaceTypeFilter = 'all' | 'video' | 'image'
export type GenSpaceSortKey = 'createdAt' | 'type' | 'duration' | 'resolution' | 'ratio'
export type GenSpaceSortDir = 'asc' | 'desc'

export const GENSPACE_TYPE_FILTER_OPTIONS: { value: GenSpaceTypeFilter; label: string }[] = [
  { value: 'all', label: 'All' },
  { value: 'video', label: 'Videos' },
  { value: 'image', label: 'Images' },
]

export const GENSPACE_SORT_OPTIONS: { value: GenSpaceSortKey; label: string }[] = [
  { value: 'createdAt', label: 'Date' },
  { value: 'type', label: 'Type' },
  { value: 'duration', label: 'Duration' },
  { value: 'resolution', label: 'Resolution' },
  { value: 'ratio', label: 'Ratio' },
]

const VISUAL_TYPES = new Set(['video', 'image'])

export function defaultSortDir(key: GenSpaceSortKey): GenSpaceSortDir {
  switch (key) {
    case 'createdAt':
    case 'duration':
    case 'resolution':
    case 'ratio':
    case 'type':
      return 'desc'
  }
}

export function filterGenSpaceAssets(
  assets: Asset[],
  typeFilter: GenSpaceTypeFilter,
  favoritesOnly: boolean,
): Asset[] {
  let result = assets.filter(asset => VISUAL_TYPES.has(asset.type))
  if (typeFilter !== 'all') {
    result = result.filter(asset => asset.type === typeFilter)
  }
  if (favoritesOnly) {
    result = result.filter(asset => asset.favorite)
  }
  return result
}

function resolutionRank(asset: Asset): number {
  if (asset.width && asset.height) return Math.min(asset.width, asset.height)
  const match = asset.resolution?.match(/(\d+)/)
  return match ? parseInt(match[1], 10) : 0
}

function durationRank(asset: Asset): number {
  return asset.duration ?? 0
}

function parseAspectRatioString(value?: string): number {
  if (!value) return 0
  const match = value.match(/^(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)$/)
  if (!match) return 0
  const height = Number(match[2])
  return height === 0 ? 0 : Number(match[1]) / height
}

function ratioRank(asset: Asset): number {
  if (asset.width && asset.height) return asset.width / asset.height
  return parseAspectRatioString(asset.generationParams?.imageAspectRatio)
}

function compareTieBreak(a: Asset, b: Asset): number {
  if (a.createdAt !== b.createdAt) return b.createdAt - a.createdAt
  return a.id.localeCompare(b.id)
}

function comparePrimary(a: Asset, b: Asset, key: GenSpaceSortKey): number {
  switch (key) {
    case 'type':
      return a.type.localeCompare(b.type)
    case 'duration':
      return durationRank(a) - durationRank(b)
    case 'resolution':
      return resolutionRank(a) - resolutionRank(b)
    case 'ratio':
      return ratioRank(a) - ratioRank(b)
    case 'createdAt':
    default:
      return a.createdAt - b.createdAt
  }
}

export function sortGenSpaceAssets(
  assets: Asset[],
  key: GenSpaceSortKey,
  direction: GenSpaceSortDir,
): Asset[] {
  const dir = direction === 'desc' ? -1 : 1
  return [...assets].sort((a, b) => {
    const primary = comparePrimary(a, b, key)
    if (primary !== 0) return dir * primary
    return compareTieBreak(a, b)
  })
}

export function shouldShowGeneratingTile(
  typeFilter: GenSpaceTypeFilter,
  mode: 'image' | 'video',
): boolean {
  if (typeFilter === 'all') return true
  return typeFilter === mode
}
