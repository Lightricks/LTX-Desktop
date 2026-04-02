import type { ElectronAPI } from '../../shared/electron-api-schema'
import type { Asset } from '../types/project'
import { logger } from './logger'

interface VisualAssetMetadataMigrationJob {
  path: string
  type: 'video' | 'image'
  needsThumbnails: boolean
  needsDimensions: boolean
}

interface ThumbnailPaths {
  bigThumbnailPath: string
  smallThumbnailPath: string
}

interface VisualAssetDimensions {
  width: number
  height: number
}

interface VisualAssetMigrationResult extends Partial<ThumbnailPaths>, Partial<VisualAssetDimensions> {}

export interface VisualAssetMetadataMigrationUpdate {
  assetId: string
  updates: Partial<Asset>
}

export type VisualAssetMetadataMigrationEvent =
  | {
      kind: 'progress'
      total: number
      completed: number
    }
  | {
      kind: 'complete'
      total: number
      completed: number
      updates: VisualAssetMetadataMigrationUpdate[]
    }

function isVisualAsset(asset: Asset): asset is Asset & { type: 'video' | 'image' } {
  return asset.type === 'video' || asset.type === 'image'
}

function isMissingThumbnailPair(item: { bigThumbnailPath?: string; smallThumbnailPath?: string }): boolean {
  return !item.bigThumbnailPath || !item.smallThumbnailPath
}

function isMissingDimensions(item: { width?: number; height?: number }): boolean {
  return !item.width || !item.height
}

function collectVisualAssetMetadataMigrationJobs(assets: Asset[]): VisualAssetMetadataMigrationJob[] {
  const jobs = new Map<string, VisualAssetMetadataMigrationJob>()

  for (const asset of assets) {
    if (!isVisualAsset(asset)) continue

    if (asset.path && (isMissingThumbnailPair(asset) || isMissingDimensions(asset))) {
      const existingJob = jobs.get(asset.path)
      jobs.set(asset.path, {
        path: asset.path,
        type: asset.type,
        needsThumbnails: (existingJob?.needsThumbnails || false) || isMissingThumbnailPair(asset),
        needsDimensions: (existingJob?.needsDimensions || false) || isMissingDimensions(asset),
      })
    }

    for (const take of asset.takes ?? []) {
      if (take.path && (isMissingThumbnailPair(take) || isMissingDimensions(take))) {
        const existingJob = jobs.get(take.path)
        jobs.set(take.path, {
          path: take.path,
          type: asset.type,
          needsThumbnails: (existingJob?.needsThumbnails || false) || isMissingThumbnailPair(take),
          needsDimensions: (existingJob?.needsDimensions || false) || isMissingDimensions(take),
        })
      }
    }
  }

  return Array.from(jobs.values())
}

function buildVisualAssetMetadataMigrationPatch(
  asset: Asset,
  migrationResults: Map<string, VisualAssetMigrationResult>,
): Partial<Asset> | null {
  if (!isVisualAsset(asset)) {
    return null
  }

  const updates: Partial<Asset> = {}
  const assetMetadata = migrationResults.get(asset.path)

  if (assetMetadata?.bigThumbnailPath && asset.bigThumbnailPath !== assetMetadata.bigThumbnailPath) {
    updates.bigThumbnailPath = assetMetadata.bigThumbnailPath
  }
  if (assetMetadata?.smallThumbnailPath && asset.smallThumbnailPath !== assetMetadata.smallThumbnailPath) {
    updates.smallThumbnailPath = assetMetadata.smallThumbnailPath
  }
  if (assetMetadata?.width && asset.width !== assetMetadata.width) {
    updates.width = assetMetadata.width
  }
  if (assetMetadata?.height && asset.height !== assetMetadata.height) {
    updates.height = assetMetadata.height
  }

  if (asset.takes && asset.takes.length > 0) {
    let takesChanged = false
    const nextTakes = asset.takes.map(take => {
      const takeMetadata = migrationResults.get(take.path)
      if (!takeMetadata) {
        return take
      }

      if (
        (takeMetadata.bigThumbnailPath === undefined || take.bigThumbnailPath === takeMetadata.bigThumbnailPath)
        && (takeMetadata.smallThumbnailPath === undefined || take.smallThumbnailPath === takeMetadata.smallThumbnailPath)
        && (takeMetadata.width === undefined || take.width === takeMetadata.width)
        && (takeMetadata.height === undefined || take.height === takeMetadata.height)
      ) {
        return take
      }

      takesChanged = true
      return {
        ...take,
        ...(takeMetadata.bigThumbnailPath ? { bigThumbnailPath: takeMetadata.bigThumbnailPath } : {}),
        ...(takeMetadata.smallThumbnailPath ? { smallThumbnailPath: takeMetadata.smallThumbnailPath } : {}),
        ...(takeMetadata.width ? { width: takeMetadata.width } : {}),
        ...(takeMetadata.height ? { height: takeMetadata.height } : {}),
      }
    })

    if (takesChanged) {
      updates.takes = nextTakes
    }
  }

  return Object.keys(updates).length > 0 ? updates : null
}

export function hasVisualAssetMetadataForMigration(assets: Asset[]): boolean {
  return assets.some(asset => {
    if (!isVisualAsset(asset)) return false
    if (isMissingThumbnailPair(asset) || isMissingDimensions(asset)) return true
    return (asset.takes ?? []).some(take => isMissingThumbnailPair(take) || isMissingDimensions(take))
  })
}

export async function* runVisualAssetMetadataMigration(
  assets: Asset[],
  electronAPI: Pick<ElectronAPI, 'makeThumbnailsForProjectAsset' | 'makeDimensionsForProjectAsset'>,
): AsyncGenerator<VisualAssetMetadataMigrationEvent> {
  const jobs = collectVisualAssetMetadataMigrationJobs(assets)

  if (jobs.length === 0) {
    yield {
      kind: 'complete',
      total: 0,
      completed: 0,
      updates: [],
    }
    return
  }

  yield {
    kind: 'progress',
    total: jobs.length,
    completed: 0,
  }

  const migrationResults = new Map<string, VisualAssetMigrationResult>()
  let completed = 0

  for (const job of jobs) {
    try {
      const nextResult: VisualAssetMigrationResult = {}

      if (job.needsThumbnails) {
        const thumbnailResult = await electronAPI.makeThumbnailsForProjectAsset({
          path: job.path,
          type: job.type,
        })
        if (thumbnailResult.success) {
          nextResult.bigThumbnailPath = thumbnailResult.bigThumbnailPath
          nextResult.smallThumbnailPath = thumbnailResult.smallThumbnailPath
        } else {
          logger.warn(`Thumbnail migration skipped for ${job.path}: ${thumbnailResult.error}`)
        }
      }

      if (job.needsDimensions) {
        const dimensionsResult = await electronAPI.makeDimensionsForProjectAsset({
          path: job.path,
          type: job.type,
        })
        if (dimensionsResult.success) {
          nextResult.width = dimensionsResult.width
          nextResult.height = dimensionsResult.height
        } else {
          logger.warn(`Dimensions migration skipped for ${job.path}: ${dimensionsResult.error}`)
        }
      }

      if (Object.keys(nextResult).length > 0) {
        migrationResults.set(job.path, nextResult)
      }
    } catch (error) {
      logger.warn(`Asset metadata migration skipped for ${job.path}: ${error}`)
    }

    completed += 1
    yield {
      kind: 'progress',
      total: jobs.length,
      completed,
    }
  }

  const updates: VisualAssetMetadataMigrationUpdate[] = []
  for (const asset of assets) {
    const assetUpdates = buildVisualAssetMetadataMigrationPatch(asset, migrationResults)
    if (assetUpdates) {
      updates.push({
        assetId: asset.id,
        updates: assetUpdates,
      })
    }
  }

  yield {
    kind: 'complete',
    total: jobs.length,
    completed,
    updates,
  }
}
