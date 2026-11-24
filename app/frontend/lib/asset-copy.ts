import { logger } from './logger'

export type ProjectAssetType = 'video' | 'image'

export interface ProjectAssetCopyResult {
  path: string
  bigThumbnailPath: string
  smallThumbnailPath: string
  width: number
  height: number
}

/**
 * Copy a video/image file to project storage and return precomputed thumbnail paths.
 */
export async function addVisualAssetToProject(
  srcPath: string,
  projectId: string,
  type: ProjectAssetType,
): Promise<ProjectAssetCopyResult | null> {
  try {
    const result = await window.electronAPI.addVisualAssetToProject({ srcPath, projectId, type })
    if (result.success) {
      return {
        path: result.path,
        bigThumbnailPath: result.bigThumbnailPath,
        smallThumbnailPath: result.smallThumbnailPath,
        width: result.width,
        height: result.height,
      }
    }
    logger.warn(`Failed to add asset to project folder: ${result.error}`)
  } catch (e) {
    logger.warn(`Failed to add asset to project folder: ${e}`)
  }
  return null
}

/**
 * Copy a file to project storage without thumbnail generation (audio path).
 */
export async function addGenericAssetToProject(
  srcPath: string,
  projectId: string,
): Promise<{ path: string } | null> {
  try {
    const result = await window.electronAPI.addGenericAssetToProject({ srcPath, projectId })
    if (result.success) {
      return { path: result.path }
    }
    logger.warn(`Failed to copy file to project folder: ${result.error}`)
  } catch (e) {
    logger.warn(`Failed to copy file to project folder: ${e}`)
  }
  return null
}
