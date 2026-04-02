import { dialog } from 'electron'
import path from 'path'
import fs from 'fs'
import { getAllowedRoots } from '../config'
import { logger } from '../logger'
import { getMainWindow } from '../window'
import { validatePath, approvePath } from '../path-validation'
import { getProjectAssetsPath, setProjectAssetsPath } from '../app-state'
import { extractVideoFrameToFile, getVideoDimensions } from '../export/ffmpeg-utils'
import { createDownsampledThumbnail, getImageDimensions, getThumbnailPaths } from './image-utils'
import { handle } from './typed-handle'

const MIME_TYPES: Record<string, string> = {
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.webp': 'image/webp',
  '.gif': 'image/gif',
  '.mp3': 'audio/mpeg',
  '.wav': 'audio/wav',
  '.ogg': 'audio/ogg',
  '.aac': 'audio/aac',
  '.flac': 'audio/flac',
  '.m4a': 'audio/mp4',
  '.mp4': 'video/mp4',
  '.webm': 'video/webm',
  '.mkv': 'video/x-matroska',
  '.mov': 'video/quicktime',
}

function readLocalFileAsBase64(filePath: string): { data: string; mimeType: string } {
  const data = fs.readFileSync(filePath)
  const base64 = data.toString('base64')
  const ext = path.extname(filePath).toLowerCase()
  const mimeType = MIME_TYPES[ext] || 'application/octet-stream'
  return { data: base64, mimeType }
}

function searchDirectoryForFilesImpl(dir: string, filenames: string[]): Record<string, string> {
  const results: Record<string, string> = {}
  const remaining = new Set(filenames.map(f => f.toLowerCase()))

  const walk = (currentDir: string, depth: number) => {
    if (remaining.size === 0 || depth > 10) return
    try {
      const entries = fs.readdirSync(currentDir, { withFileTypes: true })
      for (const entry of entries) {
        if (remaining.size === 0) break
        const fullPath = path.join(currentDir, entry.name)
        if (entry.isFile()) {
          const lower = entry.name.toLowerCase()
          if (remaining.has(lower)) {
            results[lower] = fullPath
            remaining.delete(lower)
          }
        } else if (entry.isDirectory() && !entry.name.startsWith('.')) {
          walk(fullPath, depth + 1)
        }
      }
    } catch {
      // Skip directories we can't read (permissions, etc.)
    }
  }

  walk(dir, 0)
  return results
}

function resolveLocalSourcePath(srcPath: string): string {
  if (!srcPath || !srcPath.trim()) {
    throw new Error('Source path is empty')
  }

  const normalized = srcPath.trim()

  if (!path.isAbsolute(normalized)) {
    throw new Error(`Source path must be absolute: ${srcPath}`)
  }

  const resolved = path.resolve(normalized)
  if (!fs.existsSync(resolved)) {
    throw new Error(`Source file does not exist: ${resolved}`)
  }
  if (!fs.statSync(resolved).isFile()) {
    throw new Error(`Source path is not a file: ${resolved}`)
  }
  return resolved
}

function getUniqueDestinationPath(destDir: string, fileName: string): string {
  const parsed = path.parse(fileName)
  let candidate = path.join(destDir, fileName)
  let idx = 1
  while (fs.existsSync(candidate)) {
    candidate = path.join(destDir, `${parsed.name}(${idx})${parsed.ext}`)
    idx += 1
  }
  return candidate
}

function copyToProjectAssetDirectory(srcPath: string, projectId: string): string {
  const assetsRoot = getProjectAssetsPath()
  const destDir = path.join(assetsRoot, projectId)
  fs.mkdirSync(destDir, { recursive: true })
  const fileName = path.basename(srcPath)
  const destPath = getUniqueDestinationPath(destDir, fileName)
  fs.copyFileSync(srcPath, destPath)
  return destPath
}

function createVideoBigThumbnail(videoPath: string, bigThumbnailPath: string): void {
  extractVideoFrameToFile({
    videoPath,
    seekTime: 0,
    outputPath: bigThumbnailPath,
    timeoutMs: 30000,
  })
}

function createVisualThumbnails(assetPath: string, type: 'video' | 'image'): { bigThumbnailPath: string; smallThumbnailPath: string } {
  const { bigThumbnailPath: generatedBigThumbnailPath, smallThumbnailPath } = getThumbnailPaths(assetPath)
  let bigThumbnailPath: string

  switch (type) {
    case 'video':
      bigThumbnailPath = generatedBigThumbnailPath
      createVideoBigThumbnail(assetPath, bigThumbnailPath)
      break
    case 'image':
      bigThumbnailPath = assetPath
      break
    default: {
      const unsupportedType: never = type
      throw new Error(`Unsupported visual asset type: ${unsupportedType}`)
    }
  }

  createDownsampledThumbnail(bigThumbnailPath, smallThumbnailPath)
  return { bigThumbnailPath, smallThumbnailPath }
}

function getVisualAssetDimensions(assetPath: string, type: 'video' | 'image'): { width: number; height: number } {
  switch (type) {
    case 'video':
      return getVideoDimensions(assetPath)
    case 'image':
      return getImageDimensions(assetPath)
    default: {
      const unsupportedType: never = type
      throw new Error(`Unsupported visual asset type: ${unsupportedType}`)
    }
  }
}

export function registerFileHandlers(): void {
  handle('openLtxApiKeyPage', async () => {
    const { shell } = await import('electron')
    await shell.openExternal('https://console.ltx.video/api-keys/')
    return true
  })

  handle('openFalApiKeyPage', async () => {
    const { shell } = await import('electron')
    await shell.openExternal('https://fal.ai/dashboard/keys')
    return true
  })

  handle('openParentFolderOfFile', async ({ filePath }) => {
    const { shell } = await import('electron')
    const normalizedPath = validatePath(filePath, getAllowedRoots())
    const parentDir = path.dirname(normalizedPath)
    if (!fs.existsSync(parentDir) || !fs.statSync(parentDir).isDirectory()) {
      throw new Error(`Parent directory not found: ${parentDir}`)
    }
    shell.openPath(parentDir)
  })

  handle('showItemInFolder', async ({ filePath }) => {
    const { shell } = await import('electron')
    shell.showItemInFolder(filePath)
  })

  handle('readLocalFile', async ({ filePath }) => {
    try {
      const normalizedPath = validatePath(filePath, getAllowedRoots())

      if (!fs.existsSync(normalizedPath)) {
        throw new Error(`File not found: ${normalizedPath}`)
      }

      return readLocalFileAsBase64(normalizedPath)
    } catch (error) {
      logger.error( `Error reading local file: ${error}`)
      throw error
    }
  })

  handle('showSaveDialog', async ({ title, defaultPath, filters }) => {
    const mainWindow = getMainWindow()
    if (!mainWindow) return null
    const result = await dialog.showSaveDialog(mainWindow, {
      title: title || 'Save File',
      defaultPath,
      filters: filters || [],
    })
    if (result.canceled || !result.filePath) return null
    approvePath(result.filePath)
    return result.filePath
  })

  handle('saveFile', async ({ filePath, data, encoding }) => {
    try {
      validatePath(filePath, getAllowedRoots())
      if (encoding === 'base64') {
        fs.writeFileSync(filePath, Buffer.from(data, 'base64'))
      } else {
        fs.writeFileSync(filePath, data, 'utf-8')
      }
      return { success: true, path: filePath }
    } catch (error) {
      logger.error( `Error saving file: ${error}`)
      return { success: false, error: String(error) }
    }
  })

  handle('saveBinaryFile', async ({ filePath, data }) => {
    try {
      validatePath(filePath, getAllowedRoots())
      fs.writeFileSync(filePath, Buffer.from(data))
      return { success: true, path: filePath }
    } catch (error) {
      logger.error( `Error saving binary file: ${error}`)
      return { success: false, error: String(error) }
    }
  })

  handle('showOpenDirectoryDialog', async ({ title }) => {
    const mainWindow = getMainWindow()
    if (!mainWindow) return null
    const result = await dialog.showOpenDialog(mainWindow, {
      title: title || 'Select Folder',
      properties: ['openDirectory', 'createDirectory'],
    })
    if (result.canceled || result.filePaths.length === 0) return null
    approvePath(result.filePaths[0])
    return result.filePaths[0]
  })

  handle('searchDirectoryForFiles', ({ directory, filenames }) => {
    return searchDirectoryForFilesImpl(directory, filenames)
  })

  handle('addVisualAssetToProject', ({ srcPath, projectId, type }) => {
    try {
      const resolvedSrc = resolveLocalSourcePath(srcPath)
      const destPath = copyToProjectAssetDirectory(resolvedSrc, projectId)
      const { bigThumbnailPath, smallThumbnailPath } = createVisualThumbnails(destPath, type)
      const { width, height } = getVisualAssetDimensions(destPath, type)

      return {
        success: true,
        path: destPath,
        bigThumbnailPath,
        smallThumbnailPath,
        width,
        height,
      }
    } catch (error) {
      logger.error(`Error adding asset to project: ${error}`)
      return { success: false, error: String(error) }
    }
  })

  handle('addGenericAssetToProject', ({ srcPath, projectId }) => {
    try {
      const resolvedSrc = resolveLocalSourcePath(srcPath)
      const destPath = copyToProjectAssetDirectory(resolvedSrc, projectId)
      return { success: true, path: destPath }
    } catch (error) {
      logger.error(`Error copying file to project assets: ${error}`)
      return { success: false, error: String(error) }
    }
  })

  handle('makeThumbnailsForProjectAsset', ({ path: assetPath, type }) => {
    try {
      const resolvedAssetPath = resolveLocalSourcePath(assetPath)
      const { bigThumbnailPath, smallThumbnailPath } = createVisualThumbnails(resolvedAssetPath, type)

      return {
        success: true,
        bigThumbnailPath,
        smallThumbnailPath,
      }
    } catch (error) {
      logger.error(`Error creating thumbnails for project asset: ${error}`)
      return { success: false, error: String(error) }
    }
  })

  handle('makeDimensionsForProjectAsset', ({ path: assetPath, type }) => {
    try {
      const resolvedAssetPath = resolveLocalSourcePath(assetPath)
      const { width, height } = getVisualAssetDimensions(resolvedAssetPath, type)

      return {
        success: true,
        width,
        height,
      }
    } catch (error) {
      logger.error(`Error creating dimensions for project asset: ${error}`)
      return { success: false, error: String(error) }
    }
  })

  handle('getProjectAssetsPath', () => {
    return getProjectAssetsPath()
  })

  handle('openProjectAssetsPathChangeDialog', async () => {
    try {
      const mainWindow = getMainWindow()
      if (!mainWindow) return { success: false, error: 'No window' }
      const result = await dialog.showOpenDialog(mainWindow, {
        title: 'Select Project Assets Path',
        properties: ['openDirectory', 'createDirectory'],
      })
      if (result.canceled || result.filePaths.length === 0) return { success: false, error: 'cancelled' }
      const selectedPath = path.resolve(result.filePaths[0])
      setProjectAssetsPath(selectedPath)
      approvePath(selectedPath)
      return { success: true, path: selectedPath }
    } catch (error) {
      return { success: false, error: String(error) }
    }
  })

  handle('checkFilesExist', ({ filePaths }) => {
    const results: Record<string, boolean> = {}
    for (const p of filePaths) {
      try {
        results[p] = fs.existsSync(p)
      } catch {
        results[p] = false
      }
    }
    return results
  })

  handle('showOpenFileDialog', async ({ title, filters, properties }) => {
    const mainWindow = getMainWindow()
    if (!mainWindow) return null
    const props: any[] = ['openFile']
    if (properties?.includes('multiSelections')) props.push('multiSelections')
    const result = await dialog.showOpenDialog(mainWindow, {
      title: title || 'Select File',
      filters: filters || [],
      properties: props,
    })
    if (result.canceled || result.filePaths.length === 0) return null
    for (const fp of result.filePaths) {
      approvePath(fp)
    }
    return result.filePaths
  })

}
