type ImagePathResolver = (file: File) => string | null

function defaultPathForFile(file: File): string | null {
  return window.electronAPI?.getPathForFile(file) ?? null
}

export function imagePathsFromFiles(
  files: ArrayLike<File>,
  getPathForFile: ImagePathResolver = defaultPathForFile,
): string[] {
  const paths: string[] = []
  for (let index = 0; index < files.length; index++) {
    const file = files[index]
    if (!file.type.startsWith('image/')) continue
    const path = getPathForFile(file)
    if (path) paths.push(path)
  }
  return paths
}

export function imagePathsFromDataTransfer(
  dataTransfer: DataTransfer,
  getPathForFile: ImagePathResolver = defaultPathForFile,
): string[] {
  const assetData = dataTransfer.getData('asset')
  if (assetData) {
    try {
      const asset = JSON.parse(assetData) as { type?: string; path?: string }
      return asset.type === 'image' && asset.path ? [asset.path] : []
    } catch {
      return []
    }
  }

  return imagePathsFromFiles(dataTransfer.files, getPathForFile)
}
