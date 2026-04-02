import { useCallback, useRef } from 'react'
import type { Asset } from '../../types/project'
import { addGenericAssetToProject, addVisualAssetToProject } from '../../lib/asset-copy'
import { pathToFileUrl } from '../../lib/file-url'
import { useEditorActions } from './editor-store'

interface UseEditorMediaImportParams {
  currentProjectId: string | null
}

export function useEditorMediaImport(params: UseEditorMediaImportParams) {
  const { currentProjectId } = params
  const { addAssetToEditor } = useEditorActions()
  const fileInputRef = useRef<HTMLInputElement>(null)

  const getMediaDuration = useCallback((url: string, isAudio = false): Promise<number> => {
    return new Promise((resolve) => {
      const media = document.createElement(isAudio ? 'audio' : 'video')
      media.src = url
      media.onloadedmetadata = () => resolve(media.duration)
      media.onerror = () => resolve(5)
    })
  }, [])

  const handleImportFile = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (!files || !currentProjectId) return

    for (const file of Array.from(files)) {
      const isVideo = file.type.startsWith('video/')
      const isAudio = file.type.startsWith('audio/')
      const isImage = file.type.startsWith('image/')
      if (!isVideo && !isAudio && !isImage) continue

      const electronFilePath = window.electronAPI?.getPathForFile(file)
      let persistentPath = electronFilePath || file.name
      let bigThumbnailPath: string | undefined
      let smallThumbnailPath: string | undefined
      let width: number | undefined
      let height: number | undefined

      let duration = 5
      if (isVideo || isAudio) {
        const mediaUrl = electronFilePath ? pathToFileUrl(electronFilePath) : URL.createObjectURL(file)
        duration = await getMediaDuration(mediaUrl, isAudio)
      }

      if (electronFilePath) {
        if (isVideo || isImage) {
          const copied = await addVisualAssetToProject(electronFilePath, currentProjectId, isVideo ? 'video' : 'image')
          if (!copied) continue
          persistentPath = copied.path
          bigThumbnailPath = copied.bigThumbnailPath
          smallThumbnailPath = copied.smallThumbnailPath
          width = copied.width
          height = copied.height
        } else if (isAudio) {
          const copied = await addGenericAssetToProject(electronFilePath, currentProjectId)
          if (copied?.path) persistentPath = copied.path
        }
      }

      const asset: Asset = {
        id: `asset-${Date.now()}-${Math.random().toString(36).slice(2, 11)}`,
        type: isVideo ? 'video' : isAudio ? 'audio' : 'image',
        path: persistentPath,
        bigThumbnailPath,
        smallThumbnailPath,
        width,
        height,
        prompt: `Imported: ${file.name}`,
        resolution: 'imported',
        duration,
        createdAt: Date.now(),
      }
      addAssetToEditor(asset)
    }

    e.target.value = ''
    if (fileInputRef.current && fileInputRef.current !== e.target) {
      fileInputRef.current.value = ''
    }
  }, [addAssetToEditor, currentProjectId, getMediaDuration])

  return {
    fileInputRef,
    handleImportFile,
  }
}
