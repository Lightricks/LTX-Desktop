import { pathToFileUrl } from '../lib/file-url'
import { imagePathsFromDataTransfer } from '../lib/keyframe-drop'
import type { KeyframeItem } from '../lib/multi-keyframe'

interface KeyframePreviewProps {
  keyframe: KeyframeItem | null
  aspectRatio: string
  onDropImages: (paths: string[]) => void
}

export function KeyframePreview({
  keyframe,
  aspectRatio,
  onDropImages,
}: KeyframePreviewProps) {
  const cssAspect = aspectRatio.includes(':')
    ? aspectRatio.replace(':', ' / ')
    : aspectRatio

  return (
    <div
      className="flex min-h-0 flex-1 items-center justify-center"
      onDragOver={(event) => event.preventDefault()}
      onDrop={(event) => {
        event.preventDefault()
        const paths = imagePathsFromDataTransfer(event.dataTransfer)
        if (paths.length > 0) onDropImages(paths)
      }}
    >
      <div
        className="h-full max-w-full overflow-hidden rounded-xl border border-separator bg-surface-tertiary"
        style={{ aspectRatio: cssAspect }}
      >
        {keyframe ? (
          <img
            src={pathToFileUrl(keyframe.path)}
            alt="Keyframe at the playhead"
            draggable={false}
            className="h-full w-full object-contain"
          />
        ) : (
          <div className="flex h-full items-center justify-center px-6 text-center text-sm text-fg-tertiary">
            Add keyframes to preview your sequence
          </div>
        )}
      </div>
    </div>
  )
}
