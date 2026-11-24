import { useState, useEffect, useCallback } from 'react'
import { VideoPreviewPanel } from './VideoPreviewPanel'
import { validateVideoSource } from '../lib/video-constraints'

// Preview-only panel for extend. The direction + seconds-to-add controls live in the
// PromptBar (like Video mode), so this just owns the source video + preview.

interface ExtendPanelProps {
  initialVideoPath?: string | null
  initialDuration?: number
  resetKey?: number
  isProcessing?: boolean
  processingStatus?: string
  fillHeight?: boolean
  // When the request will hit the cloud API, enforce its input constraints (4K / aspect /
  // min duration). In local mode resolution is corrected, so these don't block.
  enforceApiConstraints?: boolean
  onChange?: (data: {
    videoPath: string | null
    videoDuration: number
    width: number
    height: number
    ready: boolean
  }) => void
}

export function ExtendPanel({
  initialVideoPath,
  initialDuration,
  resetKey,
  isProcessing = false,
  processingStatus = '',
  fillHeight = false,
  enforceApiConstraints = true,
  onChange,
}: ExtendPanelProps) {
  const [videoPath, setVideoPath] = useState<string | null>(initialVideoPath || null)
  const [videoDuration, setVideoDuration] = useState<number>(initialDuration || 0)
  const [dimensions, setDimensions] = useState<{ width: number; height: number }>({ width: 0, height: 0 })

  const handleSourceChange = useCallback((data: { videoPath: string | null; videoDuration: number; width: number; height: number }) => {
    setVideoPath(data.videoPath)
    setVideoDuration(data.videoDuration)
    setDimensions({ width: data.width, height: data.height })
  }, [])

  const error = enforceApiConstraints && videoPath
    ? validateVideoSource({ width: dimensions.width, height: dimensions.height, duration: videoDuration })
    : null

  useEffect(() => {
    onChange?.({ videoPath, videoDuration, width: dimensions.width, height: dimensions.height, ready: !!videoPath && !error })
  }, [videoPath, videoDuration, dimensions, error, onChange])

  return (
    <VideoPreviewPanel
      title="Extend"
      initialVideoPath={initialVideoPath}
      initialDuration={initialDuration}
      resetKey={resetKey}
      isProcessing={isProcessing}
      processingStatus={processingStatus}
      processingDefault="Processing extend..."
      fillHeight={fillHeight}
      emptyTitle="Drop a video to extend"
      hint={{ title: 'Add new frames to your video', subtitle: 'Set direction and length in the bar below, then extend' }}
      errorMessage={error ?? undefined}
      onSourceChange={handleSourceChange}
    />
  )
}
