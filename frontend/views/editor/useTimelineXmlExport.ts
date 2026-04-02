import { useCallback } from 'react'
import type { Asset, TimelineClip } from '../../types/project'
import {
  selectActiveTimeline,
  selectAssets,
  selectClips,
} from './editor-selectors'
import { exportFcp7Xml } from '../../lib/timeline-import'
import { useEditorStore } from './editor-store'

interface TimelineXmlExportClip {
  name: string
  filePath: string
  trackIndex: number
  type: 'video' | 'audio' | 'image'
  startTime: number
  duration: number
  trimStart: number
  sourceDuration: number
  width?: number
  height?: number
}

function parseExportDimensions(resolution?: string): { width: number; height: number } {
  const match = resolution?.match(/(\d+)x(\d+)/)
  if (!match) return { width: 1920, height: 1080 }
  return {
    width: parseInt(match[1], 10),
    height: parseInt(match[2], 10),
  }
}

function resolveLiveAsset(assets: Asset[], clip: TimelineClip): Asset {
  if (!clip.assetId) return clip.asset!
  return assets.find(asset => asset.id === clip.assetId) || clip.asset!
}

function buildTimelineXmlExportClips(clips: TimelineClip[], assets: Asset[]): TimelineXmlExportClip[] {
  return clips
    .filter(clip => clip.asset && clip.type !== 'adjustment')
    .map(clip => {
      const asset = resolveLiveAsset(assets, clip)
      let filePath = asset.path

      if (asset.takes && asset.takes.length > 0) {
        const takeIndex = clip.takeIndex ?? asset.activeTakeIndex ?? 0
        const take = asset.takes[Math.min(takeIndex, asset.takes.length - 1)]
        filePath = take.path
      }

      const { width, height } = parseExportDimensions(asset.resolution)

      return {
        name: clip.importedName || asset.path?.split(/[/\\]/).pop() || 'clip',
        filePath,
        trackIndex: clip.trackIndex,
        type: clip.type as 'video' | 'image' | 'audio',
        startTime: clip.startTime,
        duration: clip.duration,
        trimStart: clip.trimStart,
        sourceDuration: asset.duration || clip.duration,
        width,
        height,
      }
    })
}

export function useTimelineXmlExport() {
  const activeTimeline = useEditorStore(selectActiveTimeline)
  const clips = useEditorStore(selectClips)
  const assets = useEditorStore(selectAssets)

  const handleExportTimelineXml = useCallback(async () => {
    const timeline = activeTimeline
    if (!timeline) return

    const xmlContent = exportFcp7Xml({
      name: timeline.name,
      fps: 24,
      width: 1920,
      height: 1080,
      clips: buildTimelineXmlExportClips(clips, assets),
    })

    if (window.electronAPI?.showSaveDialog) {
      const filePath = await window.electronAPI.showSaveDialog({
        title: 'Export Timeline as FCP 7 XML',
        defaultPath: `${timeline.name}.xml`,
        filters: [{ name: 'FCP 7 XML', extensions: ['xml'] }],
      })
      if (filePath) {
        await window.electronAPI.saveFile({ filePath, data: xmlContent })
      }
      return
    }

    const blob = new Blob([xmlContent], { type: 'text/xml' })
    const url = URL.createObjectURL(blob)
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = `${timeline.name}.xml`
    anchor.click()
    URL.revokeObjectURL(url)
  }, [activeTimeline, assets, clips])

  return {
    handleExportTimelineXml,
  }
}
