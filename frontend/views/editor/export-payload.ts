import { DEFAULT_SUBTITLE_STYLE } from '../../types/project-model'
import type { Asset, SubtitleClip, TimelineClip, Track } from '../../types/project-model'
import type { ElectronAPI } from '../../../shared/electron-api-schema'
import { selectClipPathFromAssets } from './editor-selectors'

type ExportNativeInput = Parameters<ElectronAPI['exportNative']>[0]

export type ExportTimelinePayload = Pick<ExportNativeInput, 'clips' | 'subtitles' | 'textOverlays' | 'letterbox'>

const LETTERBOX_RATIO_MAP: Record<string, number> = {
  '2.35:1': 2.35,
  '2.39:1': 2.39,
  '2.76:1': 2.76,
  '1.85:1': 1.85,
  '4:3': 4 / 3,
}

/** Everything about the active timeline that the native ffmpeg export needs, minus
 *  output settings. Shared by the Export modal and the RiX MCP server's renders. */
export function buildExportPayload(
  assets: Asset[],
  clips: TimelineClip[],
  tracks: Track[],
  subtitles: SubtitleClip[],
  { burnSubtitles = true }: { burnSubtitles?: boolean } = {},
): ExportTimelinePayload {
  const exportClips = clips
    .filter(clip => clip.type === 'video' || clip.type === 'image' || clip.type === 'audio')
    .filter(clip => tracks[clip.trackIndex]?.enabled !== false)
    .map(clip => ({
      path: selectClipPathFromAssets(assets, clip),
      type: clip.type,
      startTime: clip.startTime,
      duration: clip.duration,
      trimStart: clip.trimStart,
      speed: clip.speed || 1,
      reversed: clip.reversed || false,
      flipH: clip.flipH || false,
      flipV: clip.flipV || false,
      opacity: clip.opacity ?? 100,
      trackIndex: clip.trackIndex,
      muted: clip.muted || false,
      volume: clip.volume ?? 1,
      audioFadeIn: clip.audioFadeIn ?? 0,
      audioFadeOut: clip.audioFadeOut ?? 0,
      volumeKeyframes: clip.volumeKeyframes,
      colorCorrection: clip.colorCorrection,
      transitionIn: clip.transitionIn,
      transitionOut: clip.transitionOut,
    }))

  const subtitleData = subtitles.map(subtitle => {
    const track = tracks[subtitle.trackIndex]
    return {
      text: subtitle.text,
      startTime: subtitle.startTime,
      endTime: subtitle.endTime,
      style: {
        ...DEFAULT_SUBTITLE_STYLE,
        ...(track?.subtitleStyle || {}),
        ...(subtitle.style || {}),
      },
    }
  })

  // Text-overlay clips (type 'text' with a textStyle) are burned into the export
  // by ffmpeg drawtext — the live preview draws them as DOM, so without this they
  // showed in the editor but were missing from the rendered file.
  const textOverlayData = clips
    .filter(clip => clip.type === 'text' && Boolean(clip.textStyle) && tracks[clip.trackIndex]?.enabled !== false)
    .map(clip => {
      const ts = clip.textStyle!
      return {
        text: ts.text,
        startTime: clip.startTime,
        endTime: clip.startTime + clip.duration,
        fadeIn: clip.textFadeIn ?? 0.5,
        fadeOut: clip.textFadeOut ?? 0.5,
        style: {
          fontSize: ts.fontSize,
          color: ts.color,
          backgroundColor: ts.backgroundColor,
          positionX: ts.positionX,
          positionY: ts.positionY,
          strokeColor: ts.strokeColor,
          strokeWidth: ts.strokeWidth,
          shadowColor: ts.shadowColor,
          shadowOffsetX: ts.shadowOffsetX,
          shadowOffsetY: ts.shadowOffsetY,
          opacity: ts.opacity,
          padding: ts.padding,
          textAlign: ts.textAlign,
        },
      }
    })

  const adjustmentClips = clips.filter(
    clip => clip.type === 'adjustment' && clip.letterbox?.enabled && tracks[clip.trackIndex]?.enabled !== false,
  )
  let letterbox: ExportTimelinePayload['letterbox']
  if (adjustmentClips.length > 0) {
    const best = adjustmentClips.reduce((currentBest, candidate) => (
      candidate.duration > currentBest.duration ? candidate : currentBest
    ))
    const config = best.letterbox!
    letterbox = {
      ratio: config.aspectRatio === 'custom'
        ? (config.customRatio || 2.35)
        : (LETTERBOX_RATIO_MAP[config.aspectRatio] || 2.35),
      color: config.color || '#000000',
      opacity: (config.opacity ?? 100) / 100,
    }
  }

  return {
    clips: exportClips,
    subtitles: burnSubtitles && subtitleData.length > 0 ? subtitleData : undefined,
    textOverlays: textOverlayData.length > 0 ? textOverlayData : undefined,
    letterbox,
  }
}
