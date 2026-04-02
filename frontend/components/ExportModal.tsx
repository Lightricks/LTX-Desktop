import { useState, useRef, useCallback, useEffect, useMemo } from 'react'
import { X, Download, FolderOpen, Film, Package, Loader2, Check, AlertCircle, ChevronDown } from 'lucide-react'
import { Button } from './ui/button'
import { DEFAULT_SUBTITLE_STYLE } from '../types/project'
import type { Track, TimelineClip } from '../types/project'
import {
  selectActiveTimeline,
  selectAssets,
  selectClipPathFromAssets,
  selectClips,
  selectShowExportModal,
  selectSubtitles,
  selectTracks,
} from '../views/editor/editor-selectors'
import { useEditorActions, useEditorStore } from '../views/editor/editor-store'

interface ExportModalProps {
  projectName: string
}

type ExportStatus = 'idle' | 'exporting' | 'done' | 'error'
type ExportCodec = 'h264' | 'prores' | 'vp9'

interface ExportSettings {
  codec: ExportCodec
  width: number
  height: number
  fps: number
  quality: number // CRF for h264, profile for prores, bitrate(Mbps) for vp9
}

const CODEC_INFO: Record<ExportCodec, { label: string; ext: string; description: string; filterName: string }> = {
  h264: { label: 'H.264 / MP4', ext: 'mp4', description: 'Most compatible format', filterName: 'MP4 Video' },
  prores: { label: 'ProRes / MOV', ext: 'mov', description: 'Professional editing format', filterName: 'QuickTime Movie' },
  vp9: { label: 'VP9 / WebM', ext: 'webm', description: 'Web-optimized format', filterName: 'WebM Video' },
}

const RESOLUTIONS = [
  { label: '4K (3840 x 2160)', width: 3840, height: 2160 },
  { label: '1080p (1920 x 1080)', width: 1920, height: 1080 },
  { label: '720p (1280 x 720)', width: 1280, height: 720 },
]

const FRAME_RATES = [24, 25, 30, 60]

const PRORES_PROFILES = [
  { value: 0, label: 'Proxy' },
  { value: 1, label: 'LT' },
  { value: 2, label: 'Standard' },
  { value: 3, label: 'HQ' },
]

const LETTERBOX_RATIO_MAP: Record<string, number> = {
  '2.35:1': 2.35,
  '2.39:1': 2.39,
  '2.76:1': 2.76,
  '1.85:1': 1.85,
  '4:3': 4 / 3,
}

// Generate FCPXML for Premiere / DaVinci
function generateFCPXML(
  clips: TimelineClip[],
  tracks: Track[],
  projectName: string,
  timelineName: string,
  fps: number = 24
): string {
  const frameDuration = `${Math.round(100 * fps)}/${100 * fps}s`
  const totalDuration = clips.reduce((max, c) => Math.max(max, c.startTime + c.duration), 0)
  const totalFrames = Math.ceil(totalDuration * fps)

  const assetEntries: string[] = []
  const seenAssets = new Set<string>()
  for (const clip of clips) {
    const assetId = clip.assetId || clip.id
    if (seenAssets.has(assetId)) continue
    seenAssets.add(assetId)

    const assetPath = clip.asset?.path || ''
    const dur = clip.asset?.duration || clip.duration
    const durFrames = Math.ceil(dur * fps)
    const format = clip.type === 'audio' ? 'audio' : 'video'

    assetEntries.push(
      `        <asset id="${escapeXml(assetId)}" name="${escapeXml(clip.asset?.prompt?.slice(0, 60) || clip.importedName || 'Clip')}" src="${escapeXml(assetPath)}" start="0s" duration="${durFrames}/${fps}s" hasVideo="${format === 'video' ? '1' : '0'}" hasAudio="1" format="r1" />`
    )
  }

  const trackGroups: Map<number, TimelineClip[]> = new Map()
  for (const clip of clips) {
    if (!trackGroups.has(clip.trackIndex)) trackGroups.set(clip.trackIndex, [])
    trackGroups.get(clip.trackIndex)!.push(clip)
  }

  const laneXml: string[] = []
  const sortedTrackIndices = [...trackGroups.keys()].sort((a, b) => a - b)
  
  for (const trackIdx of sortedTrackIndices) {
    const trackClips = trackGroups.get(trackIdx)!.sort((a, b) => a.startTime - b.startTime)
    const clipElements: string[] = []
    
    for (const clip of trackClips) {
      const assetId = clip.assetId || clip.id
      const startFrame = Math.round(clip.startTime * fps)
      const durFrames = Math.round(clip.duration * fps)
      const trimStartFrame = Math.round(clip.trimStart * fps)
      const name = clip.asset?.prompt?.slice(0, 60) || clip.importedName || 'Clip'

      let clipXml = `            <asset-clip ref="${escapeXml(assetId)}" name="${escapeXml(name)}" offset="${startFrame}/${fps}s" duration="${durFrames}/${fps}s" start="${trimStartFrame}/${fps}s"`
      if (clip.speed !== 1) {
        clipXml += ` tcFormat="NDF"`
      }
      clipXml += ` />`
      clipElements.push(clipXml)
    }

    const trackName = tracks[trackIdx]?.name || `Track ${trackIdx + 1}`
    laneXml.push(
      `          <!-- ${escapeXml(trackName)} -->\n` +
      clipElements.join('\n')
    )
  }

  return `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE fcpxml>
<fcpxml version="1.10">
  <resources>
    <format id="r1" name="FFVideoFormat${fps === 24 ? '1080p2398' : '1080p' + fps}" frameDuration="${frameDuration}" width="1920" height="1080" />
${assetEntries.join('\n')}
  </resources>
  <library>
    <event name="${escapeXml(projectName)}">
      <project name="${escapeXml(timelineName)}">
        <sequence format="r1" duration="${totalFrames}/${fps}s" tcStart="0s" tcFormat="NDF">
          <spine>
${laneXml.join('\n')}
          </spine>
        </sequence>
      </project>
    </event>
  </library>
</fcpxml>`
}

function escapeXml(str: string): string {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&apos;')
}

export function ExportModal({ projectName }: ExportModalProps) {
  const { closeExportModal } = useEditorActions()
  const isOpen = useEditorStore(selectShowExportModal)
  const timeline = useEditorStore(selectActiveTimeline)
  const assets = useEditorStore(selectAssets)
  const clips = useEditorStore(selectClips)
  const tracks = useEditorStore(selectTracks)
  const subtitles = useEditorStore(selectSubtitles)

  const exportClips = useMemo(() => (
    clips
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
      }))
  ), [assets, clips, tracks])

  const subtitleData = useMemo(() => (
    subtitles.map(subtitle => {
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
  ), [subtitles, tracks])

  const letterbox = useMemo(() => {
    const adjustmentClips = clips.filter(
      clip =>
        clip.type === 'adjustment'
        && clip.letterbox?.enabled
        && tracks[clip.trackIndex]?.enabled !== false,
    )
    if (adjustmentClips.length === 0) return null
    const best = adjustmentClips.reduce((currentBest, candidate) => (
      candidate.duration > currentBest.duration ? candidate : currentBest
    ))
    const config = best.letterbox!
    return {
      ratio: config.aspectRatio === 'custom'
        ? (config.customRatio || 2.35)
        : (LETTERBOX_RATIO_MAP[config.aspectRatio] || 2.35),
      color: config.color || '#000000',
      opacity: (config.opacity ?? 100) / 100,
    }
  }, [clips, tracks])

  const [exportStatus, setExportStatus] = useState<ExportStatus>('idle')
  const [exportType, setExportType] = useState<'package' | 'video' | null>(null)
  const [exportProgress, setExportProgress] = useState(0)
  const [exportError, setExportError] = useState<string | null>(null)
  const [exportPath, setExportPath] = useState<string | null>(null)
  const [exportFrameInfo, setExportFrameInfo] = useState('')
  const abortRef = useRef(false)

  // Export settings
  const [settings, setSettings] = useState<ExportSettings>({
    codec: 'h264',
    width: 1920,
    height: 1080,
    fps: 24,
    quality: 18, // CRF 18 for h264
  })
  const [burnSubtitles, setBurnSubtitles] = useState(true)

  const closeModal = useCallback(() => {
    closeExportModal()
  }, [closeExportModal])

  const hasSubtitles = subtitleData.length > 0

  useEffect(() => {
    if (!isOpen) return
    setExportStatus('idle')
    setExportType(null)
    setExportProgress(0)
    setExportError(null)
    setExportPath(null)
    setExportFrameInfo('')
    abortRef.current = false
  }, [isOpen])

  // Update quality default when codec changes
  const handleCodecChange = useCallback((codec: ExportCodec) => {
    let quality = 18
    if (codec === 'prores') quality = 3 // HQ profile
    if (codec === 'vp9') quality = 8 // 8 Mbps
    setSettings(prev => ({ ...prev, codec, quality }))
  }, [])

  const handleExportPackage = useCallback(async () => {
    if (!timeline) return
    setExportType('package')
    setExportStatus('exporting')
    setExportProgress(0)
    setExportError(null)

    try {
      const filePath = await window.electronAPI?.showSaveDialog({
        title: 'Export FCPXML Package',
        defaultPath: `${projectName}_${timeline.name}.fcpxml`,
        filters: [
          { name: 'Final Cut Pro XML', extensions: ['fcpxml'] },
          { name: 'All Files', extensions: ['*'] },
        ],
      })

      if (!filePath) {
        setExportStatus('idle')
        return
      }

      setExportProgress(50)
      const xml = generateFCPXML(clips, tracks, projectName, timeline.name)
      const result = await window.electronAPI?.saveFile({ filePath, data: xml })
      if (result?.success) {
        setExportProgress(100)
        setExportPath(filePath)
        setExportStatus('done')
      } else {
        throw new Error(result && !result.success ? result.error : 'Failed to save file')
      }
    } catch (err) {
      setExportError(String(err))
      setExportStatus('error')
    }
  }, [clips, tracks, timeline, projectName])

  const handleExportVideo = useCallback(async () => {
    if (!timeline || clips.length === 0) return
    setExportType('video')
    setExportStatus('exporting')
    setExportProgress(0)
    setExportError(null)
    setExportFrameInfo('Preparing...')
    abortRef.current = false

    try {
      const codecInfo = CODEC_INFO[settings.codec]
      
      const filePath = await window.electronAPI?.showSaveDialog({
        title: `Export ${codecInfo.label}`,
        defaultPath: `${projectName}_${timeline.name}.${codecInfo.ext}`,
        filters: [
          { name: codecInfo.filterName, extensions: [codecInfo.ext] },
          { name: 'All Files', extensions: ['*'] },
        ],
      })

      if (!filePath) {
        setExportStatus('idle')
        return
      }

      // Build clip data for ffmpeg native export (video/image + audio clips)
      setExportFrameInfo('Starting ffmpeg...')

      const result = await window.electronAPI?.exportNative({
        clips: exportClips,
        outputPath: filePath,
        codec: settings.codec,
        width: settings.width,
        height: settings.height,
        fps: settings.fps,
        quality: settings.quality,
        letterbox: letterbox || undefined,
        subtitles: burnSubtitles && subtitleData.length > 0 ? subtitleData : undefined,
      })

      if (result && !result.success) {
        throw new Error(result.error)
      }

      setExportProgress(100)
      setExportPath(filePath)
      setExportFrameInfo('Export complete')
      setExportStatus('done')
    } catch (err) {
      setExportError(String(err))
      setExportStatus('error')
    }
  }, [burnSubtitles, exportClips, letterbox, projectName, settings, subtitleData, timeline])

  const handleCancel = useCallback(async () => {
    abortRef.current = true
    window.electronAPI?.exportCancel({ sessionId: 'current' }).catch(() => {})
    setExportStatus('idle')
  }, [])

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4" onClick={closeModal}>
      <div 
        className="bg-zinc-900 rounded-2xl border border-zinc-700/50 shadow-2xl w-full max-w-lg relative overflow-hidden max-h-[calc(100vh-2rem)] flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-zinc-800">
          <h2 className="text-lg font-bold text-white">Export</h2>
          <button
            onClick={closeModal}
            className="p-1.5 rounded-lg text-zinc-500 hover:text-white hover:bg-zinc-800 transition-colors"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        <div className="p-6 overflow-y-auto flex-1">
          {/* Exporting state */}
          {exportStatus === 'exporting' && (
            <div className="space-y-4">
              <div className="flex items-center gap-3">
                <Loader2 className="h-5 w-5 text-blue-400 animate-spin" />
                <span className="text-sm text-zinc-300">
                  {exportType === 'package' ? 'Generating FCPXML...' : 'Rendering video...'}
                </span>
              </div>
              <div className="w-full bg-zinc-800 rounded-full h-2 overflow-hidden">
                <div 
                  className="h-full bg-blue-500 rounded-full transition-all duration-300"
                  style={{ width: `${exportProgress}%` }}
                />
              </div>
              <div className="flex items-center justify-between">
                <p className="text-xs text-zinc-500">{exportProgress}% complete</p>
                {exportFrameInfo && <p className="text-xs text-zinc-500">{exportFrameInfo}</p>}
              </div>
              {exportType === 'video' && (
                <Button
                  variant="outline"
                  size="sm"
                  className="border-zinc-700 text-zinc-400"
                  onClick={handleCancel}
                >
                  Cancel
                </Button>
              )}
            </div>
          )}

          {/* Done state */}
          {exportStatus === 'done' && (
            <div className="space-y-4">
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-full bg-green-500/20 flex items-center justify-center">
                  <Check className="h-5 w-5 text-green-400" />
                </div>
                <div>
                  <p className="text-sm text-white font-medium">Export complete</p>
                  <p className="text-xs text-zinc-500 truncate max-w-[340px]">{exportPath}</p>
                  {exportFrameInfo && <p className="text-xs text-zinc-500">{exportFrameInfo}</p>}
                </div>
              </div>
              <div className="flex gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  className="border-zinc-700 text-zinc-300"
                  onClick={() => {
                    if (exportPath) {
                      window.electronAPI?.openParentFolderOfFile({ filePath: exportPath })
                    }
                  }}
                >
                  <FolderOpen className="h-4 w-4 mr-2" />
                  Show in Folder
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  className="border-zinc-700 text-zinc-300"
                  onClick={() => {
                    setExportStatus('idle')
                    setExportType(null)
                  }}
                >
                  Export Another
                </Button>
              </div>
            </div>
          )}

          {/* Error state */}
          {exportStatus === 'error' && (
            <div className="space-y-4">
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-full bg-red-500/20 flex items-center justify-center">
                  <AlertCircle className="h-5 w-5 text-red-400" />
                </div>
                <div>
                  <p className="text-sm text-white font-medium">Export failed</p>
                  <p className="text-xs text-red-400 max-w-[340px] break-words">{exportError}</p>
                </div>
              </div>
              <Button
                variant="outline"
                size="sm"
                className="border-zinc-700 text-zinc-300"
                onClick={() => {
                  setExportStatus('idle')
                  setExportType(null)
                }}
              >
                Try Again
              </Button>
            </div>
          )}

          {/* Idle state — settings */}
          {exportStatus === 'idle' && (
            <div className="space-y-5">
              {/* Package export (compact) */}
              <button
                onClick={handleExportPackage}
                className="w-full flex items-center gap-3 p-3 rounded-xl border border-zinc-700/50 bg-zinc-800/50 hover:bg-zinc-800 hover:border-zinc-600 transition-all group"
              >
                <div className="w-10 h-10 rounded-lg bg-zinc-700/50 flex items-center justify-center flex-shrink-0">
                  <Package className="h-5 w-5 text-zinc-300" />
                </div>
                <div className="flex-1 text-left">
                  <p className="text-sm font-semibold text-white">Package (FCPXML)</p>
                  <p className="text-[10px] text-zinc-500">For Premiere Pro &amp; DaVinci Resolve</p>
                </div>
                <div className="flex items-center gap-1.5 flex-shrink-0">
                  <div className="w-6 h-6 rounded bg-zinc-700 flex items-center justify-center" title="DaVinci Resolve">
                    <span className="text-[8px] font-bold text-orange-400">DR</span>
                  </div>
                  <div className="w-6 h-6 rounded bg-zinc-700 flex items-center justify-center" title="Premiere Pro">
                    <span className="text-[8px] font-bold text-blue-400">Pr</span>
                  </div>
                  <Download className="h-4 w-4 text-zinc-500 group-hover:text-zinc-300 transition-colors ml-1" />
                </div>
              </button>

              {/* Divider */}
              <div className="flex items-center gap-3">
                <div className="flex-1 h-px bg-zinc-800" />
                <span className="text-[10px] text-zinc-600 uppercase tracking-wider font-semibold">Video Export</span>
                <div className="flex-1 h-px bg-zinc-800" />
              </div>

              {/* Format selector */}
              <div>
                <label className="text-[10px] text-zinc-500 uppercase tracking-wider font-semibold mb-2 block">Format</label>
                <div className="grid grid-cols-3 gap-2">
                  {(Object.keys(CODEC_INFO) as ExportCodec[]).map(codec => (
                    <button
                      key={codec}
                      onClick={() => handleCodecChange(codec)}
                      className={`p-2.5 rounded-lg border text-center transition-all ${
                        settings.codec === codec
                          ? 'border-blue-500 bg-blue-500/10 text-white'
                          : 'border-zinc-700 bg-zinc-800/50 text-zinc-400 hover:border-zinc-600 hover:text-zinc-300'
                      }`}
                    >
                      <p className="text-xs font-semibold">{CODEC_INFO[codec].label.split(' / ')[0]}</p>
                      <p className="text-[9px] text-zinc-500 mt-0.5">.{CODEC_INFO[codec].ext}</p>
                    </button>
                  ))}
                </div>
              </div>

              {/* Resolution & Frame rate row */}
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="text-[10px] text-zinc-500 uppercase tracking-wider font-semibold mb-1.5 block">Resolution</label>
                  <div className="relative">
                    <select
                      value={`${settings.width}x${settings.height}`}
                      onChange={(e) => {
                        const [w, h] = e.target.value.split('x').map(Number)
                        setSettings(prev => ({ ...prev, width: w, height: h }))
                      }}
                      className="w-full appearance-none bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500 pr-8 cursor-pointer"
                    >
                      {RESOLUTIONS.map(r => (
                        <option key={`${r.width}x${r.height}`} value={`${r.width}x${r.height}`}>
                          {r.label}
                        </option>
                      ))}
                    </select>
                    <ChevronDown className="absolute right-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-zinc-500 pointer-events-none" />
                  </div>
                </div>
                <div>
                  <label className="text-[10px] text-zinc-500 uppercase tracking-wider font-semibold mb-1.5 block">Frame Rate</label>
                  <div className="relative">
                    <select
                      value={settings.fps}
                      onChange={(e) => setSettings(prev => ({ ...prev, fps: parseInt(e.target.value) }))}
                      className="w-full appearance-none bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500 pr-8 cursor-pointer"
                    >
                      {FRAME_RATES.map(fps => (
                        <option key={fps} value={fps}>{fps} fps</option>
                      ))}
                    </select>
                    <ChevronDown className="absolute right-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-zinc-500 pointer-events-none" />
                  </div>
                </div>
              </div>

              {/* Quality */}
              <div>
                <label className="text-[10px] text-zinc-500 uppercase tracking-wider font-semibold mb-1.5 block">Quality</label>
                {settings.codec === 'h264' && (
                  <div className="flex items-center gap-3">
                    <input
                      type="range"
                      min={15}
                      max={28}
                      step={1}
                      value={settings.quality}
                      onChange={(e) => setSettings(prev => ({ ...prev, quality: parseInt(e.target.value) }))}
                      className="flex-1 h-1.5 accent-blue-500 cursor-pointer"
                      // Note: lower CRF = higher quality (inverted display)
                    />
                    <span className="text-xs text-zinc-400 w-16 text-right">
                      {settings.quality <= 18 ? 'High' : settings.quality <= 23 ? 'Medium' : 'Low'}
                      <span className="text-zinc-600 ml-1">({settings.quality})</span>
                    </span>
                  </div>
                )}
                {settings.codec === 'prores' && (
                  <div className="grid grid-cols-4 gap-1.5">
                    {PRORES_PROFILES.map(p => (
                      <button
                        key={p.value}
                        onClick={() => setSettings(prev => ({ ...prev, quality: p.value }))}
                        className={`py-1.5 px-2 rounded-md text-xs font-medium transition-all ${
                          settings.quality === p.value
                            ? 'bg-blue-500/20 border border-blue-500 text-blue-300'
                            : 'bg-zinc-800 border border-zinc-700 text-zinc-400 hover:border-zinc-600'
                        }`}
                      >
                        {p.label}
                      </button>
                    ))}
                  </div>
                )}
                {settings.codec === 'vp9' && (
                  <div className="flex items-center gap-3">
                    <input
                      type="range"
                      min={2}
                      max={20}
                      step={1}
                      value={settings.quality}
                      onChange={(e) => setSettings(prev => ({ ...prev, quality: parseInt(e.target.value) }))}
                      className="flex-1 h-1.5 accent-blue-500 cursor-pointer"
                    />
                    <span className="text-xs text-zinc-400 w-20 text-right">
                      {settings.quality} Mbps
                    </span>
                  </div>
                )}
              </div>

              {/* Options */}
              {hasSubtitles && (
                <div className="space-y-2">
                  <div className="flex items-center gap-3">
                    <div className="flex-1 h-px bg-zinc-800" />
                    <span className="text-[10px] text-zinc-600 uppercase tracking-wider font-semibold">Options</span>
                    <div className="flex-1 h-px bg-zinc-800" />
                  </div>
                  <label className="flex items-center gap-2.5 cursor-pointer group">
                    <input
                      type="checkbox"
                      checked={burnSubtitles}
                      onChange={(e) => setBurnSubtitles(e.target.checked)}
                      className="w-4 h-4 rounded border-zinc-600 bg-zinc-800 accent-blue-500 cursor-pointer"
                    />
                    <span className="text-xs text-zinc-300 group-hover:text-white transition-colors">Burn-in subtitles</span>
                  </label>
                </div>
              )}

              {/* Export button */}
              <button
                onClick={handleExportVideo}
                disabled={clips.length === 0}
                className="w-full py-3 rounded-xl bg-blue-600 hover:bg-blue-500 disabled:opacity-40 disabled:cursor-not-allowed text-white font-semibold text-sm flex items-center justify-center gap-2 transition-colors"
              >
                <Film className="h-4 w-4" />
                Export Video
              </button>

              {clips.length === 0 && (
                <p className="text-xs text-zinc-500 text-center">Add clips to the timeline to export.</p>
              )}
            </div>
          )}
        </div>

      </div>
    </div>
  )
}
