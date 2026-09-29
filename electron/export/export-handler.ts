import path from 'path'
import fs from 'fs'
import os from 'os'
import { getAllowedRoots } from '../config'
import { getMainWindow } from '../window'
import { logger } from '../logger'
import { validatePath } from '../path-validation'
import { findFfmpegPath, runFfmpeg, stopExportProcess } from './ffmpeg-utils'
import { buildDissolveTimeRemap, computeFinalVideoDuration, flattenTimeline } from './timeline'
import { buildVideoFilterGraph } from './video-filter'
import { mixAudioToPcm } from './audio-mix'
import { handle } from '../ipc/typed-handle'
import type { z } from 'zod'
import type { electronAPISchemas } from '../../shared/electron-api-schema'

/** First existing system SANS font, so exported text overlays match the editor
 *  preview's sans look instead of ffmpeg's default serif. Arial first (it's the
 *  preview font stack's fallback). Returns undefined if none found — drawtext
 *  then omits fontfile and falls back to ffmpeg's default (still renders). */
function resolveExportFont(): string | undefined {
  const candidates = process.platform === 'win32'
    ? ['C:/Windows/Fonts/arial.ttf', 'C:/Windows/Fonts/segoeui.ttf', 'C:/Windows/Fonts/calibri.ttf', 'C:/Windows/Fonts/tahoma.ttf']
    : process.platform === 'darwin'
      ? ['/System/Library/Fonts/Supplemental/Arial.ttf', '/Library/Fonts/Arial.ttf', '/System/Library/Fonts/Helvetica.ttc']
      : ['/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf']
  for (const f of candidates) {
    try { if (fs.existsSync(f)) return f } catch { /* ignore */ }
  }
  return undefined
}

export type ExportNativeInput = z.infer<typeof electronAPISchemas.exportNative.input>
export type ExportNativeResult = { success: true } | { success: false; error: string }

/** Render the timeline to a file. `preview` trades quality for speed (ultrafast,
 *  high CRF) — used by the RiX MCP server to render frames for review. */
export async function exportTimelineNative(
  { clips, outputPath, codec, width, height, fps, quality, letterbox, subtitles, textOverlays }: ExportNativeInput,
  { preview = false }: { preview?: boolean } = {},
): Promise<ExportNativeResult> {
  const ffmpegPath = findFfmpegPath()
  if (!ffmpegPath) return { success: false, error: 'FFmpeg not found' }

  try {
    validatePath(outputPath, getAllowedRoots())
    for (const clip of clips) {
      const fp = clip.path
      if (fp) validatePath(fp, getAllowedRoots())
    }
  } catch (err) {
    return { success: false, error: String(err) }
  }

  const segments = flattenTimeline(clips)
  if (segments.length === 0) return { success: false, error: 'No clips to export' }

  for (const seg of segments) {
    if (seg.filePath && !fs.existsSync(seg.filePath)) {
      return { success: false, error: `Source file not found: ${path.basename(seg.filePath)}` }
    }
  }

  // Total program duration drives the progress percentage: ffmpeg reports the
  // encoded position (`time=`), which we divide by this to get a fraction.
  const totalDur = computeFinalVideoDuration(segments)
  const emitProgress = (percent: number, stage: string) => {
    getMainWindow()?.webContents.send('export-progress', {
      percent: Math.max(0, Math.min(100, Math.round(percent))),
      stage,
    })
  }
  // The video encode (step 1) is by far the longest, so it owns most of the
  // bar; audio + mux share the tail. (An h264 mux is a stream copy and flies.)
  const VIDEO_SHARE = 85

  const tmpDir = os.tmpdir()
  const ts = Date.now()
  const tmpVideo = path.join(tmpDir, `ltx-export-video-${ts}.mkv`)
  const tmpAudio = path.join(tmpDir, `ltx-export-audio-${ts}.wav`)
  const cleanup = () => {
    try { fs.unlinkSync(tmpVideo) } catch {}
    try { fs.unlinkSync(tmpAudio) } catch {}
  }

  try {
    logger.info( `[Export] Step 1: Video-only export (${segments.length} segments)`)
    {
      const fontFile = resolveExportFont()
      const { inputs, filterScript } = buildVideoFilterGraph(segments, { width, height, fps, letterbox, subtitles, textOverlays, fontFile })

      const filterFile = path.join(tmpDir, `ltx-filter-v-${ts}.txt`)
      fs.writeFileSync(filterFile, filterScript, 'utf8')

      emitProgress(0, 'Encoding video')
      const r = await runFfmpeg(ffmpegPath, [
        '-y', ...inputs, '-filter_complex_script', filterFile,
        '-map', '[outv]', '-an', '-c:v', 'libx264',
        ...(preview ? ['-preset', 'ultrafast', '-crf', '28'] : ['-preset', 'fast', '-crf', '16']),
        '-pix_fmt', 'yuv420p', tmpVideo
      ], (t) => {
        const frac = totalDur > 0 ? t / totalDur : 0
        emitProgress(frac * VIDEO_SHARE, 'Encoding video')
      })
      try { fs.unlinkSync(filterFile) } catch {}
      if (!r.success) { cleanup(); return { success: false, error: r.error } }
    }

    emitProgress(VIDEO_SHARE, 'Mixing audio')
    logger.info( '[Export] Step 2: Audio mixdown (PCM buffer approach)')
    // A dissolve overlaps two clips, shrinking the program's real duration
    // below the naive sum of clip lengths (see buildDissolveTimeRemap) -
    // every clip's nominal startTime needs the same conversion applied to
    // video, or its audio drifts later relative to the picture with every
    // dissolve that came before it.
    const remapTime = buildDissolveTimeRemap(segments)
    const remappedClips = clips.map(c => ({ ...c, startTime: remapTime(c.startTime) }))

    let totalDuration = computeFinalVideoDuration(segments)
    for (const c of remappedClips) {
      totalDuration = Math.max(totalDuration, c.startTime + c.duration)
    }

    const { pcmBuffer, sampleRate, channels: audioChannels } = await mixAudioToPcm(remappedClips, totalDuration, ffmpegPath)

    const tmpRawPcm = path.join(tmpDir, `ltx-pcm-${ts}.raw`)
    fs.writeFileSync(tmpRawPcm, pcmBuffer)
    logger.info( `[Export] Wrote raw PCM: ${pcmBuffer.length} bytes (${totalDuration.toFixed(2)}s)`)

    {
      const r = await runFfmpeg(ffmpegPath, [
        '-y', '-f', 's16le', '-ar', String(sampleRate), '-ac', String(audioChannels),
        '-i', tmpRawPcm, '-c:a', 'pcm_s16le', tmpAudio,
      ])
      try { fs.unlinkSync(tmpRawPcm) } catch {}
      if (!r.success) { cleanup(); return { success: false, error: r.error } }
    }

    emitProgress(90, 'Finalizing')
    logger.info( '[Export] Step 3: Combining video + audio')
    let videoCodecArgs: string[]
    let audioCodecArgs: string[]
    if (codec === 'h264') {
      videoCodecArgs = ['-c:v', 'libx264', '-preset', 'medium', '-crf', String(quality || 18), '-pix_fmt', 'yuv420p', '-movflags', '+faststart']
      audioCodecArgs = ['-c:a', 'aac', '-b:a', '192k']
    } else if (codec === 'prores') {
      videoCodecArgs = ['-c:v', 'prores_ks', '-profile:v', String(quality || 3), '-pix_fmt', 'yuva444p10le']
      audioCodecArgs = ['-c:a', 'pcm_s16le']
    } else if (codec === 'vp9') {
      videoCodecArgs = ['-c:v', 'libvpx-vp9', '-b:v', `${quality || 8}M`, '-pix_fmt', 'yuv420p']
      audioCodecArgs = ['-c:a', 'libopus', '-b:a', '128k']
    } else {
      cleanup()
      return { success: false, error: `Unknown codec: ${codec}` }
    }

    const canCopyVideo = codec === 'h264'
    const r = await runFfmpeg(ffmpegPath, [
      '-y', '-i', tmpVideo, '-i', tmpAudio,
      '-map', '0:v', '-map', '1:a',
      ...(canCopyVideo ? ['-c:v', 'copy'] : videoCodecArgs),
      ...audioCodecArgs, '-shortest', outputPath
    ], (t) => {
      // Re-encoding codecs (ProRes/VP9) spend real time here; map it to the
      // last 10%. h264 stream-copies and finishes near-instantly.
      const frac = totalDur > 0 ? t / totalDur : 0
      emitProgress(90 + frac * 10, 'Finalizing')
    })

    cleanup()
    if (!r.success) return { success: false, error: r.error }
    emitProgress(100, 'Done')
    logger.info( `[Export] Done: ${outputPath}`)
    return { success: true }
  } catch (err) {
    cleanup()
    return { success: false, error: String(err) }
  }
}

export function registerExportHandlers(): void {
  handle('exportNative', (input) => exportTimelineNative(input))

  handle('exportCancel', () => {
    stopExportProcess()
    return { success: true }
  })
}
