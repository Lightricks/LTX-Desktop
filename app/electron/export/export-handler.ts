import path from 'path'
import fs from 'fs'
import os from 'os'
import { getAllowedRoots } from '../config'
import { logger } from '../logger'
import { validatePath } from '../path-validation'
import { findFfmpegPath, runFfmpeg, stopExportProcess } from './ffmpeg-utils'
import { flattenTimeline } from './timeline'
import { buildVideoFilterGraph } from './video-filter'
import { mixAudioToPcm } from './audio-mix'
import { handle } from '../ipc/typed-handle'

export function registerExportHandlers(): void {
  handle('exportNative', async ({ clips, outputPath, codec, width, height, fps, quality, letterbox, subtitles }) => {
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
        const { inputs, filterScript } = buildVideoFilterGraph(segments, { width, height, fps, letterbox, subtitles })

        const filterFile = path.join(tmpDir, `ltx-filter-v-${ts}.txt`)
        fs.writeFileSync(filterFile, filterScript, 'utf8')

        const r = await runFfmpeg(ffmpegPath, [
          '-y', ...inputs, '-filter_complex_script', filterFile,
          '-map', '[outv]', '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '16', '-pix_fmt', 'yuv420p', tmpVideo
        ])
        try { fs.unlinkSync(filterFile) } catch {}
        if (!r.success) { cleanup(); return { success: false, error: r.error } }
      }

      logger.info( '[Export] Step 2: Audio mixdown (PCM buffer approach)')
      let totalDuration = segments.reduce((max, s) => Math.max(max, s.startTime + s.duration), 0)
      for (const c of clips) {
        totalDuration = Math.max(totalDuration, c.startTime + c.duration)
      }

      const { pcmBuffer, sampleRate, channels: audioChannels } = await mixAudioToPcm(clips, totalDuration, ffmpegPath)

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
      ])

      cleanup()
      if (!r.success) return { success: false, error: r.error }
      logger.info( `[Export] Done: ${outputPath}`)
      return { success: true }
    } catch (err) {
      cleanup()
      return { success: false, error: String(err) }
    }
  })

  handle('exportCancel', () => {
    stopExportProcess()
    return { success: true }
  })
}
