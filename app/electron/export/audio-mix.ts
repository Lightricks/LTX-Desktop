import { spawn } from 'child_process'
import path from 'path'
import fs from 'fs'
import { logger } from '../logger'
import { fileHasAudio } from './ffmpeg-utils'
import type { ExportClip } from './timeline'

const SAMPLE_RATE = 48000
const NUM_CHANNELS = 2
const BYTES_PER_SAMPLE = 2 // 16-bit signed LE
const BYTES_PER_FRAME = NUM_CHANNELS * BYTES_PER_SAMPLE // 4 bytes per stereo frame

/** Extract raw PCM from a file via ffmpeg stdout pipe */
function extractPcmBuffer(
  ffmpegPath: string,
  filePath: string, trimStart: number, trimEnd: number, speed: number, reversed: boolean
): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    // Build audio filter chain: trim -> reset PTS -> speed -> reverse
    // Using atrim (not -ss/-t) for sample-accurate trimming
    const filters: string[] = [
      `atrim=start=${trimStart.toFixed(6)}:end=${trimEnd.toFixed(6)}`,
      'asetpts=PTS-STARTPTS',
    ]
    if (speed !== 1) {
      // atempo only supports 0.5-100, chain multiple for extreme values
      let remaining = speed
      while (remaining > 2.0) { filters.push('atempo=2.0'); remaining /= 2.0 }
      while (remaining < 0.5) { filters.push('atempo=0.5'); remaining /= 0.5 }
      filters.push(`atempo=${remaining.toFixed(6)}`)
    }
    if (reversed) filters.push('areverse')

    const args = [
      '-i', filePath,
      '-af', filters.join(','),
      '-f', 's16le', '-ac', String(NUM_CHANNELS), '-ar', String(SAMPLE_RATE),
      'pipe:1',
    ]
    const proc = spawn(ffmpegPath, args, { stdio: ['pipe', 'pipe', 'pipe'] })
    const chunks: Buffer[] = []
    proc.stdout?.on('data', (chunk: Buffer) => chunks.push(chunk))
    proc.stderr?.on('data', () => {}) // drain stderr to prevent blocking
    proc.on('close', (code) => {
      if (code === 0) resolve(Buffer.concat(chunks))
      else reject(new Error(`PCM extraction failed (code ${code}) for ${filePath}`))
    })
    proc.on('error', reject)
  })
}

interface AudioSource {
  filePath: string; trimStart: number; trimEnd: number;
  timelineStart: number; speed: number; reversed: boolean; volume: number;
}

/**
 * Mix all audio from clips into a single PCM buffer.
 * Returns raw Int16LE PCM data with the sample rate and channel count.
 */
export async function mixAudioToPcm(
  clips: ExportClip[],
  totalDuration: number,
  ffmpegPath: string,
): Promise<{ pcmBuffer: Buffer; sampleRate: number; channels: number }> {
  // Collect audio sources from ORIGINAL clips
  const audioProbeCache = new Map<string, boolean>()
  const audioSources: AudioSource[] = []

  for (const c of clips) {
    if (c.muted || c.volume <= 0) continue
    const fp = c.path
    if (!fp || !fs.existsSync(fp)) continue

    if (c.type === 'audio') {
      audioSources.push({
        filePath: fp,
        trimStart: c.trimStart,
        trimEnd: c.trimStart + c.duration * c.speed,
        timelineStart: c.startTime,
        speed: c.speed,
        reversed: c.reversed,
        volume: c.volume,
      })
    } else if (c.type === 'video') {
      if (!audioProbeCache.has(fp)) {
        audioProbeCache.set(fp, fileHasAudio(ffmpegPath, fp))
      }
      if (!audioProbeCache.get(fp)) continue
      audioSources.push({
        filePath: fp,
        trimStart: c.trimStart,
        trimEnd: c.trimStart + c.duration * c.speed,
        timelineStart: c.startTime,
        speed: c.speed,
        reversed: c.reversed,
        volume: c.volume,
      })
    }
  }

  logger.info( `[Export] Audio: ${audioSources.length} source(s) from ${clips.length} clip(s)`)

  // Create master mix buffer (Float64 to accumulate without clipping)
  const totalFrames = Math.ceil(totalDuration * SAMPLE_RATE)
  const totalSamples = totalFrames * NUM_CHANNELS
  const mixBuffer = new Float64Array(totalSamples) // initialized to 0 (silence)

  // Extract each source and mix into the master buffer
  for (let i = 0; i < audioSources.length; i++) {
    const src = audioSources[i]
    logger.info( `[Export] Audio ${i + 1}/${audioSources.length}: ${path.basename(src.filePath)} trim=${src.trimStart.toFixed(2)}-${src.trimEnd.toFixed(2)} @${src.timelineStart.toFixed(2)}s vol=${src.volume}`)
    try {
      const pcm = await extractPcmBuffer(ffmpegPath, src.filePath, src.trimStart, src.trimEnd, src.speed, src.reversed)
      const startFrame = Math.round(src.timelineStart * SAMPLE_RATE)
      const startSample = startFrame * NUM_CHANNELS
      const numPcmSamples = Math.floor(pcm.length / BYTES_PER_SAMPLE)

      for (let s = 0; s < numPcmSamples; s++) {
        const destIdx = startSample + s
        if (destIdx < 0 || destIdx >= totalSamples) continue
        const value = pcm.readInt16LE(s * BYTES_PER_SAMPLE)
        mixBuffer[destIdx] += value * src.volume
      }
      logger.info( `[Export] Audio ${i + 1}: mixed ${numPcmSamples} samples (${(numPcmSamples / SAMPLE_RATE / NUM_CHANNELS).toFixed(2)}s) at offset frame ${startFrame}`)
    } catch (err: any) {
      logger.warn( `[Export] Failed to extract audio from ${src.filePath}: ${err.message}`)
    }
  }

  // Convert Float64 accumulator -> Int16 PCM buffer (with clamp)
  const outputPcm = Buffer.alloc(totalFrames * BYTES_PER_FRAME)
  for (let s = 0; s < totalSamples; s++) {
    const clamped = Math.max(-32768, Math.min(32767, Math.round(mixBuffer[s])))
    outputPcm.writeInt16LE(clamped, s * BYTES_PER_SAMPLE)
  }

  return { pcmBuffer: outputPcm, sampleRate: SAMPLE_RATE, channels: NUM_CHANNELS }
}
