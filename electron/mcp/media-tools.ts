import { spawnSync } from 'child_process'
import fs from 'fs'
import os from 'os'
import path from 'path'
import { extractVideoFrameToFile, findFfmpegPath } from '../export/ffmpeg-utils'

// ffmpeg-backed perception helpers for the RiX MCP server: lets the agent
// actually look at clips (sampled frames) and hear music (beats/energy).

const IMAGE_EXT = /\.(png|jpe?g|webp|gif|bmp)$/i
const AUDIO_EXT = /\.(mp3|wav|ogg|aac|flac|m4a)$/i

function ffmpeg(): string {
  const p = findFfmpegPath()
  if (!p) throw new Error('ffmpeg not found (the Python backend environment must be installed)')
  return p
}

export type MediaInfo = {
  kind: 'image' | 'video' | 'audio'
  duration?: number
  width?: number
  height?: number
  fps?: number
  hasAudio: boolean
}

/** Parse ffmpeg's -i banner — cheap, and avoids depending on a separate ffprobe binary. */
export function probeMedia(filePath: string): MediaInfo {
  if (!fs.existsSync(filePath)) throw new Error(`File not found: ${filePath}`)
  const r = spawnSync(ffmpeg(), ['-hide_banner', '-i', filePath], { encoding: 'utf8', timeout: 10000 })
  const out = `${r.stdout || ''}\n${r.stderr || ''}`
  const dur = out.match(/Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)/)
  const videoLine = out.split('\n').find(l => l.includes('Video:'))
  const dims = videoLine?.match(/(\d{2,5})x(\d{2,5})(?:[,\s[]|$)/)
  const fps = videoLine?.match(/(\d+(?:\.\d+)?)\s*fps/)
  const kind: MediaInfo['kind'] = IMAGE_EXT.test(filePath) ? 'image' : AUDIO_EXT.test(filePath) || !videoLine ? 'audio' : 'video'
  return {
    kind,
    duration: dur && kind !== 'image' ? Number(dur[1]) * 3600 + Number(dur[2]) * 60 + parseFloat(dur[3]) : undefined,
    width: dims ? Number(dims[1]) : undefined,
    height: dims ? Number(dims[2]) : undefined,
    fps: fps ? Number(fps[1]) : undefined,
    hasAudio: out.includes('Audio:'),
  }
}

/** Where a file has sound (speech, cackles, SFX) vs silence, via ffmpeg silencedetect:
 *  [start, end] seconds of non-silence at ~0.1s precision. For a dialogue clip these
 *  are the phrases — cut on their boundaries so lines are never clipped mid-word. */
export function soundSegments(filePath: string, duration: number, opts: { noiseDb?: number; minSilence?: number } = {}): Array<[number, number]> {
  const r = spawnSync(ffmpeg(), [
    '-hide_banner', '-i', filePath,
    '-af', `silencedetect=noise=${opts.noiseDb ?? -32}dB:d=${opts.minSilence ?? 0.25}`,
    '-f', 'null', '-',
  ], { encoding: 'utf8', timeout: 60000 })
  const out = `${r.stdout || ''}\n${r.stderr || ''}`
  const r2 = (x: number) => Math.round(x * 100) / 100
  // silencedetect logs silence_start / silence_end in pairs; a trailing silence has no end.
  const starts = [...out.matchAll(/silence_start: (-?[\d.]+)/g)].map(x => Math.max(0, Number(x[1])))
  const ends = [...out.matchAll(/silence_end: ([\d.]+)/g)].map(x => Number(x[1]))
  const silences = starts.map((a, i): [number, number] => [a, ends[i] ?? duration])
  const segments: Array<[number, number]> = []
  let cursor = 0
  for (const [a, b] of silences) {
    if (a - cursor > 0.05) segments.push([r2(cursor), r2(a)])
    cursor = Math.max(cursor, b)
  }
  if (duration - cursor > 0.05) segments.push([r2(cursor), r2(duration)])
  return segments
}

export type Frame = { time?: number; base64: string; mimeType: 'image/jpeg' }

/** Sample frames from a video (evenly spaced, or at explicit times), or a
 *  downscaled copy of a still. JPEG, `width` px wide. */
export function sampleFrames(filePath: string, opts: { count?: number; times?: number[]; width?: number } = {}): { info: MediaInfo & { soundSegments?: Array<[number, number]> }; frames: Frame[] } {
  const probed = probeMedia(filePath)
  const info = probed.kind === 'video' && probed.hasAudio && probed.duration
    ? { ...probed, soundSegments: soundSegments(filePath, probed.duration) }
    : probed
  const width = Math.max(64, Math.min(opts.width ?? 384, 1280))
  const tmp = () => path.join(os.tmpdir(), `rix_mcp_${Date.now()}_${Math.random().toString(36).slice(2, 8)}.jpg`)
  const read = (p: string) => { const b = fs.readFileSync(p).toString('base64'); try { fs.unlinkSync(p) } catch { /* temp */ } return b }

  if (info.kind === 'audio') throw new Error('This is an audio file — use analyze_audio instead')
  if (info.kind === 'image') {
    const out = tmp()
    const r = spawnSync(ffmpeg(), ['-y', '-i', filePath, '-vf', `scale=${width}:-2`, '-frames:v', '1', '-q:v', '4', out], { timeout: 15000 })
    if (r.status !== 0 || !fs.existsSync(out)) throw new Error('Could not decode image')
    return { info, frames: [{ base64: read(out), mimeType: 'image/jpeg' }] }
  }

  const duration = info.duration ?? 0
  const count = Math.max(1, Math.min(opts.count ?? 6, 16))
  const times = (opts.times?.length ? opts.times : Array.from({ length: count }, (_, i) => duration * (i + 0.5) / count))
    .slice(0, 16)
    .map(t => Math.max(0, Math.min(t, Math.max(0, duration - 0.05))))
  const frames = times.map(time => {
    const out = extractVideoFrameToFile({ videoPath: filePath, seekTime: time, width, quality: 4, outputPath: tmp(), timeoutMs: 15000 })
    return { time: Math.round(time * 100) / 100, base64: read(out), mimeType: 'image/jpeg' as const }
  })
  return { info, frames }
}

/**
 * Beat/energy analysis for cutting to music. Decodes to 11.025 kHz mono, builds a
 * half-wave-rectified energy-flux onset envelope (~23 ms hops), estimates tempo by
 * autocorrelation in 70–180 BPM, then phase-aligns a beat grid to the onsets.
 * Heuristic, not a DAW — good enough to place cuts on the pulse.
 */
export function analyzeAudio(filePath: string) {
  const info = probeMedia(filePath)
  if (!info.hasAudio) throw new Error('No audio stream in this file')
  const SR = 11025
  const r = spawnSync(ffmpeg(), ['-v', 'error', '-i', filePath, '-ac', '1', '-ar', String(SR), '-f', 'f32le', '-'], {
    timeout: 60000,
    maxBuffer: 1024 * 1024 * 200,
  })
  if (r.status !== 0 || !r.stdout?.length) throw new Error('Could not decode audio')
  const buf = r.stdout as Buffer
  const samples = new Float32Array(buf.buffer, buf.byteOffset, Math.floor(buf.length / 4))

  const HOP = 256 // ~23 ms
  const n = Math.floor(samples.length / HOP)
  const energy = new Float32Array(n)
  for (let i = 0; i < n; i++) {
    let s = 0
    for (let j = i * HOP; j < (i + 1) * HOP; j++) s += samples[j] * samples[j]
    energy[i] = Math.sqrt(s / HOP)
  }
  const onset = new Float32Array(n)
  for (let i = 1; i < n; i++) onset[i] = Math.max(0, Math.log1p(energy[i] * 100) - Math.log1p(energy[i - 1] * 100))

  const hopSec = HOP / SR
  const minLag = Math.round(60 / 180 / hopSec)
  const maxLag = Math.round(60 / 70 / hopSec)
  let bestLag = minLag
  let bestScore = -Infinity
  for (let lag = minLag; lag <= maxLag; lag++) {
    let score = 0
    for (let i = lag; i < n; i++) score += onset[i] * onset[i - lag]
    score /= (n - lag)
    if (score > bestScore) { bestScore = score; bestLag = lag }
  }
  let bestPhase = 0
  let bestPhaseScore = -Infinity
  for (let phase = 0; phase < bestLag; phase++) {
    let score = 0
    for (let i = phase; i < n; i += bestLag) score += onset[i]
    if (score > bestPhaseScore) { bestPhaseScore = score; bestPhase = phase }
  }
  const r2 = (x: number) => Math.round(x * 100) / 100
  const beats: number[] = []
  for (let i = bestPhase; i < n; i += bestLag) beats.push(r2(i * hopSec))

  // Strongest individual hits (drops, accents): local maxima well above the mean.
  let mean = 0
  for (const v of onset) mean += v
  mean /= n || 1
  let sd = 0
  for (const v of onset) sd += (v - mean) ** 2
  sd = Math.sqrt(sd / (n || 1))
  const peaks: Array<{ t: number; s: number }> = []
  for (let i = 2; i < n - 2; i++) {
    const v = onset[i]
    if (v > mean + 2 * sd && v >= onset[i - 1] && v >= onset[i + 1] && v >= onset[i - 2] && v >= onset[i + 2]) {
      peaks.push({ t: i * hopSec, s: v })
    }
  }
  const strong = peaks.sort((a, b) => b.s - a.s).slice(0, 40).map(p => r2(p.t)).sort((a, b) => a - b)

  // Loudness per second (0–1) to spot intros, builds and drops.
  const perSec = Math.round(1 / hopSec)
  const loud: number[] = []
  for (let i = 0; i < n; i += perSec) {
    let s = 0
    const end = Math.min(n, i + perSec)
    for (let j = i; j < end; j++) s += energy[j]
    loud.push(s / (end - i))
  }
  const peak = Math.max(...loud, 1e-9)

  const totalDur = info.duration ?? samples.length / SR
  return {
    duration: r2(totalDur),
    soundSegments: totalDur <= 60 ? soundSegments(filePath, totalDur) : undefined, // phrases, for dialogue clips
    bpm: r2(60 / (bestLag * hopSec)),
    beatInterval: r2(bestLag * hopSec),
    beats,
    downbeatsGuess: beats.filter((_, i) => i % 4 === 0),
    strongOnsets: strong,
    loudnessPerSecond: loud.map(v => Math.round((v / peak) * 100) / 100),
  }
}
