import { useRef, useEffect, useState, useCallback } from 'react'
import { Music } from 'lucide-react'
import { logger } from '../lib/logger'

interface AudioClipInfo {
  url: string
  name: string
  startTime: number
  duration: number
}

interface AudioWaveformProps {
  audioClips: AudioClipInfo[]
  currentTime: number
  isPlaying: boolean
}

// High-resolution amplitude envelope of a whole audio file, decoded once per URL.
// Every view (monitor, timeline clips at any zoom) derives what it draws from this.
export interface WaveformEnvelope {
  peak: Float32Array // max |sample| per window
  rms: Float32Array // root-mean-square per window (the "body" of the sound)
  rate: number // windows per second of audio
  duration: number // seconds
}

const ENVELOPE_RATE = 200 // 5ms windows: smooth at any practical timeline zoom

const envelopeCache = new Map<string, WaveformEnvelope>()
const pendingEnvelopes = new Map<string, Promise<WaveformEnvelope>>()

// Global waveform cache: `${url}@${buckets}` → peak amplitudes resampled to `buckets`
export const waveformCache = new Map<string, Float32Array>()

// Convert a base64 string to an ArrayBuffer
function base64ToArrayBuffer(base64: string): ArrayBuffer {
  const binaryString = atob(base64)
  const bytes = new Uint8Array(binaryString.length)
  for (let i = 0; i < binaryString.length; i++) {
    bytes[i] = binaryString.charCodeAt(i)
  }
  return bytes.buffer
}

async function decodeEnvelope(url: string): Promise<WaveformEnvelope> {
  let arrayBuffer: ArrayBuffer

  if (url.startsWith('file://') && (window as any).electronAPI?.readLocalFile) {
    const { data } = await (window as any).electronAPI.readLocalFile({ filePath: url })
    arrayBuffer = base64ToArrayBuffer(data)
  } else {
    const response = await fetch(url)
    arrayBuffer = await response.arrayBuffer()
  }

  const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)()
  const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer)
  audioCtx.close()

  // Mix all channels so a hard-panned stereo track still shows its full shape
  const channels = Array.from({ length: audioBuffer.numberOfChannels }, (_, c) => audioBuffer.getChannelData(c))
  const length = audioBuffer.length
  const samplesPerWindow = Math.max(1, Math.round(audioBuffer.sampleRate / ENVELOPE_RATE))
  const windows = Math.ceil(length / samplesPerWindow)
  const peak = new Float32Array(windows)
  const rms = new Float32Array(windows)

  for (let i = 0; i < windows; i++) {
    const start = i * samplesPerWindow
    const end = Math.min(start + samplesPerWindow, length)
    let max = 0
    let sumSq = 0
    for (let j = start; j < end; j++) {
      let s = 0
      for (const ch of channels) s += ch[j]
      s /= channels.length
      const abs = Math.abs(s)
      if (abs > max) max = abs
      sumSq += s * s
    }
    peak[i] = max
    rms[i] = Math.sqrt(sumSq / Math.max(1, end - start))
  }

  return { peak, rms, rate: audioBuffer.sampleRate / samplesPerWindow, duration: audioBuffer.duration }
}

// Decode (once) and return the high-resolution envelope for a file
export async function getWaveformEnvelope(url: string): Promise<WaveformEnvelope> {
  const cached = envelopeCache.get(url)
  if (cached) return cached
  let pending = pendingEnvelopes.get(url)
  if (!pending) {
    pending = decodeEnvelope(url)
      .then(env => { envelopeCache.set(url, env); return env })
      .finally(() => pendingEnvelopes.delete(url))
    pendingEnvelopes.set(url, pending)
  }
  return pending
}

// Peak amplitudes of the whole file resampled to `buckets` (max-pooled, so short hits survive)
export async function computeWaveform(url: string, buckets: number = 800): Promise<Float32Array> {
  const key = `${url}@${buckets}`
  const cached = waveformCache.get(key)
  if (cached) return cached

  const { peak } = await getWaveformEnvelope(url)
  const peaks = new Float32Array(buckets)
  for (let i = 0; i < buckets; i++) {
    const start = Math.floor((i / buckets) * peak.length)
    const end = Math.max(start + 1, Math.floor(((i + 1) / buckets) * peak.length))
    let max = 0
    for (let j = start; j < end && j < peak.length; j++) if (peak[j] > max) max = peak[j]
    peaks[i] = max
  }

  waveformCache.set(key, peaks)
  return peaks
}

export function AudioWaveform({ audioClips, currentTime, isPlaying }: AudioWaveformProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const [waveforms, setWaveforms] = useState<Map<string, Float32Array>>(new Map())
  const animRef = useRef<number>(0)

  // Load waveform data for all clips
  useEffect(() => {
    let cancelled = false
    const loadAll = async () => {
      const newMap = new Map<string, Float32Array>()
      for (const clip of audioClips) {
        if (!clip.url) continue
        try {
          const peaks = await computeWaveform(clip.url)
          if (cancelled) return
          newMap.set(clip.url, peaks)
        } catch (e) {
          logger.warn(`Failed to decode audio waveform: ${clip.url} ${e}`)
        }
      }
      if (!cancelled) setWaveforms(newMap)
    }
    loadAll()
    return () => { cancelled = true }
  }, [audioClips.map(c => c.url).join(',')])

  // Draw waveform on canvas
  const draw = useCallback(() => {
    const canvas = canvasRef.current
    const container = containerRef.current
    if (!canvas || !container) return

    const dpr = window.devicePixelRatio || 1
    const rect = container.getBoundingClientRect()
    const w = rect.width
    const h = rect.height

    canvas.width = w * dpr
    canvas.height = h * dpr
    canvas.style.width = `${w}px`
    canvas.style.height = `${h}px`

    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.scale(dpr, dpr)

    // Background
    ctx.fillStyle = '#0a0a0a'
    ctx.fillRect(0, 0, w, h)

    // Grid lines (subtle)
    ctx.strokeStyle = 'rgba(255,255,255,0.04)'
    ctx.lineWidth = 1
    const centerY = h / 2
    // Horizontal center line
    ctx.beginPath()
    ctx.moveTo(0, centerY)
    ctx.lineTo(w, centerY)
    ctx.stroke()
    // Quarter lines
    for (const frac of [0.25, 0.75]) {
      ctx.beginPath()
      ctx.moveTo(0, h * frac)
      ctx.lineTo(w, h * frac)
      ctx.stroke()
    }

    if (audioClips.length === 0) return

    // For simplicity, render the first (or longest) audio clip's waveform
    // filling the entire monitor width. If multiple clips, overlay them.
    const maxAmplitude = h * 0.4 // 40% of height above and below center

    for (let ci = 0; ci < audioClips.length; ci++) {
      const clip = audioClips[ci]
      const peaks = waveforms.get(clip.url)
      if (!peaks || peaks.length === 0) continue

      // Map clip's time range to screen
      const clipProgress = (currentTime - clip.startTime) / clip.duration

      // Color: emerald with some alpha for overlapping
      const alpha = audioClips.length > 1 ? 0.6 : 0.9
      const gradient = ctx.createLinearGradient(0, centerY - maxAmplitude, 0, centerY + maxAmplitude)
      gradient.addColorStop(0, `rgba(52, 211, 153, ${alpha})`)   // emerald-400
      gradient.addColorStop(0.5, `rgba(16, 185, 129, ${alpha})`) // emerald-500
      gradient.addColorStop(1, `rgba(52, 211, 153, ${alpha})`)

      // Draw filled waveform (mirrored around center)
      ctx.fillStyle = gradient
      ctx.beginPath()

      // Top half (positive)
      for (let i = 0; i < w; i++) {
        const peakIdx = Math.floor((i / w) * peaks.length)
        const amp = peaks[Math.min(peakIdx, peaks.length - 1)]
        const y = centerY - amp * maxAmplitude
        if (i === 0) ctx.moveTo(i, y)
        else ctx.lineTo(i, y)
      }

      // Bottom half (negative, traced backwards)
      for (let i = w - 1; i >= 0; i--) {
        const peakIdx = Math.floor((i / w) * peaks.length)
        const amp = peaks[Math.min(peakIdx, peaks.length - 1)]
        const y = centerY + amp * maxAmplitude
        ctx.lineTo(i, y)
      }

      ctx.closePath()
      ctx.fill()

      // Played region: brighter overlay
      if (clipProgress > 0 && clipProgress <= 1) {
        const playedX = clipProgress * w
        ctx.save()
        ctx.beginPath()
        ctx.rect(0, 0, playedX, h)
        ctx.clip()

        const brightGradient = ctx.createLinearGradient(0, centerY - maxAmplitude, 0, centerY + maxAmplitude)
        brightGradient.addColorStop(0, 'rgba(52, 211, 153, 0.3)')
        brightGradient.addColorStop(0.5, 'rgba(16, 185, 129, 0.3)')
        brightGradient.addColorStop(1, 'rgba(52, 211, 153, 0.3)')

        ctx.fillStyle = brightGradient
        ctx.beginPath()
        for (let i = 0; i < w; i++) {
          const peakIdx = Math.floor((i / w) * peaks.length)
          const amp = peaks[Math.min(peakIdx, peaks.length - 1)]
          const y = centerY - amp * maxAmplitude
          if (i === 0) ctx.moveTo(i, y)
          else ctx.lineTo(i, y)
        }
        for (let i = w - 1; i >= 0; i--) {
          const peakIdx = Math.floor((i / w) * peaks.length)
          const amp = peaks[Math.min(peakIdx, peaks.length - 1)]
          const y = centerY + amp * maxAmplitude
          ctx.lineTo(i, y)
        }
        ctx.closePath()
        ctx.fill()
        ctx.restore()
      }

      // Playhead line
      if (clipProgress >= 0 && clipProgress <= 1) {
        const px = clipProgress * w
        ctx.strokeStyle = '#ffffff'
        ctx.lineWidth = 1.5
        ctx.beginPath()
        ctx.moveTo(px, 4)
        ctx.lineTo(px, h - 4)
        ctx.stroke()

        // Small triangle at top
        ctx.fillStyle = '#ffffff'
        ctx.beginPath()
        ctx.moveTo(px, 2)
        ctx.lineTo(px - 4, 8)
        ctx.lineTo(px + 4, 8)
        ctx.closePath()
        ctx.fill()
      }
    }

    // If no waveform data loaded yet, show loading indicator
    if (waveforms.size === 0) {
      ctx.fillStyle = 'rgba(255,255,255,0.3)'
      ctx.font = '12px system-ui, sans-serif'
      ctx.textAlign = 'center'
      ctx.fillText('Loading waveform...', w / 2, centerY)
    }
  }, [audioClips, currentTime, waveforms])

  // Animate during playback
  useEffect(() => {
    if (isPlaying) {
      const animate = () => {
        draw()
        animRef.current = requestAnimationFrame(animate)
      }
      animRef.current = requestAnimationFrame(animate)
      return () => cancelAnimationFrame(animRef.current)
    } else {
      draw()
    }
  }, [isPlaying, draw])

  // Redraw on resize
  useEffect(() => {
    const observer = new ResizeObserver(() => draw())
    if (containerRef.current) observer.observe(containerRef.current)
    return () => observer.disconnect()
  }, [draw])

  return (
    <div ref={containerRef} className="w-full h-full flex flex-col">
      {/* Canvas fills available space */}
      <div className="flex-1 relative min-h-0">
        <canvas
          ref={canvasRef}
          className="absolute inset-0 w-full h-full"
        />
        {/* Small music icon badge */}
        <div className="absolute top-3 left-3 flex items-center gap-2 px-2 py-1 rounded bg-black/60">
          <Music className="h-3 w-3 text-emerald-400" />
          <span className="text-[10px] text-emerald-400 font-medium">Audio</span>
        </div>
      </div>
      {/* Clip names */}
      {audioClips.length > 0 && (
        <div className="flex-shrink-0 px-3 py-1.5 bg-zinc-950 border-t border-zinc-800">
          {audioClips.map((clip, i) => (
            <p key={i} className="text-[10px] text-zinc-500 truncate">
              {clip.name}
            </p>
          ))}
        </div>
      )}
    </div>
  )
}

// --- Compact inline waveform for timeline audio clips ---

interface ClipWaveformProps {
  url: string
  className?: string
  // Outer (peak) layer
  color?: string
  // Inner (RMS) layer: the loudness body of the sound
  bodyColor?: string
  // Source window the clip plays. Omit to show the whole file.
  trimStart?: number
  duration?: number
  speed?: number
  reversed?: boolean
}

// Envelope value over [a, b) window indices: max for peaks, power-mean for RMS.
// Sub-window spans are linearly interpolated so zoomed-in waveforms stay smooth.
function sampleEnvelope(data: Float32Array, a: number, b: number, mode: 'max' | 'rms'): number {
  const n = data.length
  if (n === 0) return 0
  if (b - a <= 1) {
    const t = Math.min(Math.max((a + b) / 2 - 0.5, 0), n - 1)
    const i = Math.floor(t)
    const f = t - i
    return data[i] * (1 - f) + data[Math.min(i + 1, n - 1)] * f
  }
  const start = Math.max(0, Math.floor(a))
  const end = Math.min(n, Math.ceil(b))
  let acc = 0
  for (let i = start; i < end; i++) {
    const v = data[i]
    if (mode === 'max') { if (v > acc) acc = v } else acc += v * v
  }
  return mode === 'max' ? acc : Math.sqrt(acc / Math.max(1, end - start))
}

export function ClipWaveform({
  url,
  className = '',
  color = 'rgba(52, 211, 153, 0.45)',
  bodyColor = 'rgba(110, 231, 183, 0.95)',
  trimStart = 0,
  duration,
  speed = 1,
  reversed = false,
}: ClipWaveformProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const [envelope, setEnvelope] = useState<WaveformEnvelope | null>(null)

  useEffect(() => {
    if (!url) return
    let cancelled = false
    getWaveformEnvelope(url).then(env => {
      if (!cancelled) setEnvelope(env)
    }).catch(() => {})
    return () => { cancelled = true }
  }, [url])

  const draw = useCallback(() => {
    const canvas = canvasRef.current
    const container = containerRef.current
    if (!canvas || !container || !envelope) return

    const dpr = window.devicePixelRatio || 1
    const rect = container.getBoundingClientRect()
    const w = rect.width
    const h = rect.height
    if (w === 0 || h === 0) return

    canvas.width = w * dpr
    canvas.height = h * dpr
    canvas.style.width = `${w}px`
    canvas.style.height = `${h}px`

    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.scale(dpr, dpr)
    ctx.clearRect(0, 0, w, h)

    const centerY = h / 2
    const maxAmp = h * 0.45

    // Source seconds this clip plays (in envelope-window units)
    const srcLen = (duration ?? envelope.duration - trimStart) * speed
    const srcStart = trimStart * envelope.rate
    const srcSpan = Math.max(0, srcLen * envelope.rate)
    const cols = Math.max(1, Math.ceil(w))
    const peakCol = new Float32Array(cols)
    const rmsCol = new Float32Array(cols)
    for (let x = 0; x < cols; x++) {
      const fa = x / cols
      const fb = (x + 1) / cols
      const a = srcStart + (reversed ? 1 - fb : fa) * srcSpan
      const b = srcStart + (reversed ? 1 - fa : fb) * srcSpan
      peakCol[x] = sampleEnvelope(envelope.peak, a, b, 'max')
      rmsCol[x] = Math.min(peakCol[x], sampleEnvelope(envelope.rms, a, b, 'rms'))
    }

    const fillMirrored = (amps: Float32Array, style: string) => {
      ctx.fillStyle = style
      ctx.beginPath()
      ctx.moveTo(0, centerY)
      for (let x = 0; x < cols; x++) ctx.lineTo(x + 0.5, centerY - amps[x] * maxAmp)
      ctx.lineTo(cols, centerY)
      for (let x = cols - 1; x >= 0; x--) ctx.lineTo(x + 0.5, centerY + amps[x] * maxAmp)
      ctx.closePath()
      ctx.fill()
    }

    fillMirrored(peakCol, color)
    fillMirrored(rmsCol, bodyColor)

    // Hairline centre so silence still reads as "audio here"
    ctx.fillStyle = bodyColor
    ctx.globalAlpha = 0.35
    ctx.fillRect(0, Math.round(centerY) - 0.5, w, 1)
    ctx.globalAlpha = 1
  }, [envelope, color, bodyColor, trimStart, duration, speed, reversed])

  useEffect(() => {
    draw()
  }, [draw])

  useEffect(() => {
    const observer = new ResizeObserver(() => draw())
    if (containerRef.current) observer.observe(containerRef.current)
    return () => observer.disconnect()
  }, [draw])

  return (
    <div ref={containerRef} className={`absolute inset-0 ${className}`}>
      <canvas ref={canvasRef} className="absolute inset-0 w-full h-full" />
    </div>
  )
}
