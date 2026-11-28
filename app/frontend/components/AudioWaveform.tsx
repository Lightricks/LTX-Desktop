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

// Global waveform cache: URL → Float32Array of peak amplitudes (one per pixel-bucket)
export const waveformCache = new Map<string, Float32Array>()
const pendingDecodes = new Set<string>()

// Convert a base64 string to an ArrayBuffer
function base64ToArrayBuffer(base64: string): ArrayBuffer {
  const binaryString = atob(base64)
  const bytes = new Uint8Array(binaryString.length)
  for (let i = 0; i < binaryString.length; i++) {
    bytes[i] = binaryString.charCodeAt(i)
  }
  return bytes.buffer
}

// Decode audio file and extract amplitude envelope
export async function computeWaveform(url: string, buckets: number = 800): Promise<Float32Array> {
  if (waveformCache.has(url)) return waveformCache.get(url)!

  if (pendingDecodes.has(url)) {
    while (pendingDecodes.has(url)) {
      await new Promise(r => setTimeout(r, 50))
    }
    if (waveformCache.has(url)) return waveformCache.get(url)!
  }

  pendingDecodes.add(url)
  try {
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

    const channelData = audioBuffer.getChannelData(0)
    const samplesPerBucket = Math.floor(channelData.length / buckets)
    const peaks = new Float32Array(buckets)

    for (let i = 0; i < buckets; i++) {
      let max = 0
      const start = i * samplesPerBucket
      const end = Math.min(start + samplesPerBucket, channelData.length)
      for (let j = start; j < end; j++) {
        const abs = Math.abs(channelData[j])
        if (abs > max) max = abs
      }
      peaks[i] = max
    }

    waveformCache.set(url, peaks)
    return peaks
  } finally {
    pendingDecodes.delete(url)
  }
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
  color?: string
}

export function ClipWaveform({ url, className = '', color = 'rgba(52, 211, 153, 0.7)' }: ClipWaveformProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const [peaks, setPeaks] = useState<Float32Array | null>(null)

  useEffect(() => {
    if (!url) return
    let cancelled = false
    computeWaveform(url, 200).then(p => {
      if (!cancelled) setPeaks(p)
    }).catch(() => {})
    return () => { cancelled = true }
  }, [url])

  const draw = useCallback(() => {
    const canvas = canvasRef.current
    const container = containerRef.current
    if (!canvas || !container || !peaks) return

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

    ctx.fillStyle = color
    ctx.beginPath()
    for (let i = 0; i < w; i++) {
      const peakIdx = Math.floor((i / w) * peaks.length)
      const amp = peaks[Math.min(peakIdx, peaks.length - 1)]
      const y = centerY - amp * maxAmp
      if (i === 0) ctx.moveTo(i, y)
      else ctx.lineTo(i, y)
    }
    for (let i = w - 1; i >= 0; i--) {
      const peakIdx = Math.floor((i / w) * peaks.length)
      const amp = peaks[Math.min(peakIdx, peaks.length - 1)]
      const y = centerY + amp * maxAmp
      ctx.lineTo(i, y)
    }
    ctx.closePath()
    ctx.fill()
  }, [peaks, color])

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
