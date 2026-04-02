import React, { useState, useRef, useEffect, useCallback } from 'react'

export function VideoThumbnailCard({ videoUrl, thumbnailUrl }: { videoUrl: string; thumbnailUrl?: string }) {
  const containerRef = useRef<HTMLDivElement>(null)
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [isHovering, setIsHovering] = useState(false)
  const [videoReady, setVideoReady] = useState(false)
  const [scrubProgress, setScrubProgress] = useState(0)
  const [scrubTime, setScrubTime] = useState('')
  const rafRef = useRef<number>(0)

  const drawFrame = useCallback(() => {
    const video = videoRef.current
    const canvas = canvasRef.current
    if (!video || !canvas || video.readyState < 2) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    const rect = canvas.getBoundingClientRect()
    if (canvas.width !== rect.width || canvas.height !== rect.height) {
      canvas.width = rect.width
      canvas.height = rect.height
    }
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
  }, [])

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    const video = videoRef.current
    const container = containerRef.current
    if (!video || !container || !video.duration || isNaN(video.duration)) return
    const rect = container.getBoundingClientRect()
    const x = Math.max(0, Math.min(e.clientX - rect.left, rect.width))
    const pct = x / rect.width
    const targetTime = pct * video.duration
    video.currentTime = targetTime
    setScrubProgress(pct)
    const mins = Math.floor(targetTime / 60)
    const secs = Math.floor(targetTime % 60)
    const frames = Math.floor((targetTime % 1) * 24)
    setScrubTime(`${mins}:${secs.toString().padStart(2, '0')}:${frames.toString().padStart(2, '0')}`)
    cancelAnimationFrame(rafRef.current)
    rafRef.current = requestAnimationFrame(drawFrame)
  }, [drawFrame])

  const handleMouseEnter = useCallback(() => {
    setIsHovering(true)
  }, [])

  const handleMouseLeave = useCallback(() => {
    setIsHovering(false)
    setVideoReady(false)
    setScrubProgress(0)
    setScrubTime('')
  }, [])

  useEffect(() => {
    if (!isHovering) return
    const video = document.createElement('video')
    video.muted = true
    video.playsInline = true
    video.preload = 'auto'
    video.src = videoUrl
    videoRef.current = video

    const onLoaded = () => {
      setVideoReady(true)
      requestAnimationFrame(drawFrame)
    }
    const onSeeked = () => requestAnimationFrame(drawFrame)

    video.addEventListener('loadeddata', onLoaded)
    video.addEventListener('seeked', onSeeked)
    video.load()

    return () => {
      video.removeEventListener('loadeddata', onLoaded)
      video.removeEventListener('seeked', onSeeked)
      video.pause()
      video.removeAttribute('src')
      video.load()
      if (videoRef.current === video) {
        videoRef.current = null
      }
    }
  }, [isHovering, videoUrl, drawFrame])

  return (
    <div
      ref={containerRef}
      className="w-full aspect-video relative overflow-hidden bg-zinc-900"
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      onMouseMove={isHovering ? handleMouseMove : undefined}
    >
      {thumbnailUrl ? (
        <img
          src={thumbnailUrl}
          alt=""
          className={`w-full h-full object-cover absolute inset-0 ${isHovering && videoReady ? 'opacity-0' : 'opacity-100'} transition-opacity duration-100`}
        />
      ) : (
        <div className={`w-full h-full absolute inset-0 ${isHovering && videoReady ? 'opacity-0' : 'opacity-100'} transition-opacity duration-100`} />
      )}

      <canvas
        ref={canvasRef}
        className={`w-full h-full object-cover absolute inset-0 ${isHovering && videoReady ? 'opacity-100' : 'opacity-0'} transition-opacity duration-100`}
      />

      {isHovering && (
        <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-black/40">
          <div
            className="h-full bg-blue-500 transition-none"
            style={{ width: `${scrubProgress * 100}%` }}
          />
        </div>
      )}

      {isHovering && videoReady && scrubTime && (
        <div className="absolute bottom-1.5 left-1/2 -translate-x-1/2 px-1.5 py-0.5 rounded bg-black/80 backdrop-blur-sm">
          <span className="text-[9px] text-white font-mono tabular-nums">{scrubTime}</span>
        </div>
      )}
    </div>
  )
}
