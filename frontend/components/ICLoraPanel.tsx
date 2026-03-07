import { useState, useRef, useEffect, useCallback } from 'react'
import {
  X, Play, Pause, Upload, Loader2, Film, Sparkles,
  FolderOpen, ChevronDown, RefreshCw, Settings, Download, Check, AlertCircle,
} from 'lucide-react'
import { backendFetch } from '../lib/backend'
import { logger } from '../lib/logger'

interface ICLoraModel {
  name: string
  path: string
  conditioning_type: string
  reference_downscale_factor: number
}

interface ICLoraPanelProps {
  isOpen: boolean
  onClose: () => void
  initialVideoUrl?: string
  initialVideoPath?: string
  initialClipName?: string
  sourceClipId?: string | null
  onResult: (result: { videoPath: string; sourceClipId: string | null }) => void
}

type ConditioningType = 'canny' | 'depth'

const CONDITIONING_TYPES: { value: ConditioningType; label: string; desc: string }[] = [
  { value: 'canny', label: 'Canny Edges', desc: 'Edge detection' },
  { value: 'depth', label: 'Depth Map', desc: 'Estimated depth' },
]

const OFFICIAL_IC_LORA_MODELS = [
  {
    id: 'canny',
    label: 'Canny Control',
    repo_id: 'Lightricks/LTX-2-19b-IC-LoRA-Canny-Control',
    filename: 'ltx-2-19b-ic-lora-canny-control.safetensors',
  },
  {
    id: 'depth',
    label: 'Depth Control',
    repo_id: 'Lightricks/LTX-2-19b-IC-LoRA-Depth-Control',
    filename: 'ltx-2-19b-ic-lora-depth-control.safetensors',
  },
  {
    id: 'pose',
    label: 'Pose Control',
    repo_id: 'Lightricks/LTX-2-19b-IC-LoRA-Pose-Control',
    filename: 'ltx-2-19b-ic-lora-pose-control.safetensors',
  },
  {
    id: 'detailer',
    label: 'Video Detailer',
    repo_id: 'Lightricks/LTX-2-19b-IC-LoRA-Detailer',
    filename: 'ltx-2-19b-ic-lora-detailer.safetensors',
  },
]

export function ICLoraPanel({
  isOpen,
  onClose,
  initialVideoUrl,
  initialVideoPath,
  sourceClipId,
  onResult,
}: ICLoraPanelProps) {
  // Close on Escape
  useEffect(() => {
    if (!isOpen) return
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') { onClose(); e.stopPropagation() }
    }
    document.addEventListener('keydown', handleKey)
    return () => document.removeEventListener('keydown', handleKey)
  }, [isOpen, onClose])

  // Video state
  const inputVideoRef = useRef<HTMLVideoElement>(null)
  const outputVideoRef = useRef<HTMLVideoElement>(null)
  const [inputVideoUrl, setInputVideoUrl] = useState(initialVideoUrl || '')
  const [inputVideoPath, setInputVideoPath] = useState(initialVideoPath || '')
  const [inputPlaying, setInputPlaying] = useState(false)
  const [inputTime, setInputTime] = useState(0)
  const [inputDuration, setInputDuration] = useState(0)

  // Conditioning preview
  const [conditioningType, setConditioningType] = useState<ConditioningType>('canny')
  const [conditioningPreview, setConditioningPreview] = useState<string | null>(null)
  const [isExtracting, setIsExtracting] = useState(false)

  // LoRA models
  const [models, setModels] = useState<ICLoraModel[]>([])
  const [selectedModel, setSelectedModel] = useState<ICLoraModel | null>(null)
  const [showModelDropdown, setShowModelDropdown] = useState(false)
  // Download state
  const [downloadingModels, setDownloadingModels] = useState<Record<string, 'downloading' | 'done' | 'error'>>({})

  // Reference image(s)
  const [refImages, setRefImages] = useState<{ path: string; url: string; frame: number; strength: number }[]>([])

  // Generation params
  const [prompt, setPrompt] = useState('')
  const [conditioningStrength, setConditioningStrength] = useState(1.0)
  const [seed, setSeed] = useState(42)
  // Note: IC-LoRA pipeline Stage 1 runs at this res, Stage 2 upsamples 2x
  // So 512x352 → output 1024x704, 608x352 → output 1216x704
  const [width, setWidth] = useState(608)
  const [height, setHeight] = useState(352)
  const [numFrames, setNumFrames] = useState(121)
  const [frameRate, setFrameRate] = useState(24)

  // Generation state
  const [isGenerating, setIsGenerating] = useState(false)
  const [generationStatus, setGenerationStatus] = useState('')
  const [generationError, setGenerationError] = useState<string | null>(null)
  const [outputVideoUrl, setOutputVideoUrl] = useState<string | null>(null)
  const [outputVideoPath, setOutputVideoPath] = useState<string | null>(null)
  const [showSettings, setShowSettings] = useState(false)

  // Reset when opened
  useEffect(() => {
    if (isOpen) {
      setInputVideoUrl(initialVideoUrl || '')
      setInputVideoPath(initialVideoPath || '')
      setConditioningPreview(null)
      setOutputVideoUrl(null)
      setOutputVideoPath(null)
      setIsGenerating(false)
      setGenerationStatus('')
      setPrompt('')
      setRefImages([])
      fetchModels()
    }
  }, [isOpen, initialVideoUrl, initialVideoPath])

  // Fetch available models
  const fetchModels = useCallback(async () => {
    try {
      const resp = await backendFetch('/api/ic-lora/list-models')
      if (resp.ok) {
        const data = await resp.json()
        setModels(data.models || [])
        if (data.models?.length > 0 && !selectedModel) {
          setSelectedModel(data.models[0])
        }
      }
    } catch (e) {
      logger.warn(`Failed to fetch IC-LoRA models: ${e}`)
    }
  }, [selectedModel])

  // Extract conditioning preview when time or type changes
  const extractConditioning = useCallback(async () => {
    if (!inputVideoPath || isExtracting) return
    setIsExtracting(true)
    try {
      const resp = await backendFetch('/api/ic-lora/extract-conditioning', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          video_path: inputVideoPath,
          conditioning_type: conditioningType,
          frame_time: inputTime,
        }),
      })
      if (resp.ok) {
        const data = await resp.json()
        setConditioningPreview(data.conditioning)
      }
    } catch (e) {
      logger.warn(`Failed to extract conditioning: ${e}`)
    } finally {
      setIsExtracting(false)
    }
  }, [inputVideoPath, conditioningType, inputTime, isExtracting])

  // Debounced conditioning extraction on time/type change
  const extractTimerRef = useRef<NodeJS.Timeout | null>(null)
  useEffect(() => {
    if (!inputVideoPath) return
    if (extractTimerRef.current) clearTimeout(extractTimerRef.current)
    extractTimerRef.current = setTimeout(() => {
      extractConditioning()
    }, 300)
    return () => { if (extractTimerRef.current) clearTimeout(extractTimerRef.current) }
  }, [inputTime, conditioningType, inputVideoPath])

  // Video time tracking
  useEffect(() => {
    const video = inputVideoRef.current
    if (!video) return
    const onTime = () => setInputTime(video.currentTime)
    const onLoaded = () => setInputDuration(video.duration || 0)
    video.addEventListener('timeupdate', onTime)
    video.addEventListener('loadedmetadata', onLoaded)
    return () => {
      video.removeEventListener('timeupdate', onTime)
      video.removeEventListener('loadedmetadata', onLoaded)
    }
  }, [inputVideoUrl])

  const toggleInputPlay = useCallback(() => {
    const v = inputVideoRef.current
    if (!v) return
    if (v.paused) { v.play(); setInputPlaying(true) }
    else { v.pause(); setInputPlaying(false) }
  }, [])

  // Browse for a video file
  const handleImportVideo = useCallback(async () => {
    const paths = await window.electronAPI.showOpenFileDialog({
      title: 'Select Driving Video',
      filters: [{ name: 'Video', extensions: ['mp4', 'mov', 'avi', 'webm', 'mkv'] }],
    })
    if (paths && paths.length > 0) {
      const filePath = paths[0]
      const normalized = filePath.replace(/\\/g, '/')
      const url = normalized.startsWith('/') ? `file://${normalized}` : `file:///${normalized}`
      setInputVideoPath(filePath)
      setInputVideoUrl(url)
      setConditioningPreview(null)
      setOutputVideoUrl(null)
    }
  }, [])

  // Import a reference image
  const handleImportImage = useCallback(async () => {
    const paths = await window.electronAPI.showOpenFileDialog({
      title: 'Select Reference Image',
      filters: [{ name: 'Images', extensions: ['png', 'jpg', 'jpeg', 'webp', 'bmp'] }],
    })
    if (paths && paths.length > 0) {
      const filePath = paths[0]
      const normalized = filePath.replace(/\\/g, '/')
      const url = normalized.startsWith('/') ? `file://${normalized}` : `file:///${normalized}`
      setRefImages(prev => [...prev, { path: filePath, url, frame: 0, strength: 1.0 }])
    }
  }, [])

  // Remove a reference image
  const handleRemoveImage = useCallback((idx: number) => {
    setRefImages(prev => prev.filter((_, i) => i !== idx))
  }, [])

  // Update reference image params
  const handleUpdateImage = useCallback((idx: number, updates: Partial<{ frame: number; strength: number }>) => {
    setRefImages(prev => prev.map((img, i) => i === idx ? { ...img, ...updates } : img))
  }, [])

  // Download an official IC-LoRA model
  const handleDownloadModel = useCallback(async (modelDef: typeof OFFICIAL_IC_LORA_MODELS[0]) => {
    if (downloadingModels[modelDef.id] === 'downloading') return
    setDownloadingModels(prev => ({ ...prev, [modelDef.id]: 'downloading' }))
    try {
      const resp = await backendFetch('/api/ic-lora/download-model', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: modelDef.id }),
      })
      const data = await resp.json()
      if (resp.ok && (data.status === 'complete')) {
        setDownloadingModels(prev => ({ ...prev, [modelDef.id]: 'done' }))
        // Refresh the model list
        fetchModels()
      } else {
        setDownloadingModels(prev => ({ ...prev, [modelDef.id]: 'error' }))
      }
    } catch {
      setDownloadingModels(prev => ({ ...prev, [modelDef.id]: 'error' }))
    }
  }, [downloadingModels])

  // Browse for a custom LoRA file
  const handleBrowseLora = useCallback(async () => {
    const paths = await window.electronAPI.showOpenFileDialog({
      title: 'Select IC-LoRA Model',
      filters: [{ name: 'SafeTensors', extensions: ['safetensors'] }],
    })
    if (paths && paths.length > 0) {
      const custom: ICLoraModel = {
        name: paths[0].split(/[/\\]/).pop()?.replace('.safetensors', '') || 'Custom LoRA',
        path: paths[0],
        conditioning_type: 'unknown',
        reference_downscale_factor: 1,
      }
      setSelectedModel(custom)
      setShowModelDropdown(false)
    }
  }, [])

  // Generate
  const handleGenerate = useCallback(async () => {
    if (!inputVideoPath || !selectedModel || isGenerating || !prompt.trim()) return

    setIsGenerating(true)
    setGenerationStatus('Loading IC-LoRA pipeline...')
    setGenerationError(null)
    setOutputVideoUrl(null)
    setOutputVideoPath(null)

    try {
      setGenerationStatus('Generating video with IC-LoRA...')
      const resp = await backendFetch('/api/ic-lora/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          video_path: inputVideoPath,
          lora_path: selectedModel.path,
          conditioning_type: conditioningType,
          prompt,
          conditioning_strength: conditioningStrength,
          seed,
          height,
          width,
          num_frames: numFrames,
          frame_rate: frameRate,
          images: refImages.map(img => ({ path: img.path, frame: img.frame, strength: img.strength })),
        }),
      })

      const data = await resp.json()
      if (resp.ok && data.status === 'complete' && data.video_path) {
        const pathNorm = data.video_path.replace(/\\/g, '/')
        const url = pathNorm.startsWith('/') ? `file://${pathNorm}` : `file:///${pathNorm}`
        setOutputVideoUrl(url)
        setOutputVideoPath(data.video_path)
        setGenerationStatus('Generation complete!')
      } else {
        const errorMsg = data.error || 'Unknown error'
        setGenerationStatus(`Error: ${errorMsg}`)
        setGenerationError(errorMsg)
      }
    } catch (e) {
      setGenerationStatus(`Error: ${(e as Error).message}`)
      setGenerationError((e as Error).message)
    } finally {
      setIsGenerating(false)
    }
  }, [inputVideoPath, selectedModel, prompt, conditioningStrength, seed, height, width, numFrames, frameRate, isGenerating])

  // Accept the output
  const handleAcceptOutput = useCallback(() => {
    if (outputVideoPath) {
      onResult({ videoPath: outputVideoPath, sourceClipId: sourceClipId || null })
    }
  }, [outputVideoPath, sourceClipId, onResult])

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 z-[200] flex items-center justify-center bg-black/80 backdrop-blur-sm">
      <div className="bg-zinc-900 rounded-xl border border-zinc-700 shadow-2xl flex flex-col overflow-hidden"
        style={{ width: '90vw', maxWidth: '1400px', height: '85vh', maxHeight: '900px' }}
        onClick={e => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-3 border-b border-zinc-800 flex-shrink-0">
          <div className="flex items-center gap-3">
            <div className="w-7 h-7 rounded-lg bg-blue-600/20 flex items-center justify-center">
              <Sparkles className="h-4 w-4 text-blue-400" />
            </div>
            <div>
              <h2 className="text-sm font-semibold text-white">IC-LoRA / Style Transfer</h2>
              <p className="text-[10px] text-zinc-500">Video-to-video generation with conditioning LoRAs</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setShowSettings(!showSettings)}
              className={`p-1.5 rounded hover:bg-zinc-800 transition-colors ${showSettings ? 'text-blue-400' : 'text-zinc-400 hover:text-white'}`}
            >
              <Settings className="h-4 w-4" />
            </button>
            <button
              onClick={onClose}
              disabled={isGenerating}
              className="p-1.5 rounded hover:bg-zinc-800 text-zinc-400 hover:text-white transition-colors disabled:opacity-50"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
        </div>

        {/* Main content: three columns */}
        <div className="flex-1 flex min-h-0 overflow-hidden">
          {/* Left: Input Video */}
          <div className="flex-1 flex flex-col border-r border-zinc-800 min-w-0">
            <div className="px-3 py-2 border-b border-zinc-800 flex items-center justify-between">
              <span className="text-[11px] font-semibold text-zinc-400 uppercase tracking-wider">Input Video</span>
              <button
                onClick={handleImportVideo}
                className="flex items-center gap-1 px-2 py-1 rounded text-[10px] text-zinc-400 hover:text-white hover:bg-zinc-800 transition-colors"
              >
                <Upload className="h-3 w-3" />
                Import
              </button>
            </div>
            <div className="flex-1 bg-black flex items-center justify-center min-h-0 relative">
              {inputVideoUrl ? (
                <video
                  ref={inputVideoRef}
                  src={inputVideoUrl}
                  className="max-w-full max-h-full object-contain"
                  onClick={toggleInputPlay}
                  onEnded={() => setInputPlaying(false)}
                />
              ) : (
                <div className="text-center p-4">
                  <div className="w-12 h-12 rounded-full bg-zinc-800 flex items-center justify-center mx-auto mb-2">
                    <Film className="h-6 w-6 text-zinc-600" />
                  </div>
                  <p className="text-zinc-500 text-xs">No video selected</p>
                  <button
                    onClick={handleImportVideo}
                    className="mt-2 px-3 py-1.5 text-[10px] text-blue-400 border border-blue-500/30 rounded-lg hover:bg-blue-600/10 transition-colors"
                  >
                    Import Video
                  </button>
                </div>
              )}
            </div>
            {inputVideoUrl && (
              <div className="px-3 py-1.5 border-t border-zinc-800 flex items-center gap-2">
                <button onClick={toggleInputPlay} className="p-1 rounded hover:bg-zinc-800 text-white">
                  {inputPlaying ? <Pause className="h-3 w-3" /> : <Play className="h-3 w-3" />}
                </button>
                <div className="flex-1 h-1 bg-zinc-800 rounded-full relative cursor-pointer"
                  onClick={(e) => {
                    const rect = e.currentTarget.getBoundingClientRect()
                    const frac = (e.clientX - rect.left) / rect.width
                    if (inputVideoRef.current) inputVideoRef.current.currentTime = frac * inputDuration
                  }}
                >
                  <div className="h-full bg-blue-500 rounded-full" style={{ width: `${inputDuration > 0 ? (inputTime / inputDuration) * 100 : 0}%` }} />
                </div>
                <span className="text-[10px] font-mono text-zinc-500 min-w-[60px] text-right">
                  {inputTime.toFixed(1)}s / {inputDuration.toFixed(1)}s
                </span>
              </div>
            )}
          </div>

          {/* Center: Conditioning Preview */}
          <div className="flex-1 flex flex-col border-r border-zinc-800 min-w-0">
            <div className="px-3 py-2 border-b border-zinc-800 flex items-center justify-between">
              <span className="text-[11px] font-semibold text-zinc-400 uppercase tracking-wider">Conditioning</span>
              <div className="flex gap-1">
                {CONDITIONING_TYPES.map(ct => (
                  <button
                    key={ct.value}
                    onClick={() => setConditioningType(ct.value)}
                    className={`px-2 py-0.5 rounded text-[10px] transition-colors ${
                      conditioningType === ct.value
                        ? 'bg-blue-600/20 text-blue-300 border border-blue-500/30'
                        : 'text-zinc-500 hover:text-zinc-300 border border-transparent'
                    }`}
                  >
                    {ct.label}
                  </button>
                ))}
              </div>
            </div>
            <div className="flex-1 bg-black flex items-center justify-center min-h-0 relative">
              {isExtracting && (
                <div className="absolute inset-0 flex items-center justify-center bg-black/50 z-10">
                  <Loader2 className="h-5 w-5 text-blue-400 animate-spin" />
                </div>
              )}
              {conditioningPreview ? (
                <img src={conditioningPreview} alt="Conditioning preview" className="max-w-full max-h-full object-contain" />
              ) : (
                <div className="text-center p-4">
                  <p className="text-zinc-600 text-xs">
                    {inputVideoUrl ? 'Scrub the input video to see conditioning preview' : 'Import a video to see conditioning'}
                  </p>
                </div>
              )}
            </div>
            <div className="px-3 py-1.5 border-t border-zinc-800 flex items-center justify-center">
              <button
                onClick={extractConditioning}
                disabled={!inputVideoPath || isExtracting}
                className="flex items-center gap-1.5 px-3 py-1 rounded text-[10px] text-zinc-400 hover:text-white hover:bg-zinc-800 transition-colors disabled:opacity-50"
              >
                <RefreshCw className={`h-3 w-3 ${isExtracting ? 'animate-spin' : ''}`} />
                Refresh Preview
              </button>
            </div>
          </div>

          {/* Right: Output Video */}
          <div className="flex-1 flex flex-col min-w-0">
            <div className="px-3 py-2 border-b border-zinc-800 flex items-center justify-between">
              <span className="text-[11px] font-semibold text-zinc-400 uppercase tracking-wider">Output</span>
              {outputVideoPath && (
                <button
                  onClick={handleAcceptOutput}
                  className="flex items-center gap-1 px-2 py-1 rounded text-[10px] bg-blue-600 hover:bg-blue-500 text-white transition-colors"
                >
                  {sourceClipId ? 'Add as Take' : 'Add to Assets'}
                </button>
              )}
            </div>
            <div className="flex-1 bg-black flex items-center justify-center min-h-0 relative">
              {isGenerating && (
                <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/60 z-10 gap-2">
                  <Loader2 className="h-6 w-6 text-blue-400 animate-spin" />
                  <span className="text-xs text-blue-300">{generationStatus}</span>
                </div>
              )}
              {generationError && !isGenerating && !outputVideoUrl && (
                <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/60 z-10 gap-2 p-4">
                  <AlertCircle className="h-6 w-6 text-red-400" />
                  <span className="text-xs text-red-400 text-center">{generationError}</span>
                  <button
                    onClick={() => setGenerationError(null)}
                    className="mt-1 text-xs text-zinc-400 hover:text-zinc-200 underline"
                  >
                    Dismiss
                  </button>
                </div>
              )}
              {outputVideoUrl ? (
                <video
                  ref={outputVideoRef}
                  src={outputVideoUrl}
                  className="max-w-full max-h-full object-contain"
                  controls
                />
              ) : !isGenerating && !generationError ? (
                <div className="text-center p-4">
                  <p className="text-zinc-600 text-xs">Output will appear here after generation</p>
                </div>
              ) : null}
            </div>
          </div>
        </div>

        {/* Bottom controls */}
        <div className="flex-shrink-0 border-t border-zinc-800 px-5 py-3 space-y-2">
          {/* Settings row (collapsible) */}
          {showSettings && (
            <div className="flex items-center gap-4 pb-2 border-b border-zinc-800 mb-2 flex-wrap">
              <div className="flex items-center gap-1.5">
                <label className="text-[10px] text-zinc-500">Width</label>
                <input type="number" value={width} onChange={e => setWidth(Number(e.target.value))}
                  className="w-16 bg-zinc-800 border border-zinc-700 rounded px-1.5 py-0.5 text-[10px] text-white" />
              </div>
              <div className="flex items-center gap-1.5">
                <label className="text-[10px] text-zinc-500">Height</label>
                <input type="number" value={height} onChange={e => setHeight(Number(e.target.value))}
                  className="w-16 bg-zinc-800 border border-zinc-700 rounded px-1.5 py-0.5 text-[10px] text-white" />
              </div>
              <div className="flex items-center gap-1.5">
                <label className="text-[10px] text-zinc-500">Frames</label>
                <input type="number" value={numFrames} onChange={e => setNumFrames(Number(e.target.value))}
                  className="w-16 bg-zinc-800 border border-zinc-700 rounded px-1.5 py-0.5 text-[10px] text-white" />
              </div>
              <div className="flex items-center gap-1.5">
                <label className="text-[10px] text-zinc-500">FPS</label>
                <input type="number" value={frameRate} onChange={e => setFrameRate(Number(e.target.value))}
                  className="w-14 bg-zinc-800 border border-zinc-700 rounded px-1.5 py-0.5 text-[10px] text-white" />
              </div>
              <div className="flex items-center gap-1.5">
                <label className="text-[10px] text-zinc-500">Seed</label>
                <input type="number" value={seed} onChange={e => setSeed(Number(e.target.value))}
                  className="w-20 bg-zinc-800 border border-zinc-700 rounded px-1.5 py-0.5 text-[10px] text-white" />
              </div>
            </div>
          )}

          {/* Reference images row */}
          <div className="flex items-center gap-2 pb-2 border-b border-zinc-800 mb-2">
            <span className="text-[10px] text-zinc-500 whitespace-nowrap">Reference Images</span>
            <div className="flex items-center gap-2 flex-1 overflow-x-auto min-h-[48px]">
              {refImages.map((img, i) => (
                <div key={i} className="relative flex-shrink-0 group">
                  <img src={img.url} alt={`Ref ${i + 1}`} className="h-10 w-14 object-cover rounded border border-zinc-700" />
                  <button
                    onClick={() => handleRemoveImage(i)}
                    className="absolute -top-1 -right-1 w-3.5 h-3.5 bg-red-600 rounded-full flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity"
                  >
                    <X className="h-2 w-2 text-white" />
                  </button>
                  <div className="flex items-center gap-1 mt-0.5">
                    <label className="text-[8px] text-zinc-600">Fr</label>
                    <input
                      type="number" min="0" value={img.frame}
                      onChange={e => handleUpdateImage(i, { frame: Number(e.target.value) })}
                      className="w-8 bg-zinc-800 border border-zinc-700 rounded px-0.5 text-[8px] text-white text-center"
                    />
                    <label className="text-[8px] text-zinc-600">S</label>
                    <input
                      type="number" min="0" max="2" step="0.1" value={img.strength}
                      onChange={e => handleUpdateImage(i, { strength: Number(e.target.value) })}
                      className="w-8 bg-zinc-800 border border-zinc-700 rounded px-0.5 text-[8px] text-white text-center"
                    />
                  </div>
                </div>
              ))}
              <button
                onClick={handleImportImage}
                className="flex-shrink-0 h-10 w-14 border border-dashed border-zinc-700 rounded flex flex-col items-center justify-center text-zinc-600 hover:text-zinc-400 hover:border-zinc-500 transition-colors"
              >
                <Upload className="h-3 w-3" />
                <span className="text-[7px] mt-0.5">Add</span>
              </button>
            </div>
          </div>

          {/* Main controls row */}
          <div className="flex items-center gap-3">
            {/* LoRA model picker */}
            <div className="relative">
              <button
                onClick={() => setShowModelDropdown(!showModelDropdown)}
                className="flex items-center gap-2 px-3 py-1.5 rounded-lg border border-zinc-700 bg-zinc-800 hover:bg-zinc-750 text-xs text-zinc-300 transition-colors min-w-[180px]"
              >
                <Sparkles className="h-3 w-3 text-blue-400 flex-shrink-0" />
                <span className="flex-1 text-left truncate">
                  {selectedModel ? selectedModel.name : 'Select LoRA model...'}
                </span>
                <ChevronDown className="h-3 w-3 text-zinc-500 flex-shrink-0" />
              </button>
              {showModelDropdown && (
                <div className="absolute bottom-full left-0 mb-1 w-80 bg-zinc-800 rounded-lg border border-zinc-700 shadow-xl z-50 overflow-hidden max-h-[320px] overflow-y-auto">
                  {/* Installed models */}
                  {models.length > 0 && (
                    <>
                      <div className="px-3 py-1.5 text-[9px] text-zinc-500 uppercase tracking-wider font-semibold bg-zinc-800/80 border-b border-zinc-700/50">
                        Installed Models
                      </div>
                      {models.map(m => (
                        <button
                          key={m.path}
                          onClick={() => { setSelectedModel(m); setShowModelDropdown(false) }}
                          className={`w-full text-left px-3 py-2 text-[11px] hover:bg-zinc-700 transition-colors ${
                            selectedModel?.path === m.path ? 'bg-blue-600/15 text-blue-300' : 'text-zinc-300'
                          }`}
                        >
                          <div className="font-medium">{m.name}</div>
                          <div className="text-[9px] text-zinc-500 mt-0.5">
                            Type: {m.conditioning_type} | Scale: {m.reference_downscale_factor}x
                          </div>
                        </button>
                      ))}
                    </>
                  )}

                  {/* Official models to download — only show ones NOT yet installed */}
                  {(() => {
                    const notInstalled = OFFICIAL_IC_LORA_MODELS.filter(
                      m => !models.some(inst => inst.path.includes(m.filename))
                    )
                    if (notInstalled.length === 0) return null
                    return (
                      <>
                        <div className="px-3 py-1.5 text-[9px] text-zinc-500 uppercase tracking-wider font-semibold bg-zinc-800/80 border-b border-zinc-700/50 border-t border-zinc-700/50">
                          Download Official Models
                        </div>
                        {notInstalled.map(m => {
                          const dlState = downloadingModels[m.id]
                          return (
                            <div
                              key={m.id}
                              className="flex items-center px-3 py-2 text-[11px] hover:bg-zinc-700/50 transition-colors"
                            >
                              <div className="flex-1 min-w-0">
                                <div className="font-medium text-zinc-300">{m.label}</div>
                                <div className="text-[9px] text-zinc-500 truncate">{m.repo_id}</div>
                              </div>
                              {dlState === 'downloading' ? (
                                <span className="flex items-center gap-1 text-[10px] text-amber-400 flex-shrink-0">
                                  <Loader2 className="h-3 w-3 animate-spin" />
                                  Downloading...
                                </span>
                              ) : dlState === 'done' ? (
                                <span className="flex items-center gap-1 text-[10px] text-green-400 flex-shrink-0">
                                  <Check className="h-3 w-3" />
                                  Done
                                </span>
                              ) : dlState === 'error' ? (
                                <button
                                  onClick={() => handleDownloadModel(m)}
                                  className="flex items-center gap-1 text-[10px] text-red-400 hover:text-red-300 flex-shrink-0"
                                >
                                  <Download className="h-3 w-3" />
                                  Retry
                                </button>
                              ) : (
                                <button
                                  onClick={() => handleDownloadModel(m)}
                                  className="flex items-center gap-1 text-[10px] text-blue-400 hover:text-blue-300 flex-shrink-0"
                                >
                                  <Download className="h-3 w-3" />
                                  Download
                                </button>
                              )}
                            </div>
                          )
                        })}
                      </>
                    )
                  })()}

                  {/* Browse custom */}
                  <div className="border-t border-zinc-700">
                    <button
                      onClick={handleBrowseLora}
                      className="w-full text-left px-3 py-2 text-[11px] text-zinc-400 hover:bg-zinc-700 hover:text-white flex items-center gap-2 transition-colors"
                    >
                      <FolderOpen className="h-3 w-3" />
                      Browse for custom LoRA...
                    </button>
                  </div>
                </div>
              )}
            </div>

            {/* Conditioning strength */}
            <div className="flex items-center gap-1.5">
              <label className="text-[10px] text-zinc-500 whitespace-nowrap">Strength</label>
              <input
                type="range" min="0" max="2" step="0.05" value={conditioningStrength}
                onChange={e => setConditioningStrength(Number(e.target.value))}
                className="w-20 accent-blue-500"
              />
              <span className="text-[10px] text-zinc-400 min-w-[28px]">{conditioningStrength.toFixed(2)}</span>
            </div>

            {/* Prompt */}
            <div className="flex-1">
              <input
                type="text"
                value={prompt}
                onChange={e => setPrompt(e.target.value)}
                disabled={isGenerating}
                placeholder="Describe the output style or content..."
                className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-1.5 text-xs text-white placeholder-zinc-600 focus:outline-none focus:border-blue-500 transition-colors disabled:opacity-50"
              />
            </div>

            {/* Generate button */}
            <div className="relative group">
              <button
                onClick={handleGenerate}
                disabled={!inputVideoPath || !selectedModel || isGenerating || !prompt.trim()}
                className="flex items-center gap-1.5 px-5 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-500 text-white text-xs font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isGenerating ? (
                  <>
                    <Loader2 className="h-3 w-3 animate-spin" />
                    Generating...
                  </>
                ) : (
                  <>
                    <Sparkles className="h-3 w-3" />
                    Generate
                  </>
                )}
              </button>
              {(!inputVideoPath || !selectedModel || !prompt.trim()) && !isGenerating && (
                <div className="absolute bottom-full right-0 mb-1.5 px-2.5 py-1.5 bg-zinc-800 border border-zinc-700 rounded-lg text-[10px] text-zinc-400 whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none shadow-lg">
                  {!inputVideoPath ? 'Import a driving video first' : !selectedModel ? 'Select a LoRA model first' : 'Enter a prompt first'}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
