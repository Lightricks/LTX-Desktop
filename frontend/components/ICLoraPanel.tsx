import { useState, useRef, useEffect, useCallback } from 'react'
import {
  Upload, Loader2, Film, Sparkles,
  RefreshCw, Download, AlertCircle, Trash2,
} from 'lucide-react'
import { ApiClient } from '../lib/api-client'
import { logger } from '../lib/logger'
import { pathToFileUrl } from '../lib/file-url'

export type ICLoraConditioningType = 'canny' | 'depth'

interface ICLoraPanelProps {
  initialVideoPath?: string | null
  resetKey?: number
  fillHeight?: boolean
  isProcessing?: boolean
  processingStatus?: string
  conditioningType?: ICLoraConditioningType
  onConditioningTypeChange?: (type: ICLoraConditioningType) => void
  conditioningStrength?: number
  onConditioningStrengthChange?: (strength: number) => void
  outputVideoPath?: string | null
  onChange?: (data: {
    videoPath: string | null
    conditioningType: ICLoraConditioningType
    conditioningStrength: number
    ready: boolean
  }) => void
}

export const CONDITIONING_TYPES: { value: ICLoraConditioningType; label: string; desc: string }[] = [
  { value: 'canny', label: 'Canny Edges', desc: 'Edge detection' },
  { value: 'depth', label: 'Depth Map', desc: 'Estimated depth' },
]

const IC_LORA_MODEL_IDS = ['ic_lora', 'depth_processor'] as const
type IcLoraModelId = typeof IC_LORA_MODEL_IDS[number]

const IC_LORA_MODEL_LABELS: Record<IcLoraModelId, string> = {
  ic_lora: 'IC-LoRA Union',
  depth_processor: 'Depth Processor',
}

const EMPTY_IC_MODEL_STATUS: Record<IcLoraModelId, boolean> = {
  ic_lora: false,
  depth_processor: false,
}


export function ICLoraPanel({
  initialVideoPath,
  resetKey,
  fillHeight = false,
  isProcessing = false,
  processingStatus = '',
  conditioningType: conditioningTypeProp,
  onConditioningTypeChange,
  conditioningStrength: conditioningStrengthProp,
  onConditioningStrengthChange,
  outputVideoPath: _outputVideoPath,
  onChange,
}: ICLoraPanelProps) {
  const inputVideoRef = useRef<HTMLVideoElement>(null)
  const [inputVideoPath, setInputVideoPath] = useState<string | null>(initialVideoPath || null)
  const inputVideoUrl = inputVideoPath ? pathToFileUrl(inputVideoPath) : null
  const [inputTime, setInputTime] = useState(0)

  const [internalCondType, setInternalCondType] = useState<ICLoraConditioningType>('canny')
  const [internalCondStrength, setInternalCondStrength] = useState(1.0)
  const conditioningType = conditioningTypeProp ?? internalCondType
  const conditioningStrength = conditioningStrengthProp ?? internalCondStrength
  const [conditioningPreview, setConditioningPreview] = useState<string | null>(null)
  const [isExtracting, setIsExtracting] = useState(false)

  const [icModelDownloaded, setIcModelDownloaded] = useState<Record<IcLoraModelId, boolean>>({ ...EMPTY_IC_MODEL_STATUS })
  const [isCheckingIcLora, setIsCheckingIcLora] = useState(false)
  const [isDownloadingIcLora, setIsDownloadingIcLora] = useState(false)
  const [downloadProgress, setDownloadProgress] = useState<Awaited<ReturnType<typeof ApiClient.getModelDownloadProgress>> | null>(null)
  const [downloadError, setDownloadError] = useState<string | null>(null)
  const [downloadSessionId, setDownloadSessionId] = useState<string | null>(null)
  const [extractError, setExtractError] = useState<string | null>(null)
  const [isDragOver, setIsDragOver] = useState(false)
  const icLoraReady = IC_LORA_MODEL_IDS.every(id => icModelDownloaded[id])

  useEffect(() => {
    if (resetKey === undefined) return
    setInputVideoPath(initialVideoPath || null)
    setInputTime(0)
    setInternalCondType('canny')
    setInternalCondStrength(1.0)
    onConditioningTypeChange?.('canny')
    onConditioningStrengthChange?.(1.0)
    setConditioningPreview(null)
    setExtractError(null)
  }, [resetKey, initialVideoPath]) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const ready = !!inputVideoPath && icLoraReady
    onChange?.({
      videoPath: inputVideoPath,
      conditioningType,
      conditioningStrength,
      ready,
    })
  }, [inputVideoUrl, inputVideoPath, conditioningType, conditioningStrength, icLoraReady, onChange])

  const checkIcLoraAvailability = useCallback(async () => {
    setIsCheckingIcLora(true)
    try {
      const statusPayload = await ApiClient.getModelsStatus()
      const nextStatus: Record<IcLoraModelId, boolean> = { ...EMPTY_IC_MODEL_STATUS }
      IC_LORA_MODEL_IDS.forEach(modelId => {
        nextStatus[modelId] = statusPayload.models.some(model => model.id === modelId && model.downloaded)
      })
      setIcModelDownloaded(nextStatus)
      const isReady = IC_LORA_MODEL_IDS.every(modelId => nextStatus[modelId])

      if (isReady) {
        setIsDownloadingIcLora(false)
        setDownloadProgress(null)
        setDownloadError(null)
        setDownloadSessionId(null)
      }
    } catch (e) {
      logger.warn(`Failed to fetch IC-LoRA model status: ${e}`)
      setDownloadError((e as Error).message)
    } finally {
      setIsCheckingIcLora(false)
    }
  }, [])

  useEffect(() => {
    void checkIcLoraAvailability()
  }, [checkIcLoraAvailability])

  useEffect(() => {
    if (icLoraReady || !isDownloadingIcLora || !downloadSessionId) return

    const pollProgress = async () => {
      try {
        const progressPayload = await ApiClient.getModelDownloadProgress({ sessionId: downloadSessionId })
        setDownloadProgress(progressPayload)

        if (progressPayload.status === 'error') {
          setIsDownloadingIcLora(false)
          setDownloadError(progressPayload.error || 'Model download failed')
          return
        }

        if (progressPayload.status === 'complete') {
          setIsDownloadingIcLora(false)
          await checkIcLoraAvailability()
        }
      } catch (e) {
        logger.warn(`Failed polling IC-LoRA download progress: ${e}`)
      }
    }

    void pollProgress()
    const interval = setInterval(() => { void pollProgress() }, 1000)
    return () => clearInterval(interval)
  }, [icLoraReady, isDownloadingIcLora, downloadSessionId, checkIcLoraAvailability])

  const handleDownloadIcLora = useCallback(async () => {
    if (isDownloadingIcLora) return
    setDownloadError(null)

    try {
      const startedPayload = await ApiClient.startModelDownload({
        modelTypes: [...IC_LORA_MODEL_IDS],
      })
      if (startedPayload.status === 'started') {
        setDownloadSessionId(startedPayload.sessionId)
        setIsDownloadingIcLora(true)
        return
      }

      setDownloadError('Unexpected response while starting IC-LoRA download')
    } catch (e) {
      logger.warn(`Failed to start IC-LoRA download: ${e}`)
      setDownloadError((e as Error).message)
    }
  }, [isDownloadingIcLora])

  const isExtractingRef = useRef(false)
  const extractConditioning = useCallback(async () => {
    if (!inputVideoPath || isExtractingRef.current || !icLoraReady) return
    isExtractingRef.current = true
    setIsExtracting(true)
    setExtractError(null)
    try {
      const payload = await ApiClient.extractIcLoraConditioning({
        video_path: inputVideoPath,
        conditioning_type: conditioningType,
        frame_time: inputTime,
      })
      setConditioningPreview(payload.conditioning)
    } catch (e) {
      logger.warn(`Failed to extract conditioning: ${e}`)
      setExtractError((e as Error).message)
    } finally {
      isExtractingRef.current = false
      setIsExtracting(false)
    }
  }, [inputVideoPath, conditioningType, inputTime, icLoraReady])

  const extractTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  useEffect(() => {
    if (!inputVideoPath || !icLoraReady) return
    if (extractTimerRef.current) clearTimeout(extractTimerRef.current)
    extractTimerRef.current = setTimeout(() => {
      void extractConditioning()
    }, 300)
    return () => {
      if (extractTimerRef.current) clearTimeout(extractTimerRef.current)
    }
  }, [inputTime, conditioningType, inputVideoPath, icLoraReady, extractConditioning])

  useEffect(() => {
    const video = inputVideoRef.current
    if (!video) return
    const onTime = () => setInputTime(video.currentTime)
    const onSeeked = () => setInputTime(video.currentTime)
    video.addEventListener('timeupdate', onTime)
    video.addEventListener('seeked', onSeeked)
    return () => {
      video.removeEventListener('timeupdate', onTime)
      video.removeEventListener('seeked', onSeeked)
    }
  }, [inputVideoUrl, icLoraReady, isCheckingIcLora])

  const handleBrowse = useCallback(async () => {
    const paths = await window.electronAPI.showOpenFileDialog({
      title: 'Select Driving Video',
      filters: [{ name: 'Video', extensions: ['mp4', 'mov', 'avi', 'webm', 'mkv'] }],
    })
    if (paths && paths.length > 0) {
      const filePath = paths[0]
      setInputVideoPath(filePath)
      
      setConditioningPreview(null)
      setExtractError(null)
    }
  }, [])

  const handleClear = useCallback(() => {
    setInputVideoPath(null)

    setInputTime(0)
    setConditioningPreview(null)
    setExtractError(null)
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)

    const assetData = e.dataTransfer.getData('asset')
    if (assetData) {
      try {
        const asset = JSON.parse(assetData) as { type?: string; path?: string }
        if (asset.type === 'video' && asset.path) {
          setInputVideoPath(asset.path)
          setConditioningPreview(null)
          setExtractError(null)
          return
        }
      } catch {
        // fall through
      }
    }

    const file = e.dataTransfer.files?.[0]
    if (file) {
      const filePath = window.electronAPI?.getPathForFile(file)
      if (filePath) {
        setInputVideoPath(filePath)
        
        setConditioningPreview(null)
        setExtractError(null)
      }
    }
  }, [])

  const showDownloadGate = isCheckingIcLora || !icLoraReady
  const runningDownloadProgress =
    downloadProgress?.status === 'downloading' ? downloadProgress : null
  const gateItems = IC_LORA_MODEL_IDS.map(modelId => {
    const downloaded = icModelDownloaded[modelId]
    const isCompleted = runningDownloadProgress?.completed_files?.includes(modelId) ?? false
    const isCurrentDownload = isDownloadingIcLora && runningDownloadProgress?.current_downloading_file === modelId
    const progress = downloaded ? 100 : (isCompleted ? 100 : (isCurrentDownload ? (runningDownloadProgress?.current_file_progress ?? 0) : 0))
    const status = downloaded ? 'Ready' : (isCompleted ? 'Complete' : (isCurrentDownload ? 'Downloading' : 'Missing'))
    return { id: modelId, label: IC_LORA_MODEL_LABELS[modelId], downloaded, progress, status }
  })

  return (
    <div className={`bg-zinc-900 border border-zinc-800 rounded-2xl overflow-hidden flex flex-col ${fillHeight ? 'h-full min-h-0' : ''}`}>
      <div className="flex items-center justify-between px-4 py-3 border-b border-zinc-800 flex-shrink-0">
        <div className="flex items-center gap-2">
          <Sparkles className="h-4 w-4 text-amber-400" />
          <span className="text-sm font-semibold text-white">IC-LoRA / Style Transfer</span>
        </div>
        {inputVideoUrl && (
          <div className="flex items-center gap-2">
            <button
              onClick={handleClear}
              className="p-1.5 rounded-md hover:bg-zinc-800 text-zinc-400 hover:text-white transition-colors"
              title="Clear video"
            >
              <Trash2 className="h-3.5 w-3.5" />
            </button>
            <button
              onClick={handleBrowse}
              className="p-1.5 rounded-md hover:bg-zinc-800 text-zinc-400 hover:text-white transition-colors"
              title="Replace video"
            >
              <RefreshCw className="h-3.5 w-3.5" />
            </button>
          </div>
        )}
      </div>

      {showDownloadGate ? (
        <div className="flex-1 flex items-center justify-center p-6 min-h-0 overflow-y-auto">
          <div className="w-full max-w-xl rounded-xl border border-zinc-700 bg-zinc-800/60 p-6">
            <div className="flex items-start gap-3">
              <div className="w-9 h-9 rounded-lg bg-blue-600/20 flex items-center justify-center mt-0.5">
                <Download className="h-4 w-4 text-blue-400" />
              </div>
              <div className="flex-1 min-w-0">
                <h3 className="text-sm font-semibold text-white">Download Required: IC-LoRA Resources</h3>
                <p className="text-xs text-zinc-400 mt-1">
                  Editing is locked until all IC-LoRA preprocessing models are available locally.
                </p>
              </div>
            </div>

            <div className="mt-5 space-y-3">
              {isCheckingIcLora ? (
                <div className="flex items-center gap-2 text-xs text-zinc-300">
                  <Loader2 className="h-4 w-4 animate-spin text-blue-400" />
                  Checking model availability...
                </div>
              ) : (
                <>
                  <div className="space-y-2">
                    {gateItems.map(item => (
                      <div key={item.id} className="rounded-lg border border-zinc-700 bg-zinc-900/60 px-3 py-2">
                        <div className="flex items-center justify-between text-[11px] mb-1.5">
                          <span className="text-zinc-300">{item.label}</span>
                          <span className={item.downloaded ? 'text-blue-400' : 'text-zinc-500'}>
                            {item.status}
                          </span>
                        </div>
                        <div className="h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                          <div
                            className="h-full transition-all duration-300 bg-blue-500"
                            style={{ width: `${item.progress}%` }}
                          />
                        </div>
                        <div className="mt-1 text-[10px] text-zinc-500">{item.progress}%</div>
                      </div>
                    ))}
                  </div>
                  {downloadError && (
                    <div className="text-[11px] text-red-400">{downloadError}</div>
                  )}
                  <div className="flex items-center gap-2 pt-1">
                    <button
                      onClick={handleDownloadIcLora}
                      disabled={isDownloadingIcLora}
                      className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-500 text-white text-xs font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      {isDownloadingIcLora ? (
                        <>
                          <Loader2 className="h-3 w-3 animate-spin" />
                          Downloading...
                        </>
                      ) : (
                        <>
                          <Download className="h-3 w-3" />
                          {downloadError ? 'Retry Download' : 'Download Models'}
                        </>
                      )}
                    </button>
                    <button
                      onClick={() => { void checkIcLoraAvailability() }}
                      disabled={isCheckingIcLora}
                      className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-zinc-600 text-zinc-300 hover:text-white hover:border-zinc-500 text-xs transition-colors disabled:opacity-50"
                    >
                      <RefreshCw className={`h-3 w-3 ${isCheckingIcLora ? 'animate-spin' : ''}`} />
                      Refresh
                    </button>
                  </div>
                </>
              )}
            </div>
          </div>
        </div>
      ) : (
        <div className="flex-1 flex min-h-0 overflow-hidden">
          <div className="flex-1 flex flex-col border-r border-zinc-800 min-w-0">
            <div className="px-3 py-2 border-b border-zinc-800 flex items-center justify-between gap-2">
              <span className="text-[11px] font-semibold text-zinc-400 uppercase tracking-wider shrink-0">Input</span>
              {inputVideoPath && (
                <span className="text-[10px] text-zinc-500 truncate min-w-0">
                  {inputVideoPath.split(/[\\/]/).pop()}
                </span>
              )}
              <button
                onClick={handleBrowse}
                className="flex items-center gap-1 px-2 py-0.5 rounded text-[10px] text-zinc-400 hover:text-white hover:bg-zinc-800 transition-colors shrink-0"
              >
                <Upload className="h-3 w-3" />
                Import
              </button>
            </div>
            <div
              className={`flex-1 min-h-0 bg-black flex items-center justify-center relative ${!inputVideoUrl ? 'border-2 border-dashed border-zinc-700 m-3 rounded-lg' : ''} ${isDragOver ? 'border-blue-500 bg-blue-500/10' : ''}`}
              onDragOver={(e) => { e.preventDefault(); setIsDragOver(true) }}
              onDragLeave={() => setIsDragOver(false)}
              onDrop={handleDrop}
            >
              {inputVideoUrl ? (
                <video
                  ref={inputVideoRef}
                  src={inputVideoUrl}
                  className="w-full h-full object-contain"
                  controls
                />
              ) : (
                <div className="text-center p-4">
                  <div className="w-12 h-12 rounded-full bg-zinc-800 flex items-center justify-center mx-auto mb-2">
                    <Film className="h-6 w-6 text-zinc-600" />
                  </div>
                  <p className="text-zinc-400 text-xs">Drop or import a driving video</p>
                  <button
                    onClick={handleBrowse}
                    className="mt-2 px-3 py-1.5 text-[10px] text-blue-400 border border-blue-500/30 rounded-lg hover:bg-blue-600/10 transition-colors"
                  >
                    Import Video
                  </button>
                </div>
              )}
            </div>
          </div>

          <div className="flex-1 flex flex-col min-w-0">
            <div className="px-3 py-2 border-b border-zinc-800 flex items-center justify-between gap-2">
              <span className="text-[11px] font-semibold text-zinc-400 uppercase tracking-wider">Conditioning</span>
              <button
                onClick={() => { void extractConditioning() }}
                disabled={!inputVideoPath || isExtracting}
                className="flex items-center gap-1 px-2 py-0.5 rounded text-[10px] text-zinc-400 hover:text-white hover:bg-zinc-800 transition-colors disabled:opacity-50"
              >
                <RefreshCw className={`h-3 w-3 ${isExtracting ? 'animate-spin' : ''}`} />
              </button>
            </div>
            <div className="flex-1 bg-black flex items-center justify-center min-h-0 relative">
              {isExtracting && (
                <div className="absolute inset-0 flex items-center justify-center bg-black/50 z-10">
                  <Loader2 className="h-5 w-5 text-blue-400 animate-spin" />
                </div>
              )}
              {conditioningPreview ? (
                <img src={conditioningPreview} alt="Conditioning preview" className="w-full h-full object-contain" />
              ) : (
                <div className="text-center p-4">
                  <p className="text-zinc-600 text-xs">
                    {inputVideoUrl ? 'Scrub the input video to see conditioning preview' : 'Import a video to preview conditioning'}
                  </p>
                </div>
              )}
            </div>
          </div>

          {/* Output column */}
          <div className="flex-1 flex flex-col border-l border-zinc-800 min-w-0">
            <div className="px-3 py-2 border-b border-zinc-800 flex items-center">
              <span className="text-[11px] font-semibold text-zinc-400 uppercase tracking-wider">Output</span>
            </div>
            <div className="flex-1 bg-black flex items-center justify-center min-h-0 relative">
              {_outputVideoPath ? (
                <video
                  src={pathToFileUrl(_outputVideoPath)}
                  className="w-full h-full object-contain"
                  controls
                />
              ) : isProcessing ? (
                <div className="text-center p-4">
                  <Loader2 className="h-6 w-6 text-blue-400 animate-spin mx-auto mb-2" />
                  <p className="text-zinc-400 text-xs">{processingStatus || 'Generating...'}</p>
                </div>
              ) : (
                <div className="text-center p-4">
                  <p className="text-zinc-600 text-xs">Output video will appear here</p>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {extractError && (
        <div className="px-4 py-3 border-t border-zinc-800 flex-shrink-0">
          <div className="flex items-center gap-2 text-xs text-red-400">
            <AlertCircle className="h-3.5 w-3.5 flex-shrink-0" />
            <span>{extractError}</span>
          </div>
        </div>
      )}
    </div>
  )
}
