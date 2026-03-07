import { useState, useEffect, useRef } from 'react'
import { Loader2, CheckCircle2, Download, Clock, ChevronDown, AlertCircle } from 'lucide-react'
import { backendFetch } from '../lib/backend'
import { logger } from '../lib/logger'

interface ModelInfo {
  name: string
  description: string
  downloaded: boolean
  size: number
  expected_size: number
  required?: boolean
  is_folder?: boolean
}

interface ModelsStatus {
  models: ModelInfo[]
  all_downloaded: boolean
  total_size: number
  downloaded_size: number
  total_size_gb: number
  downloaded_size_gb: number
}

interface ModelDownloadProgress {
  status: 'idle' | 'downloading' | 'complete' | 'error'
  currentFile: string
  currentFileProgress: number
  totalProgress: number
  downloadedBytes: number
  totalBytes: number
  filesCompleted: number
  totalFiles: number
  error: string | null
  speedMbps: number
}

interface ModelStatusDropdownProps {
  className?: string
}

export function ModelStatusDropdown({ className = '' }: ModelStatusDropdownProps) {
  const [isOpen, setIsOpen] = useState(false)
  const [modelsStatus, setModelsStatus] = useState<ModelsStatus | null>(null)
  const [downloadProgress, setDownloadProgress] = useState<ModelDownloadProgress | null>(null)
  const dropdownRef = useRef<HTMLDivElement>(null)

  // Fetch models status periodically
  useEffect(() => {
    const fetchModelsStatus = async () => {
      try {
        const response = await backendFetch('/api/models/status')
        if (response.ok) {
          setModelsStatus(await response.json())
        }
      } catch (e) {
        logger.error(`Failed to fetch models status: ${e}`)
      }
    }

    fetchModelsStatus()
    const interval = setInterval(fetchModelsStatus, 5000)
    return () => clearInterval(interval)
  }, [])

  // Poll download progress when downloading
  useEffect(() => {
    if (!isOpen) return

    const pollProgress = async () => {
      try {
        const response = await backendFetch('/api/models/download/progress')
        if (response.ok) {
          setDownloadProgress(await response.json())
        }
      } catch (e) {
        logger.error(`Failed to fetch download progress: ${e}`)
      }
    }

    pollProgress()
    const interval = setInterval(pollProgress, 1000)
    return () => clearInterval(interval)
  }, [isOpen])

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false)
      }
    }

    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside)
    }
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [isOpen])

  const isDownloading = downloadProgress?.status === 'downloading'
  const isPending = !modelsStatus?.all_downloaded
  const isReady = modelsStatus?.all_downloaded === true
  const hasError = downloadProgress?.status === 'error'

  // Format bytes to human-readable
  const formatBytes = (bytes: number): string => {
    if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(1)} GB`
    if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(1)} MB`
    return `${(bytes / 1e3).toFixed(1)} KB`
  }

  // Format time remaining
  const formatTimeRemaining = (bytesRemaining: number, speedMbps: number): string => {
    if (speedMbps <= 0) return 'Calculating...'
    const secondsRemaining = (bytesRemaining / 1e6) / speedMbps
    if (secondsRemaining < 60) return `${Math.ceil(secondsRemaining)}s`
    if (secondsRemaining < 3600) return `${Math.ceil(secondsRemaining / 60)}m`
    return `${(secondsRemaining / 3600).toFixed(1)}h`
  }

  // Get status label
  const getStatusLabel = (): string => {
    if (hasError) return 'Error'
    if (isDownloading) return `Downloading... ${downloadProgress?.totalProgress || 0}%`
    if (isReady) return 'Ready'
    if (isPending) return 'Models needed'
    return 'Checking...'
  }

  const startDownload = async () => {
    try {
      await backendFetch('/api/models/download', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      })
    } catch (e) {
      logger.error(`Failed to start download: ${e}`)
    }
  }

  return (
    <div className={`relative ${className}`} ref={dropdownRef}>
      {/* Status Badge - Clickable */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={`
          flex items-center gap-2 px-3 py-1.5 rounded-lg
          transition-all cursor-pointer
          ${hasError ? 'bg-red-500/20 hover:bg-red-500/30' :
            isReady ? 'bg-green-500/10 hover:bg-green-500/20' :
            isPending ? 'bg-zinc-800 hover:bg-zinc-700' :
            'bg-zinc-800 hover:bg-zinc-700'}
        `}
      >
        {isDownloading && (
          <Loader2 className="h-3.5 w-3.5 text-blue-400 animate-spin" />
        )}
        {isReady && !isDownloading && (
          <div className="w-2 h-2 bg-green-500 rounded-full" />
        )}
        {isPending && !isDownloading && !isReady && (
          <Download className="h-3.5 w-3.5 text-zinc-400" />
        )}
        {hasError && (
          <AlertCircle className="h-3.5 w-3.5 text-red-400" />
        )}

        <span className={`text-xs font-medium ${
          hasError ? 'text-red-400' :
          isReady ? 'text-green-400' :
          isPending ? 'text-zinc-300' :
          'text-zinc-300'
        }`}>
          {getStatusLabel()}
        </span>

        {isDownloading && (
          <span className="text-xs text-blue-400 font-semibold">
            {downloadProgress?.totalProgress || 0}%
          </span>
        )}

        <ChevronDown className={`h-3.5 w-3.5 text-zinc-500 transition-transform ${isOpen ? 'rotate-180' : ''}`} />
      </button>

      {/* Dropdown Panel */}
      {isOpen && (
        <div className="absolute right-0 top-full mt-2 w-96 bg-zinc-900 border border-zinc-700 rounded-xl shadow-2xl z-50 overflow-hidden">
          {/* Header */}
          <div className="px-4 py-3 border-b border-zinc-800">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-semibold text-white">Model Status</h3>
              {modelsStatus && (
                <span className="text-xs text-zinc-500">
                  {modelsStatus.downloaded_size_gb.toFixed(1)} / {modelsStatus.total_size_gb.toFixed(1)} GB
                </span>
              )}
            </div>
          </div>

          {/* Download Progress (if downloading) */}
          {isDownloading && downloadProgress && (
            <div className="px-4 py-3 bg-blue-500/5 border-b border-zinc-800">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs text-blue-300 font-medium">Downloading</span>
                <span className="text-xs text-blue-400">{downloadProgress.speedMbps.toFixed(1)} MB/s</span>
              </div>

              {/* Progress bar */}
              <div className="h-2 bg-zinc-800 rounded-full overflow-hidden mb-2">
                <div
                  className="h-full bg-gradient-to-r from-blue-500 via-blue-500 to-blue-500 transition-all duration-300"
                  style={{ width: `${downloadProgress.totalProgress}%` }}
                />
              </div>

              <div className="flex items-center justify-between text-xs text-zinc-500">
                <span className="truncate flex-1 mr-2">{downloadProgress.currentFile}</span>
                <div className="flex items-center gap-2">
                  <Clock className="h-3 w-3" />
                  <span>
                    {formatTimeRemaining(
                      downloadProgress.totalBytes - downloadProgress.downloadedBytes,
                      downloadProgress.speedMbps
                    )}
                  </span>
                </div>
              </div>

              <div className="text-xs text-zinc-600 mt-1">
                {downloadProgress.filesCompleted} / {downloadProgress.totalFiles} files complete
              </div>
            </div>
          )}

          {/* Model List */}
          <div className="max-h-64 overflow-y-auto">
            {modelsStatus?.models.filter(m => m.required).map((model, index, arr) => (
              <div
                key={model.name}
                className={`px-4 py-3 flex items-center gap-3 ${
                  index !== arr.length - 1 ? 'border-b border-zinc-800/50' : ''
                }`}
              >
                {model.downloaded ? (
                  <CheckCircle2 className="h-4 w-4 text-green-500 flex-shrink-0" />
                ) : (
                  <Download className="h-4 w-4 text-zinc-600 flex-shrink-0" />
                )}

                <div className="flex-1 min-w-0">
                  <div className="text-sm text-white font-medium truncate">{model.name}</div>
                  <div className="text-xs text-zinc-500 truncate">{model.description}</div>
                </div>

                <div className="text-right flex-shrink-0">
                  <div className={`text-xs ${model.downloaded ? 'text-green-400' : 'text-zinc-600'}`}>
                    {model.downloaded ? 'Ready' : 'Needed'}
                  </div>
                  <div className="text-xs text-zinc-600">
                    {formatBytes(model.expected_size)}
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Footer Actions */}
          {isPending && !isDownloading && (
            <div className="px-4 py-3 border-t border-zinc-800 bg-zinc-800/30">
              <button
                onClick={startDownload}
                className="w-full py-2 px-4 bg-blue-600 hover:bg-blue-500 text-white text-sm font-medium rounded-lg transition-colors flex items-center justify-center gap-2"
              >
                <Download className="h-4 w-4" />
                Download Missing Models
              </button>
            </div>
          )}

          {/* Error State */}
          {downloadProgress?.error && (
            <div className="px-4 py-3 border-t border-red-500/20 bg-red-500/10">
              <div className="text-xs text-red-400">{downloadProgress.error}</div>
              <button
                onClick={startDownload}
                className="mt-2 text-xs text-red-300 hover:text-red-200 underline"
              >
                Retry Download
              </button>
            </div>
          )}

          {/* All Ready */}
          {isReady && (
            <div className="px-4 py-3 border-t border-zinc-800 bg-green-500/5">
              <div className="flex items-center gap-2 text-green-400 text-sm">
                <CheckCircle2 className="h-4 w-4" />
                <span>All models ready</span>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
