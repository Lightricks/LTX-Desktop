import { useState, useEffect, useRef } from 'react'
import { X, FolderOpen, RefreshCw, ChevronDown, ChevronUp, Download } from 'lucide-react'
import { useT } from '../lib/i18n'
import { Button } from './ui/button'
import { logger } from '../lib/logger'

interface LogViewerProps {
  isOpen: boolean
  onClose: () => void
  embedded?: boolean
}

export function LogViewer({ isOpen, onClose, embedded = false }: LogViewerProps) {
  const { t } = useT()
  const [logs, setLogs] = useState<string[]>([])
  const [logPath, setLogPath] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [autoScroll, setAutoScroll] = useState(true)
  const logContainerRef = useRef<HTMLDivElement>(null)

  const fetchLogs = async () => {
    if (!window.electronAPI?.getLogs) return
    
    setIsLoading(true)
    try {
      const result = await window.electronAPI.getLogs()
      setLogs(result.lines || [])
      setLogPath(result.logPath || '')
    } catch (error) {
      logger.error(`Failed to fetch logs: ${error}`)
    } finally {
      setIsLoading(false)
    }
  }

  useEffect(() => {
    if (isOpen) {
      fetchLogs()
      // Auto-refresh every 2 seconds when open
      const interval = setInterval(fetchLogs, 2000)
      return () => clearInterval(interval)
    }
  }, [isOpen])

  useEffect(() => {
    if (autoScroll && logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight
    }
  }, [logs, autoScroll])

  const handleOpenFolder = async () => {
    if (window.electronAPI?.openLogFolder) {
      await window.electronAPI.openLogFolder()
    }
  }

  const handleDownload = () => {
    const content = logs.join('\n')
    const blob = new Blob([content], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = logPath ? logPath.split(/[/\\]/).pop() || 'session.log' : 'session.log'
    a.click()
    URL.revokeObjectURL(url)
  }

  if (!isOpen) return null

  const panel = (
    <div className={`bg-zinc-900 rounded-lg border border-zinc-700 w-full ${embedded ? 'h-full' : 'max-w-4xl h-[80vh]'} flex flex-col`}>
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-zinc-700">
          <div className="flex items-center gap-3">
            <h2 className="text-lg font-semibold text-white">{t('logs.logs')}</h2>
            <span className="text-xs text-zinc-500 font-mono truncate max-w-[300px]" title={logPath}>
              {logPath}
            </span>
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="sm"
              onClick={handleDownload}
              disabled={logs.length === 0}
              className="text-zinc-400 hover:text-white"
              title={t('logs.download')}
            >
              <Download className="h-4 w-4" />
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={handleOpenFolder}
              className="text-zinc-400 hover:text-white"
              title={t('logs.openFolder')}
            >
              <FolderOpen className="h-4 w-4" />
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={fetchLogs}
              disabled={isLoading}
              className="text-zinc-400 hover:text-white"
              title={t('logs.refresh')}
            >
              <RefreshCw className={`h-4 w-4 ${isLoading ? 'animate-spin' : ''}`} />
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setAutoScroll(!autoScroll)}
              className={`${autoScroll ? 'text-blue-400' : 'text-zinc-400'} hover:text-white`}
              title={autoScroll ? t('logs.autoScrollOn') : t('logs.autoScrollOff')}
            >
              {autoScroll ? <ChevronDown className="h-4 w-4" /> : <ChevronUp className="h-4 w-4" />}
            </Button>
            {!embedded && (
              <Button
                variant="ghost"
                size="sm"
                onClick={onClose}
                className="text-zinc-400 hover:text-white"
              >
                <X className="h-4 w-4" />
              </Button>
            )}
          </div>
        </div>

        {/* Log content */}
        <div
          ref={logContainerRef}
          className="flex-1 overflow-auto p-4 font-mono text-xs bg-black"
        >
          {logs.length === 0 ? (
            <div className="text-zinc-500 text-center py-8">
              {t('logs.noLogs')}
            </div>
          ) : (
            <div className="space-y-0.5">
              {logs.map((line, index) => {
                // Color code log levels
                let lineClass = 'text-zinc-300'
                if (line.includes(' - ERROR - ') || line.includes(' - CRITICAL - ')) {
                  lineClass = 'text-red-400'
                } else if (line.includes(' - WARNING - ')) {
                  lineClass = 'text-yellow-400'
                } else if (line.includes(' - INFO - ')) {
                  lineClass = 'text-blue-300'
                } else if (line.includes(' - DEBUG - ')) {
                  lineClass = 'text-zinc-500'
                }
                
                return (
                  <div key={index} className={`${lineClass} whitespace-pre-wrap break-all`}>
                    {line}
                  </div>
                )
              })}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between p-3 border-t border-zinc-700 text-xs text-zinc-500">
          <span>{logs.length} lines {t('logs.lastLines')}</span>
          <span>{t('logs.autoRefresh')}</span>
        </div>
      </div>
  )

  if (embedded) {
    return panel
  }

  return (
    <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50 p-4">
      {panel}
    </div>
  )
}
