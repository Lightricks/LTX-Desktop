import { useState, useEffect, useCallback, useRef } from 'react'
import { logger } from '../lib/logger'
import { addGenericAssetToProject, addVisualAssetToProject } from '../lib/asset-copy'
import {
  X, FileVideo, FileAudio, Image, Check, AlertTriangle,
  FolderOpen, RefreshCw, Loader2, FileText, Link2,
  ChevronDown, ChevronRight, Upload
} from 'lucide-react'
import type { ParsedTimeline, ParsedMediaRef } from '../lib/timeline-import'
import { parseTimelineXml } from '../lib/timeline-import'
import { selectShowImportTimelineModal } from '../views/editor/editor-selectors'
import { useEditorActions, useEditorStore } from '../views/editor/editor-store'

interface ImportTimelineModalProps {
  projectId: string | null
}

type ImportStep = 'select' | 'parsing' | 'relink' | 'error'
type MediaImportStatus = 'missing' | 'ready' | 'copying' | 'copied' | 'skipped' | 'error'

interface ImportProgressState {
  total: number
  completed: number
  skipped: number
  currentName: string
}

function getFilenameFromPath(filePath: string): string {
  const parts = filePath.split(/[/\\]/)
  return parts[parts.length - 1] || ''
}

export function ImportTimelineModal({ projectId }: ImportTimelineModalProps) {
  const { closeImportTimelineModal, importParsedTimeline } = useEditorActions()
  const isOpen = useEditorStore(selectShowImportTimelineModal)
  const [step, setStep] = useState<ImportStep>('select')
  const [parsedTimeline, setParsedTimeline] = useState<ParsedTimeline | null>(null)
  const [mediaRefs, setMediaRefs] = useState<ParsedMediaRef[]>([])
  const [mediaStatus, setMediaStatus] = useState<Record<string, MediaImportStatus>>({})
  const [error, setError] = useState<string>('')
  const [importError, setImportError] = useState<string>('')
  const [expandedInfo, setExpandedInfo] = useState(true)
  const [expandedMedia, setExpandedMedia] = useState(true)
  const [isChecking, setIsChecking] = useState(false)
  const [isSearching, setIsSearching] = useState(false)
  const [isImporting, setIsImporting] = useState(false)
  const [importProgress, setImportProgress] = useState<ImportProgressState>({
    total: 0,
    completed: 0,
    skipped: 0,
    currentName: '',
  })
  const fileInputRef = useRef<HTMLInputElement>(null)

  const closeModal = useCallback(() => {
    closeImportTimelineModal()
  }, [closeImportTimelineModal])

  // Reset when modal opens
  useEffect(() => {
    if (!isOpen) return
    setStep('select')
    setParsedTimeline(null)
    setMediaRefs([])
    setMediaStatus({})
    setError('')
    setImportError('')
    setIsChecking(false)
    setIsSearching(false)
    setIsImporting(false)
    setImportProgress({ total: 0, completed: 0, skipped: 0, currentName: '' })
  }, [isOpen])

  const applyAvailabilityToStatus = useCallback((refs: ParsedMediaRef[], availabilityById: Record<string, boolean>) => {
    const nextStatus: Record<string, MediaImportStatus> = {}
    for (const ref of refs) {
      nextStatus[ref.id] = availabilityById[ref.id] ? 'ready' : 'missing'
    }
    setMediaStatus(nextStatus)
  }, [])

  // Check file existence for all media refs
  const checkMediaFiles = useCallback(async (refs: ParsedMediaRef[]): Promise<Record<string, boolean>> => {
    const availabilityById: Record<string, boolean> = {}
    if (!window.electronAPI?.checkFilesExist) {
      refs.forEach(ref => {
        availabilityById[ref.id] = false
      })
      return availabilityById
    }

    setIsChecking(true)
    try {
      const uniquePaths = Array.from(new Set(
        refs
          .map(r => r.path?.trim())
          .filter((p): p is string => Boolean(p))
      ))

      const results = await window.electronAPI.checkFilesExist({ filePaths: uniquePaths })
      refs.forEach(ref => {
        const path = ref.path?.trim()
        availabilityById[ref.id] = path ? (results[path] || false) : false
      })
      return availabilityById
    } catch (err) {
      logger.error(`Error checking files: ${err}`)
      refs.forEach(ref => {
        availabilityById[ref.id] = false
      })
      return availabilityById
    } finally {
      setIsChecking(false)
    }
  }, [])

  // Handle file selection
  const handleFileSelected = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    const filename = file.name
    const ext = filename.split('.').pop()?.toLowerCase() || ''

    // Check for AAF
    if (ext === 'aaf') {
      setError(
        'AAF files cannot be imported directly.\n\n' +
        'Please export your timeline as FCP 7 XML from your editing software:\n\n' +
        '  Premiere Pro:  File → Export → Final Cut Pro XML\n' +
        '  DaVinci Resolve:  File → Export Timeline → FCP 7 XML\n' +
        '  Avid Media Composer:  File → Export → FCP 7 XML'
      )
      setStep('error')
      return
    }

    setStep('parsing')
    setImportError('')

    try {
      const content = await file.text()
      const timeline = parseTimelineXml(content, filename)

      if (!timeline) {
        throw new Error('Could not parse timeline from file')
      }

      setParsedTimeline(timeline)
      setMediaRefs(timeline.mediaRefs)

      const availabilityById = await checkMediaFiles(timeline.mediaRefs)
      applyAvailabilityToStatus(timeline.mediaRefs, availabilityById)
      setStep('relink')
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
      setStep('error')
    }

    // Reset input
    if (fileInputRef.current) fileInputRef.current.value = ''
  }, [applyAvailabilityToStatus, checkMediaFiles])

  // Relink a single media file
  const handleRelinkFile = useCallback(async (mediaRefId: string) => {
    if (isImporting || !window.electronAPI?.showOpenFileDialog) return

    const filePaths = await window.electronAPI.showOpenFileDialog({
      title: 'Relink Media File',
      filters: [
        { name: 'Media Files', extensions: ['mp4', 'mov', 'avi', 'mkv', 'mxf', 'mp3', 'wav', 'aac', 'flac', 'jpg', 'jpeg', 'png', 'tiff', 'exr', 'dpx'] },
        { name: 'All Files', extensions: ['*'] },
      ],
    })

    if (!filePaths || filePaths.length === 0) return

    const newPath = filePaths[0]
    setImportError('')
    setMediaRefs(prev => prev.map(r =>
      r.id === mediaRefId
        ? { ...r, path: newPath }
        : r
    ))
    setMediaStatus(prev => ({ ...prev, [mediaRefId]: 'ready' }))
  }, [isImporting])

  // Search a directory for all missing files
  const handleSearchDirectory = useCallback(async () => {
    if (isImporting || !window.electronAPI?.showOpenDirectoryDialog) return

    const dir = await window.electronAPI.showOpenDirectoryDialog({ title: 'Select folder to search for media files' })
    if (!dir) return

    setIsSearching(true)
    setImportError('')

    try {
      const missingRefs = mediaRefs.filter(r => mediaStatus[r.id] === 'missing')
      if (missingRefs.length === 0) return

      // Build list of filenames to search for
      const filenames: string[] = []
      for (const ref of missingRefs) {
        const filename = ref.name || getFilenameFromPath(ref.path)
        if (filename) filenames.push(filename)
      }

      if (filenames.length === 0) return

      // Use recursive directory search (searches subdirectories up to 10 levels deep)
      if (window.electronAPI.searchDirectoryForFiles) {
        const results = await window.electronAPI.searchDirectoryForFiles({ directory: dir, filenames })
        // results is { "filename.mp4" (lowercase): "C:\\full\\path\\filename.mp4" }

        setMediaRefs(prev => prev.map(ref => {
          if (mediaStatus[ref.id] !== 'missing') return ref
          const filename = ref.name || getFilenameFromPath(ref.path)
          const foundPath = filename ? results[filename.toLowerCase()] : undefined

          if (foundPath) {
            return { ...ref, path: foundPath }
          }
          return ref
        }))

        setMediaStatus(prev => {
          const next = { ...prev }
          for (const ref of missingRefs) {
            const filename = ref.name || getFilenameFromPath(ref.path)
            const foundPath = filename ? results[filename.toLowerCase()] : undefined
            if (foundPath) {
              next[ref.id] = 'ready'
            }
          }
          return next
        })
      } else {
        // Fallback: check direct paths only (no recursive search)
        const searchPaths: string[] = []
        for (const filename of filenames) {
          const separator = dir.includes('\\') ? '\\' : '/'
          searchPaths.push(`${dir}${separator}${filename}`)
        }

        const results = await window.electronAPI.checkFilesExist({ filePaths: searchPaths })

        setMediaRefs(prev => prev.map(ref => {
          if (mediaStatus[ref.id] !== 'missing') return ref
          const filename = ref.name || getFilenameFromPath(ref.path)
          const separator = dir.includes('\\') ? '\\' : '/'
          const testPath = filename ? `${dir}${separator}${filename}` : ''

          if (testPath && results[testPath]) {
            return { ...ref, path: testPath }
          }
          return ref
        }))

        setMediaStatus(prev => {
          const next = { ...prev }
          for (const ref of missingRefs) {
            const filename = ref.name || getFilenameFromPath(ref.path)
            const separator = dir.includes('\\') ? '\\' : '/'
            const testPath = filename ? `${dir}${separator}${filename}` : ''
            if (testPath && results[testPath]) {
              next[ref.id] = 'ready'
            }
          }
          return next
        })
      }
    } catch (err) {
      logger.error(`Error searching directory: ${err}`)
    } finally {
      setIsSearching(false)
    }
  }, [isImporting, mediaRefs, mediaStatus])

  // Recheck all paths
  const handleRecheckAll = useCallback(async () => {
    if (isImporting) return
    const availabilityById = await checkMediaFiles(mediaRefs)
    applyAvailabilityToStatus(mediaRefs, availabilityById)
  }, [applyAvailabilityToStatus, checkMediaFiles, isImporting, mediaRefs])

  // Confirm import
  const handleConfirmImport = useCallback(async () => {
    if (!parsedTimeline || isImporting) return
    if (!projectId) {
      setImportError('Cannot import timeline: no active project.')
      return
    }

    setImportError('')
    setIsImporting(true)

    const refsToImport = mediaRefs.map(ref => ({ ...ref }))
    const copiedAssetBySource = new Map<string, {
      path: string
      bigThumbnailPath?: string
      smallThumbnailPath?: string
      width?: number
      height?: number
    }>()
    const total = refsToImport.length
    let completed = 0
    let skipped = 0

    setImportProgress({ total, completed: 0, skipped: 0, currentName: '' })

    try {
      for (const ref of refsToImport) {
        const sourcePath = ref.path?.trim() || ''
        const displayName = ref.name || getFilenameFromPath(sourcePath) || 'Unnamed media'
        const currentStatus = mediaStatus[ref.id] || 'missing'

        setImportProgress({
          total,
          completed,
          skipped,
          currentName: displayName,
        })

        if (!sourcePath || currentStatus === 'missing') {
          skipped += 1
          completed += 1
          setMediaStatus(prev => ({ ...prev, [ref.id]: 'skipped' }))
          setImportProgress({
            total,
            completed,
            skipped,
            currentName: displayName,
          })
          continue
        }

        const existingCopiedAsset = copiedAssetBySource.get(sourcePath)
        if (existingCopiedAsset) {
          ref.path = existingCopiedAsset.path
          ref.bigThumbnailPath = existingCopiedAsset.bigThumbnailPath
          ref.smallThumbnailPath = existingCopiedAsset.smallThumbnailPath
          ref.width = existingCopiedAsset.width
          ref.height = existingCopiedAsset.height
          completed += 1
          setMediaStatus(prev => ({ ...prev, [ref.id]: 'copied' }))
          setMediaRefs(prev => prev.map(item => (
            item.id === ref.id
              ? {
                  ...item,
                  path: existingCopiedAsset.path,
                  bigThumbnailPath: existingCopiedAsset.bigThumbnailPath,
                  smallThumbnailPath: existingCopiedAsset.smallThumbnailPath,
                  width: existingCopiedAsset.width,
                  height: existingCopiedAsset.height,
                }
              : item
          )))
          setImportProgress({
            total,
            completed,
            skipped,
            currentName: displayName,
          })
          continue
        }

        setMediaStatus(prev => ({ ...prev, [ref.id]: 'copying' }))
        let copiedAsset: {
          path: string
          bigThumbnailPath?: string
          smallThumbnailPath?: string
          width?: number
          height?: number
        } | null = null
        if (ref.type === 'video' || ref.type === 'image') {
          const copied = await addVisualAssetToProject(sourcePath, projectId, ref.type)
          if (!copied) {
            setMediaStatus(prev => ({ ...prev, [ref.id]: 'error' }))
            throw new Error(`Failed to copy media into project assets: ${displayName}`)
          }
          copiedAsset = copied
        } else {
          const copied = await addGenericAssetToProject(sourcePath, projectId)
          if (!copied?.path) {
            setMediaStatus(prev => ({ ...prev, [ref.id]: 'error' }))
            throw new Error(`Failed to copy media into project assets: ${displayName}`)
          }
          copiedAsset = { path: copied.path }
        }

        ref.path = copiedAsset.path
        ref.bigThumbnailPath = copiedAsset.bigThumbnailPath
        ref.smallThumbnailPath = copiedAsset.smallThumbnailPath
        ref.width = copiedAsset.width
        ref.height = copiedAsset.height
        copiedAssetBySource.set(sourcePath, copiedAsset)
        completed += 1

        setMediaStatus(prev => ({ ...prev, [ref.id]: 'copied' }))
        setMediaRefs(prev => prev.map(item => (
          item.id === ref.id
            ? {
                ...item,
                path: copiedAsset.path,
                bigThumbnailPath: copiedAsset.bigThumbnailPath,
                smallThumbnailPath: copiedAsset.smallThumbnailPath,
                width: copiedAsset.width,
                height: copiedAsset.height,
              }
            : item
        )))
        setImportProgress({
          total,
          completed,
          skipped,
          currentName: displayName,
        })
      }

      // Update timeline refs with copied/updated paths
      const updatedTimeline: ParsedTimeline = {
        ...parsedTimeline,
        mediaRefs: refsToImport,
      }

      importParsedTimeline(updatedTimeline)
      closeModal()
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err)
      setImportError(message)
      logger.error(`Import timeline copy failed: ${message}`)
    } finally {
      setIsImporting(false)
    }
  }, [closeModal, importParsedTimeline, isImporting, mediaRefs, mediaStatus, parsedTimeline, projectId])

  const readyCount = mediaRefs.filter(r => (mediaStatus[r.id] || 'missing') !== 'missing').length
  const totalCount = mediaRefs.length
  const allFound = readyCount === totalCount && totalCount > 0
  const interactionDisabled = isImporting
  const importPercent = importProgress.total > 0 ? (importProgress.completed / importProgress.total) * 100 : 0

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm">
      <div className="bg-zinc-900 border border-zinc-700 rounded-xl shadow-2xl w-[680px] max-h-[85vh] flex flex-col overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-zinc-800">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-blue-600/20 flex items-center justify-center">
              <Upload className="h-4 w-4 text-blue-400" />
            </div>
            <div>
              <h2 className="text-base font-semibold text-white">Import Timeline</h2>
              <p className="text-[11px] text-zinc-500">
                Premiere Pro XML, DaVinci Resolve XML, Final Cut Pro XML/FCPXML
              </p>
            </div>
          </div>
          <button
            onClick={closeModal}
            disabled={interactionDisabled}
            className="p-1.5 rounded-lg hover:bg-zinc-800 text-zinc-500 hover:text-zinc-300 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        {/* Body */}
        <div className={`flex-1 overflow-auto p-6 ${interactionDisabled ? 'pointer-events-none opacity-60' : ''}`}>
          {/* Step 1: File Selection */}
          {step === 'select' && (
            <div className="space-y-6">
              <div
                onClick={() => fileInputRef.current?.click()}
                className="border-2 border-dashed border-zinc-700 hover:border-blue-500/50 rounded-xl p-10 text-center cursor-pointer transition-colors group"
              >
                <div className="w-14 h-14 rounded-full bg-zinc-800 group-hover:bg-blue-900/30 flex items-center justify-center mx-auto mb-4 transition-colors">
                  <FileText className="h-7 w-7 text-zinc-500 group-hover:text-blue-400 transition-colors" />
                </div>
                <p className="text-sm text-zinc-300 font-medium mb-1">Click to select timeline file</p>
                <p className="text-xs text-zinc-600">Supports .xml (FCP 7 XML), .fcpxml</p>
              </div>

              <input
                ref={fileInputRef}
                type="file"
                accept=".xml,.fcpxml,.aaf"
                onChange={handleFileSelected}
                className="hidden"
              />

              <div className="bg-zinc-800/50 rounded-lg p-4 space-y-2">
                <h4 className="text-xs font-semibold text-zinc-400 uppercase tracking-wider">How to export from your NLE:</h4>
                <div className="space-y-1.5 text-[11px] text-zinc-500">
                  <p><span className="text-blue-400 font-medium">Premiere Pro:</span> File → Export → Final Cut Pro XML</p>
                  <p><span className="text-orange-400 font-medium">DaVinci Resolve:</span> File → Export Timeline → FCP 7 XML (.xml)</p>
                  <p><span className="text-blue-400 font-medium">Final Cut Pro:</span> File → Export XML</p>
                  <p className="text-zinc-600 pt-1 border-t border-zinc-700/50 mt-2">
                    AAF files are binary and cannot be imported directly. Please export as XML instead.
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Parsing indicator */}
          {step === 'parsing' && (
            <div className="flex flex-col items-center justify-center py-16">
              <Loader2 className="h-8 w-8 text-blue-400 animate-spin mb-4" />
              <p className="text-sm text-zinc-400">Parsing timeline...</p>
            </div>
          )}

          {/* Error */}
          {step === 'error' && (
            <div className="space-y-4">
              <div className="bg-red-900/20 border border-red-800/30 rounded-lg p-4">
                <div className="flex items-start gap-3">
                  <AlertTriangle className="h-5 w-5 text-red-400 flex-shrink-0 mt-0.5" />
                  <div>
                    <p className="text-sm font-medium text-red-300 mb-1">Import Error</p>
                    <p className="text-xs text-red-400/80 whitespace-pre-wrap">{error}</p>
                  </div>
                </div>
              </div>
              <button
                onClick={() => { setStep('select'); setError('') }}
                className="px-4 py-2 rounded-lg bg-zinc-800 text-zinc-300 text-sm hover:bg-zinc-700 transition-colors"
              >
                Try another file
              </button>
            </div>
          )}

          {/* Step 2: Relink media */}
          {step === 'relink' && parsedTimeline && (
            <div className="space-y-4">
              {/* Timeline info */}
              <div>
                <button
                  onClick={() => setExpandedInfo(!expandedInfo)}
                  className="flex items-center gap-2 text-xs font-semibold text-zinc-400 uppercase tracking-wider mb-2 hover:text-zinc-300"
                >
                  {expandedInfo ? <ChevronDown className="h-3.5 w-3.5" /> : <ChevronRight className="h-3.5 w-3.5" />}
                  Timeline Info
                </button>
                {expandedInfo && (
                  <div className="bg-zinc-800/50 rounded-lg p-3 grid grid-cols-2 gap-x-6 gap-y-1.5 text-[11px]">
                    <div className="flex justify-between">
                      <span className="text-zinc-500">Name:</span>
                      <span className="text-white font-medium">{parsedTimeline.name}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-zinc-500">Format:</span>
                      <span className="text-zinc-300">
                        {parsedTimeline.format === 'fcp7xml' ? 'FCP 7 XML' : parsedTimeline.format === 'fcpxml' ? 'FCPXML' : parsedTimeline.format}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-zinc-500">FPS:</span>
                      <span className="text-zinc-300">{parsedTimeline.fps.toFixed(2)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-zinc-500">Duration:</span>
                      <span className="text-zinc-300">{parsedTimeline.duration.toFixed(1)}s</span>
                    </div>
                    {parsedTimeline.width && parsedTimeline.height && (
                      <div className="flex justify-between">
                        <span className="text-zinc-500">Resolution:</span>
                        <span className="text-zinc-300">{parsedTimeline.width}x{parsedTimeline.height}</span>
                      </div>
                    )}
                    <div className="flex justify-between">
                      <span className="text-zinc-500">Clips:</span>
                      <span className="text-zinc-300">{parsedTimeline.clips.length}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-zinc-500">Video Tracks:</span>
                      <span className="text-zinc-300">{parsedTimeline.videoTrackCount}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-zinc-500">Audio Tracks:</span>
                      <span className="text-zinc-300">{parsedTimeline.audioTrackCount}</span>
                    </div>
                  </div>
                )}
              </div>

              {/* Media files */}
              <div>
                <div className="flex items-center justify-between mb-2">
                  <button
                    onClick={() => setExpandedMedia(!expandedMedia)}
                    className="flex items-center gap-2 text-xs font-semibold text-zinc-400 uppercase tracking-wider hover:text-zinc-300"
                  >
                    {expandedMedia ? <ChevronDown className="h-3.5 w-3.5" /> : <ChevronRight className="h-3.5 w-3.5" />}
                    Media Files ({readyCount}/{totalCount} linked)
                  </button>
                  <div className="flex items-center gap-1.5">
                    <button
                      onClick={handleSearchDirectory}
                      disabled={isSearching || allFound || interactionDisabled}
                      className="px-2.5 py-1 rounded-md bg-zinc-800 text-zinc-400 text-[10px] hover:bg-zinc-700 hover:text-zinc-300 flex items-center gap-1.5 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                      title="Search a folder for missing media"
                    >
                      {isSearching ? <Loader2 className="h-3 w-3 animate-spin" /> : <FolderOpen className="h-3 w-3" />}
                      Search Folder
                    </button>
                    <button
                      onClick={handleRecheckAll}
                      disabled={isChecking || interactionDisabled}
                      className="px-2.5 py-1 rounded-md bg-zinc-800 text-zinc-400 text-[10px] hover:bg-zinc-700 hover:text-zinc-300 flex items-center gap-1.5 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                      title="Recheck all file paths"
                    >
                      {isChecking ? <Loader2 className="h-3 w-3 animate-spin" /> : <RefreshCw className="h-3 w-3" />}
                      Recheck
                    </button>
                  </div>
                </div>

                {/* Availability bar */}
                <div className="mb-2">
                  <div className="h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all duration-300 ${allFound ? 'bg-green-500' : readyCount > 0 ? 'bg-amber-500' : 'bg-red-500'}`}
                      style={{ width: `${totalCount > 0 ? (readyCount / totalCount) * 100 : 0}%` }}
                    />
                  </div>
                </div>

                {expandedMedia && (
                  <div className="space-y-1 max-h-[300px] overflow-auto rounded-lg border border-zinc-800">
                    {mediaRefs.map((ref, i) => {
                      const TypeIcon = ref.type === 'video' ? FileVideo : ref.type === 'audio' ? FileAudio : Image
                      const status = mediaStatus[ref.id] || 'missing'
                      const isPositive = status === 'ready' || status === 'copied'
                      const isMissing = status === 'missing'
                      const isCopying = status === 'copying'
                      const isSkipped = status === 'skipped'
                      const isErrored = status === 'error'

                      return (
                        <div
                          key={ref.id}
                          className={`flex items-center gap-2 px-3 py-2 text-[11px] ${i % 2 === 0 ? 'bg-zinc-800/30' : 'bg-zinc-900/30'}`}
                        >
                          <TypeIcon className={`h-3.5 w-3.5 flex-shrink-0 ${
                            ref.type === 'video' ? 'text-blue-400' : ref.type === 'audio' ? 'text-green-400' : 'text-blue-400'
                          }`} />

                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-1.5">
                              {isCopying ? (
                                <Loader2 className="h-3 w-3 text-blue-400 flex-shrink-0 animate-spin" />
                              ) : isPositive ? (
                                <Check className="h-3 w-3 text-green-400 flex-shrink-0" />
                              ) : isSkipped ? (
                                <AlertTriangle className="h-3 w-3 text-amber-400 flex-shrink-0" />
                              ) : (
                                <AlertTriangle className={`h-3 w-3 flex-shrink-0 ${isErrored ? 'text-red-400' : 'text-red-400'}`} />
                              )}
                              <span className={`truncate font-medium ${
                                isPositive ? 'text-zinc-300' : isSkipped ? 'text-amber-300' : 'text-red-300'
                              }`}>
                                {ref.name}
                              </span>
                            </div>
                            <p className="text-[9px] text-zinc-600 truncate mt-0.5">
                              {ref.path || '(no path)'}
                            </p>
                          </div>

                          {isMissing && (
                            <button
                              onClick={() => handleRelinkFile(ref.id)}
                              disabled={interactionDisabled}
                              className="flex-shrink-0 px-2 py-1 rounded bg-zinc-700 hover:bg-zinc-600 text-zinc-300 text-[10px] flex items-center gap-1 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
                              title="Relink this file"
                            >
                              <Link2 className="h-3 w-3" />
                              Relink
                            </button>
                          )}
                        </div>
                      )
                    })}

                    {mediaRefs.length === 0 && (
                      <div className="p-4 text-center text-xs text-zinc-600">
                        No media files referenced in this timeline.
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        {step === 'relink' && (
          <>
            <div className={`px-6 py-4 border-t border-zinc-800 flex items-center justify-between bg-zinc-900 ${interactionDisabled ? 'pointer-events-none opacity-60' : ''}`}>
              <div className="text-[11px] text-zinc-500">
                {importError && (
                  <span className="text-red-400">{importError}</span>
                )}
                {!importError && !allFound && totalCount > 0 && (
                  <span className="text-amber-400">
                    {totalCount - readyCount} missing file{totalCount - readyCount !== 1 ? 's' : ''} — clips with missing media will be placeholders
                  </span>
                )}
                {!importError && allFound && totalCount > 0 && (
                  <span className="text-green-400">All media files found</span>
                )}
                {!importError && totalCount === 0 && (
                  <span>No media references to link</span>
                )}
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={closeModal}
                  disabled={interactionDisabled}
                  className="px-4 py-2 rounded-lg bg-zinc-800 text-zinc-300 text-sm hover:bg-zinc-700 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  Cancel
                </button>
                <button
                  onClick={handleConfirmImport}
                  disabled={interactionDisabled}
                  className="px-4 py-2 rounded-lg bg-blue-600 text-white text-sm hover:bg-blue-500 transition-colors font-medium flex items-center gap-2 disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  <Upload className="h-3.5 w-3.5" />
                  Import Timeline
                </button>
              </div>
            </div>

            {isImporting && (
              <div className="px-6 py-3 border-t border-zinc-800 bg-zinc-900/95">
                <div className="flex items-center justify-between mb-1.5 text-[11px]">
                  <span className="text-blue-300 flex items-center gap-1.5">
                    <Loader2 className="h-3 w-3 animate-spin" />
                    Copying media {importProgress.completed}/{importProgress.total}
                  </span>
                  <span className="text-zinc-500">
                    {importProgress.skipped} skipped
                  </span>
                </div>
                <div className="h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full bg-blue-500 transition-all duration-200"
                    style={{ width: `${importPercent}%` }}
                  />
                </div>
                {importProgress.currentName && (
                  <p className="text-[10px] text-zinc-500 truncate mt-1.5">
                    {importProgress.currentName}
                  </p>
                )}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}
