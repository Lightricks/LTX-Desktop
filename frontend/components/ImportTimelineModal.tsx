import { useState, useEffect, useCallback, useRef } from 'react'
import { logger } from '../lib/logger'
import { 
  X, FileVideo, FileAudio, Image, Check, AlertTriangle, 
  FolderOpen, RefreshCw, Loader2, FileText, Link2, 
  ChevronDown, ChevronRight, Upload
} from 'lucide-react'
import type { ParsedTimeline, ParsedMediaRef } from '../lib/timeline-import'
import { parseTimeline } from '../lib/timeline-import'

interface ImportTimelineModalProps {
  isOpen: boolean
  onClose: () => void
  onImport: (timeline: ParsedTimeline) => void
}

export function ImportTimelineModal({ isOpen, onClose, onImport }: ImportTimelineModalProps) {
  const [step, setStep] = useState<'select' | 'parsing' | 'relink' | 'error'>('select')
  const [parsedTimeline, setParsedTimeline] = useState<ParsedTimeline | null>(null)
  const [mediaRefs, setMediaRefs] = useState<ParsedMediaRef[]>([])
  const [error, setError] = useState<string>('')
  const [expandedInfo, setExpandedInfo] = useState(true)
  const [expandedMedia, setExpandedMedia] = useState(true)
  const [isChecking, setIsChecking] = useState(false)
  const [isSearching, setIsSearching] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Reset when modal opens
  useEffect(() => {
    if (isOpen) {
      setStep('select')
      setParsedTimeline(null)
      setMediaRefs([])
      setError('')
    }
  }, [isOpen])

  // Check file existence for all media refs
  const checkMediaFiles = useCallback(async (refs: ParsedMediaRef[]) => {
    if (!window.electronAPI?.checkFilesExist) {
      // Not in Electron - mark all as not found
      return refs.map(r => ({ ...r, found: false }))
    }

    setIsChecking(true)
    try {
      const paths = refs
        .filter(r => r.resolvedPath && r.resolvedPath !== '')
        .map(r => r.resolvedPath)
      
      const results = await window.electronAPI.checkFilesExist(paths)
      
      return refs.map(r => ({
        ...r,
        found: r.resolvedPath ? (results[r.resolvedPath] || false) : false,
      }))
    } catch (err) {
      logger.error(`Error checking files: ${err}`)
      return refs
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

    try {
      const content = await file.text()
      const timeline = parseTimeline(content, filename)

      if (!timeline) {
        throw new Error('Could not parse timeline from file')
      }

      setParsedTimeline(timeline)

      // Check which media files exist
      const checkedRefs = await checkMediaFiles(timeline.mediaRefs)
      setMediaRefs(checkedRefs)

      setStep('relink')
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
      setStep('error')
    }

    // Reset input
    if (fileInputRef.current) fileInputRef.current.value = ''
  }, [checkMediaFiles])

  // Relink a single media file
  const handleRelinkFile = useCallback(async (mediaRefId: string) => {
    if (!window.electronAPI?.showOpenFileDialog) return

    const filePaths = await window.electronAPI.showOpenFileDialog({
      title: 'Relink Media File',
      filters: [
        { name: 'Media Files', extensions: ['mp4', 'mov', 'avi', 'mkv', 'mxf', 'mp3', 'wav', 'aac', 'flac', 'jpg', 'jpeg', 'png', 'tiff', 'exr', 'dpx'] },
        { name: 'All Files', extensions: ['*'] },
      ],
    })

    if (!filePaths || filePaths.length === 0) return

    const newPath = filePaths[0]
    setMediaRefs(prev => prev.map(r =>
      r.id === mediaRefId
        ? { ...r, relinkedPath: newPath, resolvedPath: newPath, found: true }
        : r
    ))
  }, [])

  // Search a directory for all missing files
  const handleSearchDirectory = useCallback(async () => {
    if (!window.electronAPI?.showOpenDirectoryDialog) return

    const dir = await window.electronAPI.showOpenDirectoryDialog({ title: 'Select folder to search for media files' })
    if (!dir) return

    setIsSearching(true)

    try {
      const missingRefs = mediaRefs.filter(r => !r.found)
      if (missingRefs.length === 0) return

      // Build list of filenames to search for
      const filenames: string[] = []
      for (const ref of missingRefs) {
        const filename = ref.name || ref.pathUrl.split('/').pop() || ''
        if (filename) filenames.push(filename)
      }

      if (filenames.length === 0) return

      // Use recursive directory search (searches subdirectories up to 10 levels deep)
      if (window.electronAPI.searchDirectoryForFiles) {
        const results = await window.electronAPI.searchDirectoryForFiles(dir, filenames)
        // results is { "filename.mp4" (lowercase): "C:\full\path\filename.mp4" }

        setMediaRefs(prev => prev.map(ref => {
          if (ref.found) return ref

          const filename = ref.name || ref.pathUrl.split('/').pop() || ''
          const foundPath = results[filename.toLowerCase()]

          if (foundPath) {
            return { ...ref, relinkedPath: foundPath, resolvedPath: foundPath, found: true }
          }
          return ref
        }))
      } else {
        // Fallback: check direct paths only (no recursive search)
        const searchPaths: string[] = []
        for (const filename of filenames) {
          const separator = dir.includes('\\') ? '\\' : '/'
          searchPaths.push(`${dir}${separator}${filename}`)
        }

        const results = await window.electronAPI.checkFilesExist(searchPaths)

        setMediaRefs(prev => prev.map(ref => {
          if (ref.found) return ref

          const filename = ref.name || ref.pathUrl.split('/').pop() || ''
          const separator = dir.includes('\\') ? '\\' : '/'
          const testPath = `${dir}${separator}${filename}`

          if (results[testPath]) {
            return { ...ref, relinkedPath: testPath, resolvedPath: testPath, found: true }
          }
          return ref
        }))
      }
    } catch (err) {
      logger.error(`Error searching directory: ${err}`)
    } finally {
      setIsSearching(false)
    }
  }, [mediaRefs])

  // Recheck all paths
  const handleRecheckAll = useCallback(async () => {
    const checked = await checkMediaFiles(mediaRefs)
    setMediaRefs(checked)
  }, [mediaRefs, checkMediaFiles])

  // Confirm import
  const handleConfirmImport = useCallback(() => {
    if (!parsedTimeline) return

    // Update the timeline with relinked paths
    const updatedTimeline: ParsedTimeline = {
      ...parsedTimeline,
      mediaRefs: mediaRefs,
    }

    onImport(updatedTimeline)
    onClose()
  }, [parsedTimeline, mediaRefs, onImport, onClose])

  if (!isOpen) return null

  const foundCount = mediaRefs.filter(r => r.found).length
  const totalCount = mediaRefs.length
  const allFound = foundCount === totalCount && totalCount > 0

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
                Premiere Pro XML, DaVinci Resolve XML/EDL, Final Cut Pro XML/FCPXML
              </p>
            </div>
          </div>
          <button onClick={onClose} className="p-1.5 rounded-lg hover:bg-zinc-800 text-zinc-500 hover:text-zinc-300 transition-colors">
            <X className="h-4 w-4" />
          </button>
        </div>

        {/* Body */}
        <div className="flex-1 overflow-auto p-6">
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
                <p className="text-xs text-zinc-600">Supports .xml (FCP 7 XML), .fcpxml, .edl (CMX 3600)</p>
              </div>

              <input
                ref={fileInputRef}
                type="file"
                accept=".xml,.fcpxml,.edl,.aaf"
                onChange={handleFileSelected}
                className="hidden"
              />

              <div className="bg-zinc-800/50 rounded-lg p-4 space-y-2">
                <h4 className="text-xs font-semibold text-zinc-400 uppercase tracking-wider">How to export from your NLE:</h4>
                <div className="space-y-1.5 text-[11px] text-zinc-500">
                  <p><span className="text-blue-400 font-medium">Premiere Pro:</span> File → Export → Final Cut Pro XML</p>
                  <p><span className="text-orange-400 font-medium">DaVinci Resolve:</span> File → Export Timeline → EDL (.edl) or FCP 7 XML (.xml)</p>
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
                        {parsedTimeline.format === 'fcp7xml' ? 'FCP 7 XML' : parsedTimeline.format === 'fcpxml' ? 'FCPXML' : parsedTimeline.format === 'edl' ? 'CMX 3600 EDL' : parsedTimeline.format}
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
                    Media Files ({foundCount}/{totalCount} linked)
                  </button>
                  <div className="flex items-center gap-1.5">
                    <button
                      onClick={handleSearchDirectory}
                      disabled={isSearching || allFound}
                      className="px-2.5 py-1 rounded-md bg-zinc-800 text-zinc-400 text-[10px] hover:bg-zinc-700 hover:text-zinc-300 flex items-center gap-1.5 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                      title="Search a folder for missing media"
                    >
                      {isSearching ? <Loader2 className="h-3 w-3 animate-spin" /> : <FolderOpen className="h-3 w-3" />}
                      Search Folder
                    </button>
                    <button
                      onClick={handleRecheckAll}
                      disabled={isChecking}
                      className="px-2.5 py-1 rounded-md bg-zinc-800 text-zinc-400 text-[10px] hover:bg-zinc-700 hover:text-zinc-300 flex items-center gap-1.5 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                      title="Recheck all file paths"
                    >
                      {isChecking ? <Loader2 className="h-3 w-3 animate-spin" /> : <RefreshCw className="h-3 w-3" />}
                      Recheck
                    </button>
                  </div>
                </div>

                {/* Status bar */}
                <div className="mb-2">
                  <div className="h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all duration-300 ${allFound ? 'bg-green-500' : foundCount > 0 ? 'bg-amber-500' : 'bg-red-500'}`}
                      style={{ width: `${totalCount > 0 ? (foundCount / totalCount) * 100 : 0}%` }}
                    />
                  </div>
                </div>

                {expandedMedia && (
                  <div className="space-y-1 max-h-[300px] overflow-auto rounded-lg border border-zinc-800">
                    {mediaRefs.map((ref, i) => {
                      const TypeIcon = ref.type === 'video' ? FileVideo : ref.type === 'audio' ? FileAudio : Image
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
                              {ref.found ? (
                                <Check className="h-3 w-3 text-green-400 flex-shrink-0" />
                              ) : (
                                <AlertTriangle className="h-3 w-3 text-red-400 flex-shrink-0" />
                              )}
                              <span className={`truncate font-medium ${ref.found ? 'text-zinc-300' : 'text-red-300'}`}>
                                {ref.name}
                              </span>
                            </div>
                            <p className="text-[9px] text-zinc-600 truncate mt-0.5">
                              {ref.relinkedPath || ref.resolvedPath || ref.pathUrl}
                            </p>
                          </div>
                          
                          {!ref.found && (
                            <button
                              onClick={() => handleRelinkFile(ref.id)}
                              className="flex-shrink-0 px-2 py-1 rounded bg-zinc-700 hover:bg-zinc-600 text-zinc-300 text-[10px] flex items-center gap-1 transition-colors"
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
          <div className="px-6 py-4 border-t border-zinc-800 flex items-center justify-between bg-zinc-900">
            <div className="text-[11px] text-zinc-500">
              {!allFound && totalCount > 0 && (
                <span className="text-amber-400">
                  {totalCount - foundCount} missing file{totalCount - foundCount !== 1 ? 's' : ''} — clips with missing media will be placeholders
                </span>
              )}
              {allFound && totalCount > 0 && (
                <span className="text-green-400">All media files found</span>
              )}
              {totalCount === 0 && (
                <span>No media references to link</span>
              )}
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={onClose}
                className="px-4 py-2 rounded-lg bg-zinc-800 text-zinc-300 text-sm hover:bg-zinc-700 transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handleConfirmImport}
                className="px-4 py-2 rounded-lg bg-blue-600 text-white text-sm hover:bg-blue-500 transition-colors font-medium flex items-center gap-2"
              >
                <Upload className="h-3.5 w-3.5" />
                Import Timeline
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
