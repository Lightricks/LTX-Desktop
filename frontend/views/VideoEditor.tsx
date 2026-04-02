import React, { useState, useRef, useEffect, useCallback, useMemo } from 'react'
import {
  ChevronRight,
} from 'lucide-react'
import { useProjects } from '../contexts/ProjectContext'
import type { PendingIcLoraUpdate, PendingRetakeUpdate } from '../contexts/ProjectContext'
import { useKeyboardShortcuts } from '../contexts/KeyboardShortcutsContext'
import { useAppSettings } from '../contexts/AppSettingsContext'
import { useGeneration } from '../hooks/use-generation'
import { logger } from '../lib/logger'
import { Tooltip } from '../components/ui/tooltip'
import { Group, Panel, Separator, type PanelImperativeHandle } from 'react-resizable-panels'
import { ExportModal } from '../components/ExportModal'
import { MenuBar, type MenuDefinition } from '../components/MenuBar'
import { ImportTimelineModal } from '../components/ImportTimelineModal'
import type { Asset, Project, TimelineClip } from '../types/project'
import {
  AUTOSAVE_DELAY,
  type EditorLayout,
  DEFAULT_LAYOUT,
  LAYOUT_LIMITS,
  loadLayout, saveLayout,
} from './editor/video-editor-utils'
import { createInitialEditorState } from './editor/editor-state'
import {
  applyPendingClipTakeUpdate,
} from './editor/editor-actions'
import { getEditorModel, updatedProject } from './editor/editor-project-bridging'
import {
  selectActiveFocusArea,
  selectActiveTimelineInPoint,
  selectActiveTimelineOutPoint,
  selectClipPath,
  selectClips,
  selectCurrentTime,
  selectLayout,
  selectLiveAssetForClip,
  selectPixelsPerSecond,
  selectSelectedClipForProperties,
  selectSelectedClipIds,
  selectShowExportModal,
  selectShowImportTimelineModal,
  selectShowSourceMonitor,
  selectSourceSplitPercent,
  selectSubtitles,
  selectSubtitleTrackStyleIdx,
  selectTotalDuration,
} from './editor/editor-selectors'
import { VideoEditorAssetsPanel, type VideoEditorAssetsPanelHandle } from './editor/VideoEditorAssetsPanel'
import { VideoEditorTimelineControlPanel } from './editor/VideoEditorTimelineControlPanel'
import { ClipPropertiesPanel } from './editor/ClipPropertiesPanel'
import { SubtitlePropertiesPanel } from './editor/SubtitlePropertiesPanel'
import {
  VideoEditorSourceMonitor,
  type SourceKeyboardAction,
  type VideoEditorSourceMonitorHandle,
} from './editor/VideoEditorSourceMonitor'
import { ProgramMonitor, type ProgramMonitorHandle } from './editor/ProgramMonitor'
import { VideoEditorTimelineEditingPanel } from './editor/VideoEditorTimelineEditingPanel'
import { VideoEditorLayoutMenu } from './editor/VideoEditorLayoutMenu'
import { useEditorKeyboard } from './editor/useEditorKeyboard'
import { useRegeneration } from './editor/useRegeneration'
import { useBuildMenuDefinitions } from './editor/buildMenuDefinitions'
import { usePlaybackEngine } from './editor/usePlaybackEngine'
import { usePlaybackAudioSync } from './editor/usePlaybackAudioSync'
import { useSubtitleImportExport } from './editor/useSubtitleImportExport'
import { useEditorMediaImport } from './editor/useEditorMediaImport'
import { useTimelineXmlExport } from './editor/useTimelineXmlExport'
import {
  createEditorStore,
  EditorStoreProvider,
  useEditorActions,
  useEditorGetState,
  useEditorStore,
  type EditorStoreApi,
} from './editor/editor-store'
import { GenerationErrorDialog } from '../components/GenerationErrorDialog'
import { SubtitleTrackStyleEditor } from './editor/SubtitleTrackStyleEditor'

interface VideoEditorProps {
  currentProject: Project
  setCurrentProject: (project: Project) => void
  pendingRetakeUpdate: PendingRetakeUpdate | null
  pendingIcLoraUpdate: PendingIcLoraUpdate | null
}

interface VideoEditorWithStoreProps {
  currentProject: Project
  setCurrentProject: (project: Project) => void
}

export function VideoEditor(props: VideoEditorProps) {
  const { currentProject, pendingRetakeUpdate, pendingIcLoraUpdate } = props
  const editorStoreRef = useRef<EditorStoreApi | null>(null)

  if (!editorStoreRef.current) {
    const initialEditorModel = getEditorModel(currentProject)
    const retakeApplied = applyPendingClipTakeUpdate(initialEditorModel, pendingRetakeUpdate)
    const editorModel = applyPendingClipTakeUpdate(retakeApplied, pendingIcLoraUpdate)
    editorStoreRef.current = createEditorStore(createInitialEditorState(editorModel, loadLayout()))
  }

  const editorStore = editorStoreRef.current
  if (!editorStore) throw new Error('Editor store failed to initialize')

  return (
    <EditorStoreProvider store={editorStore}>
      <VideoEditorWithStore
        currentProject={props.currentProject}
        setCurrentProject={props.setCurrentProject}
      />
    </EditorStoreProvider>
  )
}

function VideoEditorWithStore({
  currentProject,
  setCurrentProject,
}: VideoEditorWithStoreProps) {
  const { 
    setCurrentTab, setGenSpaceEditImagePath, setGenSpaceEditMode, setGenSpaceAudioPath,
    setGenSpaceRetakeSource,
    setGenSpaceIcLoraSource,
  } = useProjects()

  const { activeLayout: kbLayout, isEditorOpen: isKbEditorOpen, setEditorOpen: setKbEditorOpen } = useKeyboardShortcuts()
  const { shouldVideoGenerateWithLtxApi, forceApiGenerations } = useAppSettings()
  const kbLayoutRef = useRef(kbLayout)
  kbLayoutRef.current = kbLayout
  const isKbEditorOpenRef = useRef(isKbEditorOpen)
  isKbEditorOpenRef.current = isKbEditorOpen
  
  // Generation hook for regenerating shots
  const {
    generate: regenGenerate,
    generateImage: regenGenerateImage,
    isGenerating: isRegenerating,
    progress: regenProgress,
    statusMessage: regenStatusMessage,
    videoPath: regenVideoPath,
    imagePath: regenImagePath,
    error: regenError,
    cancel: regenCancel,
    reset: regenReset,
  } = useGeneration()

  const gapGenerationApi = useMemo(() => ({
    generate: regenGenerate,
    generateImage: regenGenerateImage,
    videoPath: regenVideoPath,
    imagePath: regenImagePath,
    isGenerating: isRegenerating,
    progress: regenProgress,
    statusMessage: regenStatusMessage,
    cancel: regenCancel,
    reset: regenReset,
    error: regenError,
  }), [
    isRegenerating,
    regenCancel,
    regenError,
    regenGenerate,
    regenGenerateImage,
    regenImagePath,
    regenProgress,
    regenReset,
    regenStatusMessage,
    regenVideoPath,
  ])
  
  const currentProjectId = currentProject.id
  const actions = useEditorActions()
  const getEditorState = useEditorGetState()
  const editorModel = useEditorStore(state => state.editorModel)
  const subtitles = useEditorStore(selectSubtitles)
  const currentTime = useEditorStore(selectCurrentTime)
  const isPlaying = useEditorStore(state => state.session.transport.isPlaying)
  const selectedClipIds = useEditorStore(selectSelectedClipIds)
  const showPropertiesPanel = useEditorStore(state => state.session.ui.showPropertiesPanel)
  const showImportTimelineModal = useEditorStore(selectShowImportTimelineModal)
  const showExportModal = useEditorStore(selectShowExportModal)
  const layout = useEditorStore(selectLayout)
  const showSourceMonitor = useEditorStore(selectShowSourceMonitor)
  const activeFocusArea = useEditorStore(selectActiveFocusArea)
  const sourceSplitPercent = useEditorStore(selectSourceSplitPercent)
  const shuttleSpeed = useEditorStore(state => state.session.transport.shuttleSpeed)
  const selectedSubtitleId = useEditorStore(state => state.session.selection.subtitleId)
  const subtitleTrackStyleIdx = useEditorStore(selectSubtitleTrackStyleIdx)

  const editorModelRef = useRef(editorModel)
  editorModelRef.current = editorModel
  const currentProjectRef = useRef(currentProject)
  currentProjectRef.current = currentProject

  const bladeShiftHeldRef = useRef(false)
  const [bladeShiftHeld, setBladeShiftHeld] = useState(false)
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const held = e.shiftKey
      bladeShiftHeldRef.current = held
      setBladeShiftHeld(held)
    }
    window.addEventListener('keydown', onKey)
    window.addEventListener('keyup', onKey)
    return () => { window.removeEventListener('keydown', onKey); window.removeEventListener('keyup', onKey) }
  }, [])
  // EFFECTS HIDDEN: const [effectsSearchQuery, setEffectsSearchQuery] = useState('')

  // In/Out points
  // In/Out points stored per-timeline so they don't bleed across timelines
  // Timeline marks and play-in-out now come from the temporary root EditorState compatibility layer.
  
  const inPoint = useEditorStore(selectActiveTimelineInPoint)
  const outPoint = useEditorStore(selectActiveTimelineOutPoint)
  
  const setInPoint = useCallback((updater: (prev: number | null) => number | null) => {
    const currentInPoint = selectActiveTimelineInPoint(getEditorState())
    actions.setTimelineInPoint(updater(currentInPoint))
  }, [actions, getEditorState])
  
  const setOutPoint = useCallback((updater: (prev: number | null) => number | null) => {
    const currentOutPoint = selectActiveTimelineOutPoint(getEditorState())
    actions.setTimelineOutPoint(updater(currentOutPoint))
  }, [actions, getEditorState])
  
  // Dragging IN/OUT markers with mouse
  const [draggingMarker, setDraggingMarker] = useState<'timelineIn' | 'timelineOut' | null>(null)
  const draggingMarkerRef = useRef(draggingMarker)
  draggingMarkerRef.current = draggingMarker
  const markerDragOriginRef = useRef<'timeline' | 'scrubbar' | null>(null)
  const inPointRef = useRef(inPoint)
  inPointRef.current = inPoint
  const outPointRef = useRef(outPoint)
  outPointRef.current = outPoint

  // Clip properties panel collapsible sections

  // Resizable layout
  const leftPanelResizeRef = useRef<PanelImperativeHandle | null>(null)
  const rightPanelResizeRef = useRef<PanelImperativeHandle | null>(null)
  const timelinePanelResizeRef = useRef<PanelImperativeHandle | null>(null)
  const assetsPanelResizeRef = useRef<PanelImperativeHandle | null>(null)
  const assetsPanelActionsRef = useRef<VideoEditorAssetsPanelHandle | null>(null)
  const sourceMonitorActionsRef = useRef<VideoEditorSourceMonitorHandle | null>(null)
  const programMonitorActionsRef = useRef<ProgramMonitorHandle | null>(null)
  // JKL shuttle speed: -8, -4, -2, -1, 0, 1, 2, 4, 8

  // Timeline tab UI state
  // Open timeline tabs — only these appear in the tab bar above the timeline.
  // All timelines are always visible in the library panel on the left.
  
  // Dragging state
  const timelineRef = useRef<HTMLDivElement>(null)
  const trackContainerRef = useRef<HTMLDivElement>(null)
  const trackHeadersRef = useRef<HTMLDivElement>(null)
  const rulerScrollRef = useRef<HTMLDivElement>(null)
  const centerOnPlayheadRef = useRef(false) // Flag: center view on playhead after next zoom change

  // --- Performance refs: allow the rAF playback loop to sync video directly ---
  // These mirror React state so the hot loop doesn't depend on re-renders.
  const playbackTimeRef = useRef(0)            // authoritative time during playback
  const autoSaveTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  
  // Only sync ref ← state when NOT playing (during playback, ref is authoritative)
  useEffect(() => { if (!isPlaying) playbackTimeRef.current = currentTime }, [currentTime, isPlaying])
  
  const applyLayoutToPanels = useCallback((nextLayout: EditorLayout) => {
    leftPanelResizeRef.current?.resize(nextLayout.leftPanelWidth)
    timelinePanelResizeRef.current?.resize(nextLayout.timelineHeight)
    if (nextLayout.assetsHeight > 0) {
      assetsPanelResizeRef.current?.resize(nextLayout.assetsHeight)
    }
    if (showPropertiesPanel) {
      rightPanelResizeRef.current?.resize(nextLayout.rightPanelWidth)
    }
  }, [showPropertiesPanel])

  const updateLayoutField = useCallback((
    field: 'leftPanelWidth' | 'rightPanelWidth' | 'timelineHeight' | 'assetsHeight',
    inPixels: number,
  ) => {
    const value = Math.round(inPixels)
    const currentLayout = selectLayout(getEditorState())
    if (currentLayout[field] === value) return
    const next = { ...currentLayout, [field]: value }
    saveLayout(next)
    actions.setLayout(next)
  }, [actions, getEditorState])

  const handleApplyLayout = useCallback((nextLayout: EditorLayout) => {
    actions.setLayout(nextLayout)
    saveLayout(nextLayout)
    requestAnimationFrame(() => applyLayoutToPanels(nextLayout))
  }, [actions, applyLayoutToPanels])
  
  const handleResetLayout = useCallback(() => {
    const nextLayout = { ...DEFAULT_LAYOUT }
    actions.resetLayout()
    saveLayout(nextLayout)
    requestAnimationFrame(() => applyLayoutToPanels(nextLayout))
  }, [actions, applyLayoutToPanels])

  const { subtitleFileInputRef, handleImportSrt, handleExportSrt } = useSubtitleImportExport()
  const { handleExportTimelineXml } = useTimelineXmlExport()
  const selectedGapRef = useRef<{ trackIndex: number; startTime: number; endTime: number } | null>(null)
  const gapGenerateModeRef = useRef<'text-to-video' | 'image-to-video' | 'text-to-image' | null>(null)
  const clearSelectedGapRef = useRef<() => void>(() => {})
  const closeSelectedGapRef = useRef<() => void>(() => {})

  const openSourceAsset = useCallback((asset: Asset, opts?: { initialTime?: number; resetMarks?: boolean }) => {
    actions.setShowSourceMonitor(true)
    actions.setActiveFocusArea('source')
    if (sourceMonitorActionsRef.current) {
      sourceMonitorActionsRef.current.openAsset(asset, opts)
      return
    }
    requestAnimationFrame(() => sourceMonitorActionsRef.current?.openAsset(asset, opts))
  }, [actions])

  const pauseSourceMonitor = useCallback(() => {
    sourceMonitorActionsRef.current?.pause()
  }, [])

  const handleActivateSourceFocus = useCallback(() => {
    actions.setActiveFocusArea('source')
  }, [actions])

  const handleActivateTimelineFocus = useCallback(() => {
    actions.setActiveFocusArea('timeline')
  }, [actions])

  const handleSourcePreviewLayoutChanged = useCallback((nextLayout: Record<string, number>) => {
    const nextSourceSplit = nextLayout['editor-source-monitor-panel']
    if (typeof nextSourceSplit !== 'number') return
    if (Math.abs(selectSourceSplitPercent(getEditorState()) - nextSourceSplit) < 0.1) return
    actions.setSourceSplitPercent(nextSourceSplit)
  }, [actions, getEditorState])

  const handleProgramMonitorMarkerDrag = useCallback((v: React.SetStateAction<'timelineIn' | 'timelineOut' | null>) => {
    markerDragOriginRef.current = 'scrubbar'
    setDraggingMarker(v)
  }, [])

  const handleActivateTimelinePanelFocus = useCallback(() => {
    if (selectActiveFocusArea(getEditorState()) !== 'timeline') {
      pauseSourceMonitor()
      actions.setActiveFocusArea('timeline')
    }
  }, [actions, getEditorState, pauseSourceMonitor])

  useEffect(() => {
    if (isPlaying) pauseSourceMonitor()
  }, [isPlaying, pauseSourceMonitor])
  const { fileInputRef, handleImportFile } = useEditorMediaImport({
    currentProjectId,
  })

  const selectedClip = useEditorStore(selectSelectedClipForProperties)
  const totalDuration = useEditorStore(selectTotalDuration)
  const pixelsPerSecond = useEditorStore(selectPixelsPerSecond)

  // Global mousemove/mouseup for dragging IN/OUT markers
  useEffect(() => {
    if (!draggingMarker) return

    const handleMouseMove = (e: MouseEvent) => {
      const marker = draggingMarkerRef.current
      if (!marker) return

      if (marker === 'timelineIn' || marker === 'timelineOut') {
        let time = 0
        const origin = markerDragOriginRef.current
        const rulerEl = timelineRef.current
        const progScrub = document.getElementById('program-scrub-bar')
        if (origin === 'scrubbar' && progScrub) {
          const rect = progScrub.getBoundingClientRect()
          const pct = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width))
          time = pct * totalDuration
        } else if (rulerEl) {
          const rect = rulerEl.getBoundingClientRect()
          const scrollLeft = rulerScrollRef.current?.scrollLeft ?? 0
          const px = e.clientX - rect.left + scrollLeft
          time = Math.max(0, px / pixelsPerSecond)
        } else if (progScrub) {
          const rect = progScrub.getBoundingClientRect()
          const pct = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width))
          time = pct * totalDuration
        }
        if (marker === 'timelineIn' && outPointRef.current !== null) {
          time = Math.min(time, outPointRef.current - 0.01)
        }
        if (marker === 'timelineOut' && inPointRef.current !== null) {
          time = Math.max(time, inPointRef.current + 0.01)
        }
        time = Math.max(0, Math.min(time, totalDuration))
        if (marker === 'timelineIn') {
          setInPoint(() => time)
        } else {
          setOutPoint(() => time)
        }
      }
    }

    const handleMouseUp = () => {
      markerDragOriginRef.current = null
      setDraggingMarker(null)
    }

    window.addEventListener('mousemove', handleMouseMove)
    window.addEventListener('mouseup', handleMouseUp)
    return () => {
      window.removeEventListener('mousemove', handleMouseMove)
      window.removeEventListener('mouseup', handleMouseUp)
    }
  }, [draggingMarker, pixelsPerSecond, totalDuration, setInPoint, setOutPoint])

  // Dynamic minimum zoom: at min zoom the whole timeline fits in view
  // Falls back to 0.05 if container isn't mounted yet
  const getMinZoom = useCallback(() => {
    const container = trackContainerRef.current
    if (!container || totalDuration <= 0) return 0.05
    const containerWidth = container.clientWidth - 20
    return Math.min(0.5, Math.max(0.01, containerWidth / (totalDuration * 100)))
  }, [totalDuration])
  const getMinZoomRef = useRef(getMinZoom)
  getMinZoomRef.current = getMinZoom
  
  const hasMountedRef = useRef(false)
  useEffect(() => {
    if (!hasMountedRef.current) {
      hasMountedRef.current = true
      return
    }
    if (autoSaveTimerRef.current) clearTimeout(autoSaveTimerRef.current)
    autoSaveTimerRef.current = setTimeout(() => {
      setCurrentProject(updatedProject(currentProjectRef.current, editorModelRef.current))
    }, AUTOSAVE_DELAY)

    return () => {
      if (autoSaveTimerRef.current) clearTimeout(autoSaveTimerRef.current)
    }
  }, [editorModel, setCurrentProject])

  useEffect(() => {
    return () => {
      if (autoSaveTimerRef.current) clearTimeout(autoSaveTimerRef.current)
      setCurrentProject(updatedProject(currentProjectRef.current, editorModelRef.current))
    }
  }, [setCurrentProject])
  
  // --- Core timeline logic ---

  const deleteAssetActionRef = useRef<() => void>(() => {})
  const sourceDispatchRef = useRef<(action: SourceKeyboardAction) => void>(() => {})
  const sourcePauseRef = useRef<() => void>(() => {})
  const fitToViewRef = useRef<() => void>(() => {})
  const toggleFullscreenRef = useRef<() => void>(() => {})
  const insertEditRef = useRef<() => void>(() => {})
  const overwriteEditRef = useRef<() => void>(() => {})
  const matchFrameRef = useRef<() => void>(() => {})
  

  // Regeneration hook (state + logic extracted)
  const {
    regeneratingAssetId,
    handleRegenerate, handleCancelRegeneration,
    regenerationPreError, dismissRegenerationPreError,
  } = useRegeneration({
    projectId: currentProjectId,
    regenGenerate, regenGenerateImage,
    regenVideoPath, regenImagePath,
    isRegenerating,
    regenCancel, regenReset, regenError,
    shouldVideoGenerateWithLtxApi,
  })
  const canUseIcLora = !forceApiGenerations
  sourceDispatchRef.current = (action) => sourceMonitorActionsRef.current?.dispatchKeyboardAction(action)
  sourcePauseRef.current = () => sourceMonitorActionsRef.current?.pause()
  
  useEditorKeyboard({
    refs: {
      kbLayoutRef,
      isKbEditorOpenRef,
      getState: getEditorState,
      playbackTimeRef,
      sourceDispatchRef,
      sourcePauseRef,
      centerOnPlayheadRef,
      getMinZoomRef,
      gapGenerateModeRef,
      clearSelectedGapRef,
      closeSelectedGapRef,
      fitToViewRef,
      toggleFullscreenRef,
      insertEditRef,
      overwriteEditRef,
      matchFrameRef,
    },
    context: {
      deleteAssetActionRef,
    },
  })
  
  // Playback engine (extracted hook)
  usePlaybackEngine({
    playbackTimeRef,
  })

  usePlaybackAudioSync({
    playbackTimeRef,
  })

  deleteAssetActionRef.current = () => {
    assetsPanelActionsRef.current?.deleteAsset()
  }
  
  insertEditRef.current = () => sourceDispatchRef.current('edit.insertEdit')
  overwriteEditRef.current = () => sourceDispatchRef.current('edit.overwriteEdit')

  // --- Match Frame: load clip under playhead into source monitor at corresponding frame ---
  const handleMatchFrame = useCallback(() => {
    const editorState = getEditorState()
    const ct = selectCurrentTime(editorState)
    const timelineClips = selectClips(editorState)
    const currentSelectedClipIds = selectSelectedClipIds(editorState)
    // Find clips under the playhead
    const clipsUnderPlayhead = timelineClips.filter(c =>
      ct >= c.startTime && ct < c.startTime + c.duration &&
      (c.type === 'video' || c.type === 'audio' || c.type === 'image')
    )
    if (clipsUnderPlayhead.length === 0) return

    // Prefer the selected clip if it's under the playhead, otherwise pick the topmost (lowest trackIndex = highest video track)
    let targetClip = clipsUnderPlayhead.find(c => currentSelectedClipIds.has(c.id))
    if (!targetClip) {
      targetClip = clipsUnderPlayhead.sort((a, b) => a.trackIndex - b.trackIndex)[0]
    }

    // Find the source asset
    const asset = selectLiveAssetForClip(editorState, targetClip)
    if (!asset) return

    // Compute source time accounting for trim and speed
    const clipOffset = ct - targetClip.startTime
    const speed = targetClip.speed || 1
    let srcTime: number
    if (targetClip.reversed) {
      const assetDuration = asset.duration || targetClip.duration
      srcTime = assetDuration - (targetClip.trimEnd || 0) - clipOffset * speed
    } else {
      srcTime = (targetClip.trimStart || 0) + clipOffset * speed
    }
    srcTime = Math.max(0, Math.min(srcTime, asset.duration || Infinity))

    openSourceAsset(asset, { initialTime: srcTime, resetMarks: true })
  }, [getEditorState, openSourceAsset])
  matchFrameRef.current = handleMatchFrame

  const handleTimelineTabContextMenu = (e: React.MouseEvent, timelineId: string) => {
    e.preventDefault()
    window.dispatchEvent(new CustomEvent('video-editor:open-timeline-menu', {
      detail: { timelineId, x: e.clientX, y: e.clientY },
    }))
  }
  
  // Extract a frame from a clip at the current playhead position
  // Returns a file:// URL (videos: extracted via ffmpeg, images: already on disk)
  const extractCurrentFrame = useCallback(async (clip: TimelineClip): Promise<string | null> => {
    try {
      const editorState = getEditorState()
      const clipPath = selectClipPath(editorState, clip)
      if (!clipPath) return null

      if (clip.type === 'video') {
        const seekTime = Math.max(0, selectCurrentTime(editorState) - clip.startTime) * clip.speed + clip.trimStart
        const result = await window.electronAPI.extractVideoFrame({ videoPath: clipPath, seekTime, width: 1024, quality: 2 })
        return result.path
      }
      return clipPath
    } catch (err) {
      logger.error(`Failed to extract frame: ${err}`)
      return null
    }
  }, [getEditorState])

  // Capture a frame and send to Gen Space for video generation (I2V)
  const handleCaptureFrameForVideo = useCallback(async (clip: TimelineClip) => {
    const framePath = await extractCurrentFrame(clip)
    if (!framePath) return
    setGenSpaceEditMode('video')
    setGenSpaceEditImagePath(framePath)
    setCurrentTab('gen-space')
  }, [extractCurrentFrame, setGenSpaceEditImagePath, setGenSpaceEditMode, setCurrentTab])

  // Send an image clip directly to Gen Space I2V mode
  const handleCreateVideoFromImage = useCallback((clip: TimelineClip) => {
    const imagePath = selectClipPath(getEditorState(), clip)
    if (!imagePath) return
    setGenSpaceEditMode('video')
    setGenSpaceEditImagePath(imagePath)
    setCurrentTab('gen-space')
  }, [getEditorState, setCurrentTab, setGenSpaceEditImagePath, setGenSpaceEditMode])

  // Navigate to Gen Space with audio pre-populated for A2V
  const handleCreateVideoFromAudio = useCallback((clip: TimelineClip) => {
    const audioPath = selectClipPath(getEditorState(), clip)
    if (!audioPath) return
    setGenSpaceAudioPath(audioPath)
    setCurrentTab('gen-space')
  }, [getEditorState, setCurrentTab, setGenSpaceAudioPath])

  const handleRetakeClip = useCallback((clip: TimelineClip) => {
    const liveAsset = selectLiveAssetForClip(getEditorState(), clip)
    if (!liveAsset) return

    const takeIndex = clip.takeIndex ?? liveAsset.activeTakeIndex
    let takePath = liveAsset.path
    if (liveAsset.takes && liveAsset.takes.length > 0 && takeIndex !== undefined) {
      const idx = Math.max(0, Math.min(takeIndex, liveAsset.takes.length - 1))
      takePath = liveAsset.takes[idx].path
    }

    const linkedIds = new Set(clip.linkedClipIds || [])
    linkedIds.add(clip.id)

    setGenSpaceRetakeSource({
      videoPath: takePath,
      clipId: clip.id,
      assetId: liveAsset.id,
      linkedClipIds: [...linkedIds],
      duration: clip.duration || liveAsset.duration,
    })
    setCurrentTab('gen-space')
  }, [getEditorState, setCurrentTab, setGenSpaceRetakeSource])

  const handleICLoraClip = useCallback((clip: TimelineClip) => {
    if (!canUseIcLora) return
    const liveAsset = selectLiveAssetForClip(getEditorState(), clip)
    if (!liveAsset) return

    const takeIndex = clip.takeIndex ?? liveAsset.activeTakeIndex
    let takePath = liveAsset.path
    if (liveAsset.takes && liveAsset.takes.length > 0 && takeIndex !== undefined) {
      const idx = Math.max(0, Math.min(takeIndex, liveAsset.takes.length - 1))
      takePath = liveAsset.takes[idx].path
    }

    const linkedIds = new Set(clip.linkedClipIds || [])
    linkedIds.add(clip.id)

    setGenSpaceIcLoraSource({
      videoPath: takePath,
      clipId: clip.id,
      assetId: liveAsset.id,
      linkedClipIds: [...linkedIds],
    })
    setCurrentTab('gen-space')
  }, [canUseIcLora, getEditorState, setCurrentTab, setGenSpaceIcLoraSource])

  // Populate fullscreen ref for keyboard handler
  toggleFullscreenRef.current = () => {
    programMonitorActionsRef.current?.toggleFullscreen()
  }

  
  // Menu bar definitions (extracted)
  const menuDefinitions: MenuDefinition[] = useBuildMenuDefinitions({
    kbLayout,
    fileInputRef, subtitleFileInputRef,
    handleExportTimelineXml, handleExportSrt,
    handleInsertEdit: () => insertEditRef.current(),
    handleOverwriteEdit: () => overwriteEditRef.current(),
    handleMatchFrame: () => matchFrameRef.current(),
    setKbEditorOpen,
    fitToViewRef,
    canUseIcLora, onICLoraClip: handleICLoraClip,
    handleResetLayout,
  })

  const renderProgramMonitorPane = (showHeader: boolean) => (
    <div
      className={`flex h-full min-h-0 min-w-0 flex-col ${activeFocusArea === 'timeline' ? 'ring-2 ring-blue-500 ring-inset' : ''}`}
      onMouseDown={handleActivateTimelineFocus}
    >
      {showHeader && (
        <div className="h-7 bg-zinc-900 border-b border-zinc-800 flex items-center px-3 flex-shrink-0">
          <span className="text-[11px] font-semibold text-zinc-400 tracking-wide">Timeline Viewer</span>
        </div>
      )}
      <ProgramMonitor
        ref={programMonitorActionsRef}
        playbackTimeRef={playbackTimeRef}
        setDraggingMarker={handleProgramMonitorMarkerDrag}
        kbLayout={kbLayout}
      />
    </div>
  )

  // --- Render ---
  
  return (
    <div className="h-full flex flex-col overflow-hidden">
      {/* Menu Bar */}
      <MenuBar
        menus={menuDefinitions}
        rightContent={(
          <VideoEditorLayoutMenu
            currentLayout={layout}
            onApplyLayout={handleApplyLayout}
            onResetLayout={handleResetLayout}
          />
        )}
      />
      {/* Main Content */}
      <div className="flex-1 overflow-hidden min-h-0">
        <Group orientation="horizontal" className="h-full w-full">
          <Panel
            id="editor-left-panel"
            panelRef={leftPanelResizeRef}
            defaultSize={layout.leftPanelWidth}
            minSize={LAYOUT_LIMITS.leftPanelWidth.min}
            maxSize={LAYOUT_LIMITS.leftPanelWidth.max}
            groupResizeBehavior="preserve-pixel-size"
            onResize={(size, _id, prev) => {
              if (!prev) return
              updateLayoutField('leftPanelWidth', size.inPixels)
            }}
          >
            <Group orientation="vertical" className="h-full">
              <Panel
                id="editor-assets-panel"
                panelRef={assetsPanelResizeRef}
                defaultSize={layout.assetsHeight > 0 ? layout.assetsHeight : '60%'}
                minSize={LAYOUT_LIMITS.assetsHeight.min}
                maxSize={LAYOUT_LIMITS.assetsHeight.max}
                groupResizeBehavior="preserve-pixel-size"
                onResize={(size, _id, prev) => {
                  if (!prev) return
                  updateLayoutField('assetsHeight', size.inPixels)
                }}
                className="min-h-0"
              >
                <VideoEditorAssetsPanel
                  ref={assetsPanelActionsRef}
                  openSourceAsset={openSourceAsset}
                  handleImportFile={handleImportFile}
                  handleRegenerate={handleRegenerate}
                  handleCancelRegeneration={handleCancelRegeneration}
                  isRegenerating={isRegenerating}
                  regeneratingAssetId={regeneratingAssetId}
                  regenProgress={regenProgress}
                  regenStatusMessage={regenStatusMessage}
                />
              </Panel>
              <Separator className="h-1 flex-shrink-0 cursor-row-resize bg-transparent hover:bg-blue-500/40 active:bg-blue-500/60 transition-colors relative z-10" />
              <Panel minSize={LAYOUT_LIMITS.assetsHeight.min} className="min-h-0">
                <VideoEditorTimelineControlPanel
                  handleTimelineTabContextMenu={handleTimelineTabContextMenu}
                />
              </Panel>
            </Group>
          </Panel>
          <Separator className="w-1 flex-shrink-0 cursor-col-resize bg-transparent hover:bg-blue-500/40 active:bg-blue-500/60 transition-colors relative z-10" />
          <Panel id="editor-main-panel" className="min-w-0">
      {/* Main Editor Area */}
      <div className="h-full min-w-0 flex flex-col overflow-hidden">
        <Group orientation="vertical" className="h-full min-h-0">
        <Panel minSize={20} className="min-h-0">
        <div className="h-full min-h-0 flex flex-col">
        {/* Preview Area (optionally split into Clip Viewer + Timeline Viewer) */}
        <div className="flex-1 min-h-0 min-w-0">
          {showSourceMonitor ? (
            <Group
              id="editor-preview-group"
              orientation="horizontal"
              className="h-full w-full"
              defaultLayout={{
                'editor-source-monitor-panel': sourceSplitPercent,
                'editor-program-monitor-panel': 100 - sourceSplitPercent,
              }}
              onLayoutChanged={handleSourcePreviewLayoutChanged}
            >
              <Panel
                id="editor-source-monitor-panel"
                defaultSize={`${sourceSplitPercent}%`}
                minSize="20%"
                maxSize="80%"
                className="min-h-0 min-w-0"
              >
                <div
                  className={`flex h-full min-h-0 ${activeFocusArea === 'source' ? 'ring-2 ring-blue-500 ring-inset' : 'border-r border-zinc-800'}`}
                  onMouseDown={handleActivateSourceFocus}
                >
                  <VideoEditorSourceMonitor
                    ref={sourceMonitorActionsRef}
                  />
                </div>
              </Panel>
              <Separator className="w-1.5 cursor-col-resize bg-transparent hover:bg-blue-500/40 active:bg-blue-500/60 transition-colors relative z-10" />
              <Panel
                id="editor-program-monitor-panel"
                defaultSize={`${100 - sourceSplitPercent}%`}
                minSize="20%"
                className="min-h-0 min-w-0"
              >
                {renderProgramMonitorPane(true)}
              </Panel>
            </Group>
          ) : (
            renderProgramMonitorPane(false)
          )}
        </div> {/* end split preview area */}
        
        {/* Timeline Info Bar (shuttle indicator only — timecode moved to ruler area) */}
        {shuttleSpeed !== 0 && (
          <div className="h-6 bg-zinc-900 border-t border-zinc-800 flex items-center px-4">
            <div className={`px-2 py-0.5 rounded text-xs font-mono font-bold ${
              shuttleSpeed < 0 ? 'bg-orange-600/20 text-orange-400' : 'bg-blue-600/20 text-blue-400'
            }`}>
              {shuttleSpeed < 0 ? '◀' : '▶'}{' '}{Math.abs(shuttleSpeed)}x
            </div>
          </div>
        )}
        </div>
        </Panel>
        <Separator className="h-1 flex-shrink-0 cursor-row-resize bg-transparent hover:bg-blue-500/40 active:bg-blue-500/60 transition-colors relative z-10" />
        <Panel
          id="editor-timeline-panel"
          panelRef={timelinePanelResizeRef}
          defaultSize={layout.timelineHeight}
          minSize={LAYOUT_LIMITS.timelineHeight.min}
          maxSize={LAYOUT_LIMITS.timelineHeight.max}
          groupResizeBehavior="preserve-pixel-size"
          onResize={(size, _id, prev) => {
            if (!prev) return
            updateLayoutField('timelineHeight', size.inPixels)
          }}
          className="min-h-0"
        >
        <div className="h-full min-h-0" onMouseDown={handleActivateTimelinePanelFocus}>
          <VideoEditorTimelineEditingPanel
            currentProjectId={currentProjectId}
            playbackTimeRef={playbackTimeRef}
            centerOnPlayheadRef={centerOnPlayheadRef}
            getMinZoom={getMinZoom}
            canUseIcLora={canUseIcLora}
            handleICLoraClip={handleICLoraClip}
            kbLayout={kbLayout}
            handleExportTimelineXml={handleExportTimelineXml}
            subtitleFileInputRef={subtitleFileInputRef}
            handleImportSrt={handleImportSrt}
            handleExportSrt={handleExportSrt}
            gapGenerationApi={gapGenerationApi}
            selectedGapRefBridge={selectedGapRef}
            gapGenerateModeRefBridge={gapGenerateModeRef}
            clearSelectedGapRefBridge={clearSelectedGapRef}
            closeSelectedGapRefBridge={closeSelectedGapRef}
            timelineRefBridge={timelineRef}
            trackContainerRefBridge={trackContainerRef}
            trackHeadersRefBridge={trackHeadersRef}
            rulerScrollRefBridge={rulerScrollRef}
            markerDragOriginRef={markerDragOriginRef}
            setDraggingMarker={setDraggingMarker}
            bladeShiftHeld={bladeShiftHeld}
            onTimelinePanelContextMenu={handleTimelineTabContextMenu}
            fitToViewRef={fitToViewRef}
            onRevealAsset={(assetId) => assetsPanelActionsRef.current?.revealAsset(assetId)}
            onCreateVideoFromImage={handleCreateVideoFromImage}
            onCaptureFrameForVideo={handleCaptureFrameForVideo}
            onCreateVideoFromAudio={handleCreateVideoFromAudio}
            handleRegenerate={handleRegenerate}
            handleRetakeClip={handleRetakeClip}
            handleCancelRegeneration={handleCancelRegeneration}
            isRegenerating={isRegenerating}
            regenProgress={regenProgress}
          />
        </div>
        </Panel>
        </Group>
      </div>
          </Panel>
          {showPropertiesPanel && (
            <>
              <Separator className="w-1 flex-shrink-0 cursor-col-resize bg-transparent hover:bg-blue-500/40 active:bg-blue-500/60 transition-colors relative z-10" />
              <Panel
                id="editor-properties-panel"
                panelRef={rightPanelResizeRef}
                defaultSize={layout.rightPanelWidth}
                minSize={LAYOUT_LIMITS.rightPanelWidth.min}
                maxSize={LAYOUT_LIMITS.rightPanelWidth.max}
                groupResizeBehavior="preserve-pixel-size"
                onResize={(size, _id, prev) => {
                  if (!prev) return
                  updateLayoutField('rightPanelWidth', size.inPixels)
                }}
                className="min-w-0"
              >
                <div className="relative h-full group">
                <Tooltip content="Collapse Properties Panel" side="left">
                  <button
                    className="absolute top-1/2 -translate-y-1/2 -left-3 w-6 h-8 bg-zinc-800 border border-zinc-700 rounded-l-md flex items-center justify-center text-zinc-400 hover:text-white hover:bg-zinc-700 transition-colors opacity-0 group-hover:opacity-100 z-20 cursor-pointer"
                    onClick={() => actions.setShowPropertiesPanel(false)}
                  >
                    <ChevronRight className="h-3.5 w-3.5" />
                  </button>
                </Tooltip>
                {/* Subtitle properties */}
                {selectedSubtitleId && selectedClipIds.size === 0 && (() => {
                  const selectedSub = subtitles.find(s => s.id === selectedSubtitleId)
                  if (!selectedSub) return null
                  return (
                    <SubtitlePropertiesPanel
                    />
                  )
                })()}
                {/* Clip properties */}
                {selectedClip ? (
                  <ClipPropertiesPanel
                    onCreateVideoFromImage={handleCreateVideoFromImage}
                  />
                ) : !selectedSubtitleId ? (
                  <div className="h-full bg-zinc-950 border-l border-zinc-800 flex flex-col items-center justify-center text-zinc-600 text-[12px]">
                    <span>No clip selected</span>
                  </div>
                ) : null}
                </div>
              </Panel>
            </>
          )}
        </Group>
      </div>
      
      <input
        ref={fileInputRef}
        type="file"
        accept="video/*,audio/*,image/*"
        multiple
        onChange={handleImportFile}
        className="hidden"
      />
      
      {/* Export Modal */}
      
      {showExportModal && (
        <ExportModal
          projectName={currentProject?.name || 'Untitled'}
        />
      )}
      
      {showImportTimelineModal && (
        <ImportTimelineModal
          projectId={currentProjectId}
        />
      )}
      
      {subtitleTrackStyleIdx !== null && (
        <SubtitleTrackStyleEditor />
      )}

      {(regenError || regenerationPreError) && (
        <GenerationErrorDialog
          error={(regenError || regenerationPreError)!}
          onDismiss={() => {
            if (regenError) regenReset()
            if (regenerationPreError) dismissRegenerationPreError()
          }}
        />
      )}
      </div>
  )
}
