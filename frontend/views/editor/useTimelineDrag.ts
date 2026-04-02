import { useState, useRef, useEffect, useCallback } from 'react'
import type { Asset, TimelineClip, Track } from '../../types/project'
import { resolveOverlaps, migrateClip, type ToolType } from './video-editor-utils'

interface DragPreviewPosition {
  startTime: number
  trackIndex: number
}

interface DragPreviewNode {
  el: HTMLElement
  transform: string
  zIndex: string
  willChange: string
}

export interface DraggingClipState {
  clipId: string
  startX: number
  startY: number
  originalStartTime: number
  originalTrackIndex: number
  originalPositions: Record<string, { startTime: number; trackIndex: number }>
  isDuplicate?: boolean
  altHeld?: boolean
}

export interface ResizingClipState {
  clipId: string
  edge: 'left' | 'right'
  startX: number
  originalStartTime: number
  originalDuration: number
  originalTrimStart: number
  originalTrimEnd: number
  tool: ToolType
  adjacentClipId?: string
  adjacentOrigDuration?: number
  adjacentOrigTrimStart?: number
  adjacentOrigTrimEnd?: number
  adjacentOrigStartTime?: number
}

export interface SlipSlideClipState {
  clipId: string
  tool: 'slip' | 'slide'
  startX: number
  originalTrimStart: number
  originalTrimEnd: number
  originalStartTime: number
  originalDuration: number
  prevClipId?: string
  prevOrigDuration?: number
  nextClipId?: string
  nextOrigStartTime?: number
  nextOrigDuration?: number
  nextOrigTrimStart?: number
}

interface UseTimelineDragParams {
  activeTool: ToolType
  setActiveTool: (tool: ToolType) => void
  lastTrimTool: ToolType
  setLastTrimTool: (tool: ToolType) => void
  pixelsPerSecond: number
  totalDuration: number
  clips: TimelineClip[]
  setClips: React.Dispatch<React.SetStateAction<TimelineClip[]>>
  tracks: Track[]
  selectedClipIds: Set<string>
  setSelectedClipIds: React.Dispatch<React.SetStateAction<Set<string>>>
  currentTime: number
  setCurrentTime: (time: number) => void
  setIsPlaying: (playing: boolean) => void
  snapEnabled: boolean
  resolveClipPath: (clip: TimelineClip | null) => string
  getMaxClipDuration: (clip: TimelineClip) => number
  addClipToTimeline: (asset: Asset, trackIndex: number, startTime?: number) => void
  assets: Asset[]
  timelines: any[]
  activeTimeline: any
  currentProjectId: string | null
  timelineRef: React.RefObject<HTMLDivElement>
  trackContainerRef: React.RefObject<HTMLDivElement>
  orderedTracks: { track: Track; realIndex: number; displayRow: number }[]
  getTrackHeight: (trackIndex: number) => number
  trackTopPx: (realTrackIndex: number, padding?: number) => number
  cutPoints: any[]
  splitClipAtPlayhead: (clipId: string, atTime?: number, batchClipIds?: string[]) => void
  setSelectedSubtitleId: (id: string | null) => void
  setSelectedGap: (gap: any) => void
  audioTrackHeight: number
  videoTrackHeight: number
  subtitleTrackHeight: number
}

export function useTimelineDrag(params: UseTimelineDragParams) {
  const {
    activeTool,
    pixelsPerSecond, totalDuration,
    clips, setClips, tracks,
    selectedClipIds, setSelectedClipIds,
    currentTime, setCurrentTime, setIsPlaying,
    snapEnabled, getMaxClipDuration, addClipToTimeline,
    assets, timelines, activeTimeline,
    timelineRef, trackContainerRef,
    orderedTracks, getTrackHeight, trackTopPx,
    splitClipAtPlayhead, setSelectedSubtitleId, setSelectedGap,
    audioTrackHeight, videoTrackHeight, subtitleTrackHeight,
  } = params

  const [draggingClip, setDraggingClip] = useState<DraggingClipState | null>(null)
  const [resizingClip, setResizingClip] = useState<ResizingClipState | null>(null)
  const [slipSlideClip, setSlipSlideClip] = useState<SlipSlideClipState | null>(null)
  const [lassoRect, setLassoRect] = useState<{ startX: number; startY: number; currentX: number; currentY: number } | null>(null)
  const lassoOriginRef = useRef<{ scrollLeft: number; containerLeft: number; containerTop: number } | null>(null)
  const dragPreviewRef = useRef<Record<string, DragPreviewPosition> | null>(null)
  const dragPreviewNodesRef = useRef<Map<string, DragPreviewNode>>(new Map())
  const dragPreviewRafRef = useRef<number | null>(null)

  const clearDragPreviewDom = useCallback(() => {
    if (dragPreviewRafRef.current !== null) {
      cancelAnimationFrame(dragPreviewRafRef.current)
      dragPreviewRafRef.current = null
    }
    for (const node of dragPreviewNodesRef.current.values()) {
      node.el.style.transform = node.transform
      node.el.style.zIndex = node.zIndex
      node.el.style.willChange = node.willChange
    }
    dragPreviewNodesRef.current.clear()
    dragPreviewRef.current = null
  }, [])

  // --- Ruler scrub: click + drag to scrub playhead ---
  

  const isScrubbing = useRef(false)
  const scrubFromEvent = useCallback((clientX: number) => {
    if (!timelineRef.current) return
    const rect = timelineRef.current.getBoundingClientRect()
    // getBoundingClientRect already accounts for the parent's scroll offset,
    // so clientX - rect.left gives the correct position within the timeline.
    const x = clientX - rect.left
    const time = x / pixelsPerSecond
    setCurrentTime(Math.max(0, Math.min(time, totalDuration)))
  }, [pixelsPerSecond, totalDuration])
  
  const handleRulerMouseDown = useCallback((e: React.MouseEvent) => {
    if (e.button !== 0) return // only left button
    e.preventDefault() // prevent text selection
    isScrubbing.current = true
    setIsPlaying(false) // pause playback while scrubbing
    scrubFromEvent(e.clientX)
    
    const onMove = (ev: MouseEvent) => {
      if (!isScrubbing.current) return
      ev.preventDefault()
      scrubFromEvent(ev.clientX)
    }
    const onUp = () => {
      isScrubbing.current = false
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup', onUp)
    }
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
  }, [activeTool, scrubFromEvent])
  
  
  
  // Helper: expand a set of clip IDs to include their linked counterparts (audio ↔ video)
  // Uses transitive expansion so all members of a linked group are found,
  // e.g. clicking A2 → finds Video → finds A1, selecting all three.
  const expandWithLinkedClips = useCallback((ids: Set<string>): Set<string> => {
    const expanded = new Set(ids)
    const queue = [...ids]
    while (queue.length > 0) {
      const id = queue.pop()!
      const c = clips.find(cl => cl.id === id)
      if (c?.linkedClipIds) {
        for (const lid of c.linkedClipIds) {
          if (!expanded.has(lid) && clips.some(cl => cl.id === lid)) {
            expanded.add(lid)
            queue.push(lid)
          }
        }
      }
    }
    return expanded
  }, [clips])
  
  const handleClipMouseDown = (e: React.MouseEvent, clip: TimelineClip) => {
    e.stopPropagation()
    
    // Prevent all interactions on locked tracks (except selection)
    const clipTrack = tracks[clip.trackIndex]
    if (clipTrack?.locked) {
      // Still allow selecting the clip visually
      setSelectedClipIds(expandWithLinkedClips(new Set([clip.id])))
      return
    }
    
    if (activeTool === 'blade') {
      const rect = (e.currentTarget as HTMLElement).getBoundingClientRect()
      const clickX = e.clientX - rect.left
      const clickTime = clip.startTime + (clickX / rect.width) * clip.duration
      
      setCurrentTime(clickTime)
      
      if (e.shiftKey) {
        // Shift+blade: cut ALL clips at this time across all unlocked tracks
        const clipIds = clips
          .filter(c =>
            clickTime > c.startTime + 0.1 &&
            clickTime < c.startTime + c.duration - 0.1 &&
            !tracks[c.trackIndex]?.locked
          )
          .map(c => c.id)
        if (clipIds.length > 0) {
          splitClipAtPlayhead(clipIds[0], clickTime, clipIds)
        }
      } else {
        // Normal blade: cut only the clicked clip
        splitClipAtPlayhead(clip.id, clickTime)
      }
      return
    }
    
    // --- Slip tool: shift source content within clip ---
    if (activeTool === 'slip') {
      setSelectedClipIds(expandWithLinkedClips(new Set([clip.id])))
      setSlipSlideClip({
        clipId: clip.id,
        tool: 'slip',
        startX: e.clientX,
        originalTrimStart: clip.trimStart,
        originalTrimEnd: clip.trimEnd,
        originalStartTime: clip.startTime,
        originalDuration: clip.duration,
      })
      return
    }
    
    // --- Slide tool: move clip, adjust neighbors ---
    if (activeTool === 'slide') {
      setSelectedClipIds(expandWithLinkedClips(new Set([clip.id])))
      
      // Find the previous and next clips on the same track
      const sameTrack = clips
        .filter(c => c.trackIndex === clip.trackIndex && c.id !== clip.id)
        .sort((a, b) => a.startTime - b.startTime)
      const prevClip = sameTrack.filter(c => c.startTime + c.duration <= clip.startTime + 0.05).pop()
      const nextClip = sameTrack.find(c => c.startTime >= clip.startTime + clip.duration - 0.05)
      
      setSlipSlideClip({
        clipId: clip.id,
        tool: 'slide',
        startX: e.clientX,
        originalTrimStart: clip.trimStart,
        originalTrimEnd: clip.trimEnd,
        originalStartTime: clip.startTime,
        originalDuration: clip.duration,
        prevClipId: prevClip?.id,
        prevOrigDuration: prevClip?.duration,
        nextClipId: nextClip?.id,
        nextOrigStartTime: nextClip?.startTime,
        nextOrigDuration: nextClip?.duration,
        nextOrigTrimStart: nextClip?.trimStart,
      })
      return
    }
    
    // --- Track Select Forward: select this clip + all clips to the right ---
    if (activeTool === 'trackForward') {
      const forwardClips = clips.filter(c => {
        if (e.shiftKey) {
          // Shift held: select forward on SAME track only
          return c.trackIndex === clip.trackIndex && c.startTime >= clip.startTime
        } else {
          // Default: select forward on ALL tracks (like Premiere)
          return c.startTime >= clip.startTime
        }
      })
      const forwardIds = expandWithLinkedClips(new Set(forwardClips.map(c => c.id)))
      setSelectedClipIds(forwardIds)
      setSelectedSubtitleId(null)
      setSelectedGap(null)
      
      // Start drag so the user can slide the whole forward selection
      const originalPositions: Record<string, { startTime: number; trackIndex: number }> = {}
      clips.filter(c => forwardIds.has(c.id)).forEach(c => {
        originalPositions[c.id] = { startTime: c.startTime, trackIndex: c.trackIndex }
      })
      setDraggingClip({
        clipId: clip.id,
        startX: e.clientX,
        startY: e.clientY,
        originalStartTime: clip.startTime,
        originalTrackIndex: clip.trackIndex,
        originalPositions,
      })
      return
    }
    
    if (activeTool === 'select' || activeTool === 'ripple' || activeTool === 'roll') {
      // Compute the effective selection BEFORE React processes the state update
      let effectiveSelection: Set<string>
      if (e.shiftKey) {
        // Shift+click: toggle clip in/out of multi-selection (toggle linked group together)
        effectiveSelection = new Set(selectedClipIds)
        if (effectiveSelection.has(clip.id)) {
          effectiveSelection.delete(clip.id)
          if (!e.altKey && clip.linkedClipIds) clip.linkedClipIds.forEach(lid => effectiveSelection.delete(lid))
        } else {
          effectiveSelection.add(clip.id)
          if (!e.altKey && clip.linkedClipIds) clip.linkedClipIds.forEach(lid => {
            if (clips.some(c => c.id === lid)) effectiveSelection.add(lid)
          })
        }
        setSelectedClipIds(effectiveSelection)
      } else if (e.altKey) {
        // Alt+click: select ONLY this specific clip, ignoring linked clips (like Premiere)
        if (selectedClipIds.has(clip.id)) {
          effectiveSelection = selectedClipIds
        } else {
          effectiveSelection = new Set([clip.id])
          setSelectedClipIds(effectiveSelection)
        }
      } else {
        // Normal click: select this clip + its linked clips
        effectiveSelection = expandWithLinkedClips(new Set([clip.id]))
        setSelectedClipIds(effectiveSelection)
      }
      
      // Only drag clips that are in the effective (visual) selection.
      // This allows moving just the video or audio part of a linked clip
      // when only that part is selected (e.g. via Alt+lasso). Links are preserved.
      const originalPositions: Record<string, { startTime: number; trackIndex: number }> = {}
      for (const c of clips) {
        if (effectiveSelection.has(c.id)) {
          originalPositions[c.id] = { startTime: c.startTime, trackIndex: c.trackIndex }
        }
      }
      // Always ensure the clicked clip is in the group
      if (!originalPositions[clip.id]) {
        originalPositions[clip.id] = { startTime: clip.startTime, trackIndex: clip.trackIndex }
      }
      
      // Set up dragging. Alt+drag duplication is deferred to first mouseMove
      // so that Alt+click (no drag) only changes selection without creating duplicates.
      setDraggingClip({
        clipId: clip.id,
        startX: e.clientX,
        startY: e.clientY,
        originalStartTime: clip.startTime,
        originalTrackIndex: clip.trackIndex,
        originalPositions,
        altHeld: e.altKey || undefined,
      })
    }
  }
  
  const handleMouseMove = useCallback((e: MouseEvent) => {
    // Handle lasso dragging
    if (lassoRect) {
      setLassoRect(prev => prev ? { ...prev, currentX: e.clientX, currentY: e.clientY } : null)
      return
    }
    
    if (!draggingClip || !trackContainerRef.current) return

    // Alt+drag: create duplicates on first significant movement (deferred from mouseDown)
    if (draggingClip.altHeld && !draggingClip.isDuplicate) {
      const dx = e.clientX - draggingClip.startX
      const dy = e.clientY - draggingClip.startY
      if (Math.abs(dx) < 3 && Math.abs(dy) < 3) return

      const idMap = new Map<string, string>()
      const duplicateClips: TimelineClip[] = []
      for (const c of clips) {
        if (!draggingClip.originalPositions[c.id]) continue
        const newId = `clip-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
        idMap.set(c.id, newId)
        duplicateClips.push({ ...c, id: newId, isRegenerating: false })
      }
      setClips(prev => [...prev, ...duplicateClips])

      const dupOrigPositions: Record<string, { startTime: number; trackIndex: number }> = {}
      for (const [oldId, pos] of Object.entries(draggingClip.originalPositions)) {
        const newId = idMap.get(oldId)
        if (newId) dupOrigPositions[newId] = { ...pos }
      }
      const newPrimaryId = idMap.get(draggingClip.clipId) || draggingClip.clipId
      setSelectedClipIds(new Set(Object.keys(dupOrigPositions)))
      setDraggingClip({
        ...draggingClip,
        clipId: newPrimaryId,
        originalPositions: dupOrigPositions,
        isDuplicate: true,
        altHeld: false,
      })
      return
    }
    
    const primaryClip = clips.find(c => c.id === draggingClip.clipId)
    if (!primaryClip) return
    
    const deltaX = e.clientX - draggingClip.startX
    const deltaY = e.clientY - draggingClip.startY
    
    // Compute the primary clip's new position
    let newStartTime = draggingClip.originalStartTime + deltaX / pixelsPerSecond
    newStartTime = Math.max(0, newStartTime)
    
    // Snap the primary clip (skip other clips in the drag group for snapping)
    const origPositions = draggingClip.originalPositions
    if (snapEnabled) {
      const snapThreshold = 0.2
      for (const otherClip of clips) {
        if (origPositions[otherClip.id]) continue // skip clips in the drag group
        
        if (Math.abs(newStartTime - otherClip.startTime) < snapThreshold) {
          newStartTime = otherClip.startTime
        }
        const otherEnd = otherClip.startTime + otherClip.duration
        if (Math.abs(newStartTime - otherEnd) < snapThreshold) {
          newStartTime = otherEnd
        }
        const clipEnd = newStartTime + primaryClip.duration
        if (Math.abs(clipEnd - otherClip.startTime) < snapThreshold) {
          newStartTime = otherClip.startTime - primaryClip.duration
        }
      }
      if (Math.abs(newStartTime - currentTime) < snapThreshold) {
        newStartTime = currentTime
      }
      if (Math.abs(newStartTime + primaryClip.duration - currentTime) < snapThreshold) {
        newStartTime = currentTime - primaryClip.duration
      }
    }
    
    // Use average track height for drag delta computation
    const avgTrackH = orderedTracks.length > 0
      ? orderedTracks.reduce((s, e) => s + (e.track.type === 'subtitle' ? subtitleTrackHeight : e.track.kind === 'audio' ? audioTrackHeight : videoTrackHeight), 0) / orderedTracks.length
      : 56
    const rawDisplayDelta = Math.round(deltaY / avgTrackH)
    
    // Convert display-row delta to real-trackIndex delta for the primary clip.
    // orderedTracks maps displayRow → realIndex, so we find the primary clip's
    // current display row, offset it by the Y-delta, and look up the real index.
    const primaryRealIndex = draggingClip.originalTrackIndex
    // (primaryDisplayRow, targetDisplayRow, targetRealIndex used for debugging; trackIndexDelta removed — per-clip resolution below)
    
    
    // Compute raw deltas relative to primary clip's original position
    let timeDelta = newStartTime - draggingClip.originalStartTime
    
    // Clamp time delta so no clip goes before time 0
    for (const orig of Object.values(origPositions)) {
      if (orig.startTime + timeDelta < 0) {
        timeDelta = -orig.startTime
      }
    }
    
    // Premiere-style linked clip movement (mirrored around divider):
    //
    // Display layout (orderedTracks):
    //   row 0: V3  (top)       ← away from divider
    //   row 1: V2
    //   row 2: V1              ← nearest to divider
    //   --- divider ---
    //   row 3: A1              ← nearest to divider
    //   row 4: A2
    //   row 5: A3  (bottom)    ← away from divider
    //
    // The video and audio sections MIRROR around the divider. When you drag a
    // video clip "down" (toward divider), the linked audio should move "up"
    // (also toward divider). When you drag "up" (away from divider), audio
    // moves "down" (also away from divider). They move symmetrically relative
    // to the divider, not in the same screen direction.
    //
    // Implementation: the primary clip (being dragged) uses rawDisplayDelta.
    // Linked clips of the OPPOSITE kind get the delta INVERTED.
    // Each clip is independently clamped within its kind — if one hits its
    // boundary the other can still move.
    
    const primaryTrack = tracks[primaryRealIndex]
    const primaryKind = primaryTrack?.kind || 'video'
    
    // Build per-kind ordered lists (in display order)
    const videoDisplayRows = orderedTracks
      .filter(e => e.track.kind === 'video' && e.track.type !== 'subtitle')
      .map(e => ({ displayRow: e.displayRow, realIndex: e.realIndex }))
    const audioDisplayRows = orderedTracks
      .filter(e => e.track.kind === 'audio')
      .map(e => ({ displayRow: e.displayRow, realIndex: e.realIndex }))
    
    // Helper: resolve a clip's target track within its own kind
    const resolveTrackForClip = (origTrackIndex: number): number => {
      if (rawDisplayDelta === 0) return origTrackIndex
      
      const origTrack = tracks[origTrackIndex]
      const clipKind = origTrack?.kind || 'video'
      const kindRows = clipKind === 'audio' ? audioDisplayRows : videoDisplayRows
      
      // Find this clip's position within its kind's ordered list
      const posInKind = kindRows.findIndex(r => r.realIndex === origTrackIndex)
      if (posInKind === -1) return origTrackIndex
      
      // Primary kind follows the drag direction; opposite kind mirrors (inverted delta)
      const effectiveDelta = clipKind === primaryKind ? rawDisplayDelta : -rawDisplayDelta
      
      const newPosInKind = Math.max(0, Math.min(kindRows.length - 1, posInKind + effectiveDelta))
      const newTrackIndex = kindRows[newPosInKind].realIndex
      
      // Check if target track is locked
      if (tracks[newTrackIndex]?.locked) return origTrackIndex
      
      return newTrackIndex
    }
    
    const previewById: Record<string, DragPreviewPosition> = {}
    for (const [clipId, orig] of Object.entries(origPositions)) {
      previewById[clipId] = {
        startTime: orig.startTime + timeDelta,
        trackIndex: resolveTrackForClip(orig.trackIndex),
      }
    }
    dragPreviewRef.current = previewById

    if (dragPreviewRafRef.current === null) {
      dragPreviewRafRef.current = requestAnimationFrame(() => {
        dragPreviewRafRef.current = null
        if (!trackContainerRef.current) return
        const preview = dragPreviewRef.current
        if (!preview) return

        for (const [clipId, target] of Object.entries(preview)) {
          const original = origPositions[clipId]
          if (!original) continue

          let node = dragPreviewNodesRef.current.get(clipId)
          if (!node || !node.el.isConnected) {
            const el = trackContainerRef.current.querySelector(`[data-clip-id="${clipId}"]`) as HTMLElement | null
            if (!el) continue
            node = {
              el,
              transform: el.style.transform,
              zIndex: el.style.zIndex,
              willChange: el.style.willChange,
            }
            dragPreviewNodesRef.current.set(clipId, node)
          }

          const xPx = (target.startTime - original.startTime) * pixelsPerSecond
          const yPx = trackTopPx(target.trackIndex, 4) - trackTopPx(original.trackIndex, 4)
          node.el.style.transform = `translate3d(${xPx}px, ${yPx}px, 0)`
          node.el.style.zIndex = '40'
          node.el.style.willChange = 'transform'
        }
      })
    }
  }, [draggingClip, clips, pixelsPerSecond, snapEnabled, tracks, currentTime, lassoRect, orderedTracks, trackContainerRef, trackTopPx])
  
  const handleMouseUp = useCallback((e?: MouseEvent | Event) => {
    // Finalize lasso selection
    if (lassoRect && trackContainerRef.current) {
      const origin = lassoOriginRef.current
      if (origin) {
        const container = trackContainerRef.current
        const scrollLeft = container.scrollLeft
        const scrollTop = container.scrollTop
        
        // Compute lasso rectangle in timeline-local coordinates
        const lx1 = Math.min(lassoRect.startX, lassoRect.currentX) - origin.containerLeft + scrollLeft
        const lx2 = Math.max(lassoRect.startX, lassoRect.currentX) - origin.containerLeft + scrollLeft
        const ly1 = Math.min(lassoRect.startY, lassoRect.currentY) - origin.containerTop + scrollTop
        const ly2 = Math.max(lassoRect.startY, lassoRect.currentY) - origin.containerTop + scrollTop
        
        // Convert to time/track
        const timeStart = lx1 / pixelsPerSecond
        const timeEnd = lx2 / pixelsPerSecond
        const newSelection = new Set<string>()
        for (const clip of clips) {
          const clipLeft = clip.startTime
          const clipRight = clip.startTime + clip.duration
          const th = getTrackHeight(clip.trackIndex)
          const clipTop = trackTopPx(clip.trackIndex) + 4
          const clipBottom = clipTop + (th - 8) // clip height = trackHeight - 8px padding
          
          // Check overlap between lasso rect and clip rect
          if (clipRight > timeStart && clipLeft < timeEnd && clipBottom > ly1 && clipTop < ly2) {
            newSelection.add(clip.id)
          }
        }
        // Alt/Option held: select only what's in the lasso (skip linked clips)
        const altHeld = e instanceof MouseEvent && e.altKey
        setSelectedClipIds(altHeld ? newSelection : expandWithLinkedClips(newSelection))
      }
      setLassoRect(null)
      lassoOriginRef.current = null
    }
    
    // Commit drag result once (positions + overlap resolution) so history records one drag step.
    if (draggingClip) {
      const preview = dragPreviewRef.current
      const originalPositions = draggingClip.originalPositions
      const movedIds = new Set(Object.keys(originalPositions))
      setClips(prev => {
        const positioned = prev.map(clip => {
          const target = preview?.[clip.id]
          if (!target) return clip
          return {
            ...clip,
            startTime: target.startTime,
            trackIndex: target.trackIndex,
          }
        })
        return resolveOverlaps(positioned, movedIds)
      })
      clearDragPreviewDom()
    }
    if (resizingClip) {
      setClips(prev => resolveOverlaps(prev, new Set([resizingClip.clipId])))
    }
    
    setDraggingClip(null)
    setResizingClip(null)
  }, [lassoRect, clips, pixelsPerSecond, draggingClip, resizingClip, setClips, getTrackHeight, trackTopPx, clearDragPreviewDom])
  
  const handleResizeMove = useCallback((e: MouseEvent) => {
    if (!resizingClip) return
    
    const clip = clips.find(c => c.id === resizingClip.clipId)
    if (!clip) return
    
    const deltaX = e.clientX - resizingClip.startX
    const deltaTime = deltaX / pixelsPerSecond
    const tool = resizingClip.tool
    
    // --- ROLL TRIM: move the edit point between two adjacent clips ---
    if (tool === 'roll' && resizingClip.adjacentClipId) {
      const adjClip = clips.find(c => c.id === resizingClip.adjacentClipId)
      if (!adjClip) return
      
      const linkedIds = new Set<string>(clip.linkedClipIds || [])
      const adjLinkedIds = new Set<string>(adjClip.linkedClipIds || [])
      let dt = deltaTime
      
      if (resizingClip.edge === 'right') {
        const maxExtend = Math.min(
          getMaxClipDuration(clip) - resizingClip.originalDuration,
          (resizingClip.adjacentOrigDuration ?? adjClip.duration) - 0.5,
        )
        const maxShrink = resizingClip.originalDuration - 0.5
        dt = Math.max(-maxShrink, Math.min(maxExtend, dt))
        
        const finalDur = Math.max(0.5, resizingClip.originalDuration + dt)
        
        setClips(prev => prev.map(c => {
          if (c.id === clip.id) {
            return { ...c, duration: finalDur }
          }
          if (linkedIds.has(c.id)) {
            return { ...c, duration: finalDur }
          }
          if (c.id === resizingClip.adjacentClipId) {
            const newStart = (resizingClip.adjacentOrigStartTime ?? c.startTime) + dt
            const newDur = (resizingClip.adjacentOrigDuration ?? c.duration) - dt
            const newTrimStart = (resizingClip.adjacentOrigTrimStart ?? c.trimStart) + dt * c.speed
            return { ...c, startTime: newStart, duration: Math.max(0.5, newDur), trimStart: Math.max(0, newTrimStart) }
          }
          if (adjLinkedIds.has(c.id)) {
            const newStart = (resizingClip.adjacentOrigStartTime ?? adjClip.startTime) + dt
            const newDur = (resizingClip.adjacentOrigDuration ?? adjClip.duration) - dt
            const newTrimStart = c.trimStart + dt * c.speed
            return { ...c, startTime: newStart, duration: Math.max(0.5, newDur), trimStart: Math.max(0, newTrimStart) }
          }
          return c
        }))
      } else {
        const maxExtend = Math.min(
          getMaxClipDuration(clip) - resizingClip.originalDuration,
          (resizingClip.adjacentOrigDuration ?? adjClip.duration) - 0.5,
        )
        const maxShrink = resizingClip.originalDuration - 0.5
        dt = Math.max(-maxExtend, Math.min(maxShrink, dt))
        
        const newStart = resizingClip.originalStartTime + dt
        const newDur = Math.max(0.5, resizingClip.originalDuration - dt)
        const newTrimStart = resizingClip.originalTrimStart + dt * clip.speed
        const adjFinalDur = Math.max(0.5, (resizingClip.adjacentOrigDuration ?? adjClip.duration) + dt)
        
        setClips(prev => prev.map(c => {
          if (c.id === clip.id) {
            return { ...c, startTime: newStart, duration: newDur, trimStart: Math.max(0, newTrimStart) }
          }
          if (linkedIds.has(c.id)) {
            const linkedTrimStart = c.trimStart + dt * c.speed
            return { ...c, startTime: newStart, duration: newDur, trimStart: Math.max(0, linkedTrimStart) }
          }
          if (c.id === resizingClip.adjacentClipId) {
            return { ...c, duration: adjFinalDur }
          }
          if (adjLinkedIds.has(c.id)) {
            return { ...c, duration: adjFinalDur }
          }
          return c
        }))
      }
      return
    }
    
    // --- RIPPLE TRIM: trim edge and shift all subsequent clips ---
    if (tool === 'ripple') {
      const linkedIds = new Set<string>(clip.linkedClipIds || [])
      if (resizingClip.edge === 'left') {
        let newStartTime = resizingClip.originalStartTime + deltaTime
        let newDuration = resizingClip.originalDuration - deltaTime
        
        if (newDuration < 0.5) { newDuration = 0.5; newStartTime = resizingClip.originalStartTime + resizingClip.originalDuration - 0.5 }
        if (newStartTime < 0) { newDuration = resizingClip.originalDuration + resizingClip.originalStartTime; newStartTime = 0 }
        
        const newTrimStart = resizingClip.originalTrimStart + (newStartTime - resizingClip.originalStartTime)
        const maxDur = getMaxClipDuration({ ...clip, trimStart: Math.max(0, newTrimStart) })
        newDuration = Math.min(newDuration, maxDur)
        
        const rippleDelta = newStartTime - resizingClip.originalStartTime
        const finalDuration = Math.max(0.5, newDuration)
        const finalTrimStart = Math.max(0, newTrimStart)
        
        setClips(prev => prev.map(c => {
          if (c.id === clip.id) {
            return { ...c, startTime: newStartTime, duration: finalDuration, trimStart: finalTrimStart }
          }
          if (linkedIds.has(c.id)) {
            const linkedNewTrimStart = c.trimStart + (newStartTime - resizingClip.originalStartTime)
            return { ...c, startTime: newStartTime, duration: finalDuration, trimStart: Math.max(0, linkedNewTrimStart) }
          }
          if (c.trackIndex === clip.trackIndex && c.id !== clip.id && c.startTime < resizingClip.originalStartTime) {
            return { ...c, startTime: Math.max(0, c.startTime + rippleDelta) }
          }
          return c
        }))
      } else {
        let newDuration = resizingClip.originalDuration + deltaTime
        newDuration = Math.max(0.5, newDuration)
        const maxDur = getMaxClipDuration(clip)
        newDuration = Math.min(newDuration, maxDur)
        
        const originalEnd = resizingClip.originalStartTime + resizingClip.originalDuration
        const newEnd = resizingClip.originalStartTime + newDuration
        const rippleDelta = newEnd - originalEnd
        const finalDuration = Math.max(0.5, newDuration)
        
        setClips(prev => prev.map(c => {
          if (c.id === clip.id) {
            return { ...c, duration: finalDuration }
          }
          if (linkedIds.has(c.id)) {
            return { ...c, duration: finalDuration }
          }
          if (c.trackIndex === clip.trackIndex && c.id !== clip.id && c.startTime >= originalEnd - 0.01) {
            return { ...c, startTime: Math.max(0, c.startTime + rippleDelta) }
          }
          return c
        }))
      }
      return
    }
    
    // --- NORMAL TRIM (select tool or any tool without special handling) ---
    if (resizingClip.edge === 'left') {
      let newStartTime = resizingClip.originalStartTime + deltaTime
      let newDuration = resizingClip.originalDuration - deltaTime
      
      if (newDuration < 0.5) {
        newDuration = 0.5
        newStartTime = resizingClip.originalStartTime + resizingClip.originalDuration - 0.5
      }
      
      if (newStartTime < 0) {
        newDuration = resizingClip.originalDuration + resizingClip.originalStartTime
        newStartTime = 0
      }
      
      if (snapEnabled) {
        const snapThreshold = 0.2
        if (Math.abs(newStartTime - currentTime) < snapThreshold) {
          const adjustment = currentTime - newStartTime
          newStartTime = currentTime
          newDuration -= adjustment
        }
        for (const otherClip of clips) {
          if (otherClip.id === clip.id) continue
          const otherEnd = otherClip.startTime + otherClip.duration
          if (Math.abs(newStartTime - otherEnd) < snapThreshold) {
            const adjustment = otherEnd - newStartTime
            newStartTime = otherEnd
            newDuration -= adjustment
          }
          if (Math.abs(newStartTime - otherClip.startTime) < snapThreshold) {
            const adjustment = otherClip.startTime - newStartTime
            newStartTime = otherClip.startTime
            newDuration -= adjustment
          }
        }
      }
      
      const newTrimStart = resizingClip.originalTrimStart + (newStartTime - resizingClip.originalStartTime)
      const maxDur = getMaxClipDuration({ ...clip, trimStart: Math.max(0, newTrimStart) })
      newDuration = Math.min(newDuration, maxDur)
      
      // Build set of linked clip IDs to also trim
      const linkedIds = new Set<string>(clip.linkedClipIds || [])
      const finalDuration = Math.max(0.5, newDuration)
      const finalTrimStart = Math.max(0, newTrimStart)
      
      setClips(prev => prev.map(c => {
        if (c.id === clip.id) {
          return { ...c, startTime: newStartTime, duration: finalDuration, trimStart: finalTrimStart }
        }
        if (linkedIds.has(c.id)) {
          const linkedNewTrimStart = c.trimStart + (newStartTime - resizingClip.originalStartTime)
          return { ...c, startTime: newStartTime, duration: finalDuration, trimStart: Math.max(0, linkedNewTrimStart) }
        }
        return c
      }))
    } else {
      let newDuration = resizingClip.originalDuration + deltaTime
      newDuration = Math.max(0.5, newDuration)
      
      if (snapEnabled) {
        const snapThreshold = 0.2
        const newEndTime = clip.startTime + newDuration
        
        if (Math.abs(newEndTime - currentTime) < snapThreshold) {
          newDuration = currentTime - clip.startTime
        }
        for (const otherClip of clips) {
          if (otherClip.id === clip.id) continue
          if (Math.abs(newEndTime - otherClip.startTime) < snapThreshold) {
            newDuration = otherClip.startTime - clip.startTime
          }
          const otherEnd = otherClip.startTime + otherClip.duration
          if (Math.abs(newEndTime - otherEnd) < snapThreshold) {
            newDuration = otherEnd - clip.startTime
          }
        }
      }
      
      const maxDur = getMaxClipDuration(clip)
      newDuration = Math.min(newDuration, maxDur)
      
      // Build set of linked clip IDs to also trim
      const linkedIds = new Set<string>(clip.linkedClipIds || [])
      const finalDuration = Math.max(0.5, newDuration)
      
      setClips(prev => prev.map(c => {
        if (c.id === clip.id) return { ...c, duration: finalDuration }
        if (linkedIds.has(c.id)) return { ...c, duration: finalDuration }
        return c
      }))
    }
  }, [resizingClip, clips, pixelsPerSecond, snapEnabled, currentTime, getMaxClipDuration])
  
  const handleResizeStart = (e: React.MouseEvent, clip: TimelineClip, edge: 'left' | 'right') => {
    e.stopPropagation()
    e.preventDefault()
    
    // Prevent resizing clips on locked tracks
    if (tracks[clip.trackIndex]?.locked) return
    
    setSelectedClipIds(expandWithLinkedClips(new Set([clip.id])))
    
    // For Roll trim, find the adjacent clip at the edit point
    let adjacentClip: TimelineClip | undefined
    if (activeTool === 'roll') {
      const clipEnd = clip.startTime + clip.duration
      if (edge === 'right') {
        // Find clip that starts right at (or very near) this clip's end on the same track
        adjacentClip = clips.find(c => c.id !== clip.id && c.trackIndex === clip.trackIndex && Math.abs(c.startTime - clipEnd) < 0.05)
      } else {
        // Find clip that ends right at (or very near) this clip's start on the same track
        adjacentClip = clips.find(c => c.id !== clip.id && c.trackIndex === clip.trackIndex && Math.abs((c.startTime + c.duration) - clip.startTime) < 0.05)
      }
    }
    
    setResizingClip({
      clipId: clip.id,
      edge,
      startX: e.clientX,
      originalStartTime: clip.startTime,
      originalDuration: clip.duration,
      originalTrimStart: clip.trimStart,
      originalTrimEnd: clip.trimEnd,
      tool: activeTool,
      adjacentClipId: adjacentClip?.id,
      adjacentOrigDuration: adjacentClip?.duration,
      adjacentOrigTrimStart: adjacentClip?.trimStart,
      adjacentOrigTrimEnd: adjacentClip?.trimEnd,
      adjacentOrigStartTime: adjacentClip?.startTime,
    })
  }
  
  useEffect(() => {
    if (draggingClip || lassoRect) {
      window.addEventListener('mousemove', handleMouseMove)
      window.addEventListener('mouseup', handleMouseUp)
      return () => {
        window.removeEventListener('mousemove', handleMouseMove)
        window.removeEventListener('mouseup', handleMouseUp)
      }
    }
  }, [draggingClip, lassoRect, handleMouseMove, handleMouseUp])

  useEffect(() => {
    return () => {
      clearDragPreviewDom()
    }
  }, [clearDragPreviewDom])
  
  useEffect(() => {
    if (resizingClip) {
      window.addEventListener('mousemove', handleResizeMove)
      window.addEventListener('mouseup', handleMouseUp)
      return () => {
        window.removeEventListener('mousemove', handleResizeMove)
        window.removeEventListener('mouseup', handleMouseUp)
      }
    }
  }, [resizingClip, handleResizeMove, handleMouseUp])
  
  // --- Slip/Slide mouse move handler ---
  const handleSlipSlideMove = useCallback((e: MouseEvent) => {
    if (!slipSlideClip) return
    
    const clip = clips.find(c => c.id === slipSlideClip.clipId)
    if (!clip) return
    
    const deltaX = e.clientX - slipSlideClip.startX
    const deltaTime = deltaX / pixelsPerSecond
    
    if (slipSlideClip.tool === 'slip') {
      // SLIP: shift source content within the clip (change trimStart/trimEnd, keep position)
      // Moving right = shift source earlier = increase trimStart, decrease trimEnd
      if (clip.type !== 'video' || !clip.asset?.duration) return
      
      const mediaDuration = clip.asset.duration
      const shiftAmount = deltaTime * clip.speed // convert to media time
      
      let newTrimStart = slipSlideClip.originalTrimStart + shiftAmount
      let newTrimEnd = slipSlideClip.originalTrimEnd - shiftAmount
      
      // Clamp so neither goes negative
      if (newTrimStart < 0) {
        newTrimEnd += newTrimStart
        newTrimStart = 0
      }
      if (newTrimEnd < 0) {
        newTrimStart += newTrimEnd
        newTrimEnd = 0
      }
      
      // Ensure trimStart + trimEnd + visible media ≤ total media duration
      const visibleMedia = clip.duration * clip.speed
      if (newTrimStart + visibleMedia + newTrimEnd > mediaDuration) {
        return // Can't slip further
      }
      
      setClips(prev => prev.map(c =>
        c.id === clip.id ? { ...c, trimStart: Math.max(0, newTrimStart), trimEnd: Math.max(0, newTrimEnd) } : c
      ))
    } else {
      // SLIDE: move clip in time, adjust neighbor durations to fill the space
      let newStartTime = slipSlideClip.originalStartTime + deltaTime
      
      // Clamp: can't go before prevClip's start (or 0)
      const minStart = slipSlideClip.prevClipId
        ? clips.find(c => c.id === slipSlideClip.prevClipId)?.startTime ?? 0
        : 0
      // Clamp: can't go past nextClip's end (or infinity)
      const nextEnd = slipSlideClip.nextClipId
        ? (slipSlideClip.nextOrigStartTime ?? 0) + (slipSlideClip.nextOrigDuration ?? 0)
        : Infinity
      newStartTime = Math.max(minStart, Math.min(nextEnd - clip.duration, newStartTime))
      
      const actualDelta = newStartTime - slipSlideClip.originalStartTime
      
      setClips(prev => prev.map(c => {
        if (c.id === clip.id) {
          return { ...c, startTime: newStartTime }
        }
        // Adjust previous clip: extend its duration
        if (slipSlideClip.prevClipId && c.id === slipSlideClip.prevClipId) {
          const newDur = (slipSlideClip.prevOrigDuration ?? c.duration) + actualDelta
          return { ...c, duration: Math.max(0.5, newDur) }
        }
        // Adjust next clip: shift start and extend duration
        if (slipSlideClip.nextClipId && c.id === slipSlideClip.nextClipId) {
          const newStart = (slipSlideClip.nextOrigStartTime ?? c.startTime) + actualDelta
          const newDur = (slipSlideClip.nextOrigDuration ?? c.duration) - actualDelta
          const newTrimStart = (slipSlideClip.nextOrigTrimStart ?? c.trimStart) + actualDelta * c.speed
          return { ...c, startTime: newStart, duration: Math.max(0.5, newDur), trimStart: Math.max(0, newTrimStart) }
        }
        return c
      }))
    }
  }, [slipSlideClip, clips, pixelsPerSecond])
  
  const handleSlipSlideUp = useCallback(() => {
    setSlipSlideClip(null)
  }, [])
  
  useEffect(() => {
    if (slipSlideClip) {
      window.addEventListener('mousemove', handleSlipSlideMove)
      window.addEventListener('mouseup', handleSlipSlideUp)
      return () => {
        window.removeEventListener('mousemove', handleSlipSlideMove)
        window.removeEventListener('mouseup', handleSlipSlideUp)
      }
    }
  }, [slipSlideClip, handleSlipSlideMove, handleSlipSlideUp])
  
  const handleTrackDrop = (e: React.DragEvent, trackIndex: number) => {
    e.preventDefault()
    
    // Check if it's a timeline being dropped (flatten on drop)
    const timelineData = e.dataTransfer.getData('timeline')
    if (timelineData && trackContainerRef.current) {
      const droppedTimeline = JSON.parse(timelineData) as { id: string; name: string }
      const sourceTimeline = timelines.find(t => t.id === droppedTimeline.id)
      if (!sourceTimeline || sourceTimeline.id === activeTimeline?.id) return
      if (sourceTimeline.clips.length === 0) return
      
      const rect = trackContainerRef.current.getBoundingClientRect()
      const scrollLeft = trackContainerRef.current.scrollLeft
      const x = e.clientX - rect.left + scrollLeft
      const dropTime = Math.max(0, x / pixelsPerSecond)
      
      // Flatten: copy all clips from the source timeline, offset to drop position
      // Find the earliest clip start in the source to compute relative offsets
      const earliestStart = sourceTimeline.clips.reduce(
        (min: number, c: any) => Math.min(min, c.startTime), Infinity
      )
      
      const newClips = sourceTimeline.clips.map((srcClip: any) => migrateClip({
        ...srcClip,
        id: `clip-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
        startTime: dropTime + (srcClip.startTime - earliestStart),
        // Remap trackIndex: offset by the drop track, but keep relative spacing
        trackIndex: Math.min(trackIndex + srcClip.trackIndex, tracks.length - 1),
      }))
      
      setClips(prev => [...prev, ...newClips])
      return
    }
    
    // Multi-asset drop (from multi-select drag) — add sequentially using addClipToTimeline
    const assetIdsJson = e.dataTransfer.getData('assetIds')
    if (assetIdsJson && trackContainerRef.current) {
      try {
        const ids: string[] = JSON.parse(assetIdsJson)
        const droppedAssets = ids.map(id => assets.find(a => a.id === id)).filter(Boolean) as Asset[]
        if (droppedAssets.length > 0) {
          const rect = trackContainerRef.current.getBoundingClientRect()
          const scrollLeft = trackContainerRef.current.scrollLeft
          const x = e.clientX - rect.left + scrollLeft
          let nextStart = Math.max(0, x / pixelsPerSecond)
          for (const a of droppedAssets) {
            addClipToTimeline(a, trackIndex, nextStart)
            nextStart += a.duration || 5
          }
          return
        }
      } catch { /* ignore parse errors */ }
    }
    
    // Single asset drop
    const assetId = e.dataTransfer.getData('assetId')
    const assetData = e.dataTransfer.getData('asset')
    
    let asset: Asset | undefined
    if (assetData) {
      asset = JSON.parse(assetData)
    } else if (assetId) {
      asset = assets.find(a => a.id === assetId)
    }
    
    if (asset && trackContainerRef.current) {
      const rect = trackContainerRef.current.getBoundingClientRect()
      const scrollLeft = trackContainerRef.current.scrollLeft
      const x = e.clientX - rect.left + scrollLeft
      const startTime = Math.max(0, x / pixelsPerSecond)
      addClipToTimeline(asset, trackIndex, startTime)
    }
  }
  

  return {
    draggingClip, setDraggingClip,
    resizingClip, setResizingClip,
    slipSlideClip, setSlipSlideClip,
    lassoRect, setLassoRect,
    isScrubbing,
    scrubFromEvent,
    handleRulerMouseDown,
    expandWithLinkedClips,
    handleClipMouseDown,
    handleMouseMove,
    handleMouseUp,
    handleResizeMove,
    handleResizeStart,
    handleSlipSlideMove,
    handleSlipSlideUp,
    handleTrackDrop,
    lassoOriginRef,
  }
}
