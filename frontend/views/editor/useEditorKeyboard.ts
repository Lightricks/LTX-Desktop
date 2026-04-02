import { useEffect, useRef } from 'react'
import { resolveAction, type ActionId } from '../../lib/keyboard-shortcuts'
import type { EditorState } from './editor-state'
import type { SourceKeyboardAction } from './VideoEditorSourceMonitor'
import {
  selectActiveFocusArea,
  selectActiveTimelineInPoint,
  selectActiveTimelineOutPoint,
  selectClips,
  selectKeyboardCommandContext,
  selectSelectedGap,
  selectSelectedSubtitleId,
  selectShuttleSpeed,
  selectTracks,
} from './editor-selectors'
import { useEditorActions } from './editor-store'

// Frame duration at 24fps
const FRAME_DURATION = 1 / 24

const FORWARD_SPEEDS = [1, 2, 4, 8]
const REVERSE_SPEEDS = [-1, -2, -4, -8]

interface KeyboardRefs {
  kbLayoutRef: React.MutableRefObject<any>
  isKbEditorOpenRef: React.MutableRefObject<boolean>
  getState: () => EditorState
  playbackTimeRef: React.MutableRefObject<number>
  sourceDispatchRef: React.MutableRefObject<(action: SourceKeyboardAction) => void>
  sourcePauseRef: React.MutableRefObject<() => void>
  centerOnPlayheadRef: React.MutableRefObject<boolean>
  getMinZoomRef: React.MutableRefObject<() => number>
  gapGenerateModeRef: React.MutableRefObject<'text-to-video' | 'image-to-video' | 'text-to-image' | null>
  clearSelectedGapRef: React.MutableRefObject<() => void>
  closeSelectedGapRef: React.MutableRefObject<() => void>
  fitToViewRef: React.MutableRefObject<() => void>
  toggleFullscreenRef: React.MutableRefObject<() => void>
  insertEditRef: React.MutableRefObject<() => void>
  overwriteEditRef: React.MutableRefObject<() => void>
  matchFrameRef: React.MutableRefObject<() => void>
}

interface KeyboardContext {
  deleteAssetActionRef: React.MutableRefObject<() => void>
}

export interface UseEditorKeyboardParams {
  refs: KeyboardRefs
  context: KeyboardContext
}

export function useEditorKeyboard(params: UseEditorKeyboardParams) {
  const { refs, context } = params
  const actions = useEditorActions()
  const kHeldRef = useRef(false)
  const contextRef = useRef(context)
  const actionsRef = useRef(actions)
  contextRef.current = context
  actionsRef.current = actions

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return
      if (refs.isKbEditorOpenRef.current) return

      const context = contextRef.current
      const state = refs.getState()
      const commandContext = selectKeyboardCommandContext(state)
      const sel = commandContext.selectedClipIds
      const td = commandContext.totalDuration
      const focusArea = selectActiveFocusArea(state)

      const action: ActionId | null = resolveAction(refs.kbLayoutRef.current, e)
      if (!action) return

      e.preventDefault()
      const editorActions = actionsRef.current

      switch (action) {
        // Tools
        case 'tool.select':       editorActions.setActiveTool('select'); break
        case 'tool.blade':        editorActions.setActiveTool('blade'); break
        case 'tool.ripple':       editorActions.setActiveTool('ripple'); editorActions.setLastTrimTool('ripple'); break
        case 'tool.roll':         editorActions.setActiveTool('roll'); editorActions.setLastTrimTool('roll'); break
        case 'tool.slide':        editorActions.setActiveTool('slide'); editorActions.setLastTrimTool('slide'); break
        case 'tool.slip':         editorActions.setActiveTool('slip'); editorActions.setLastTrimTool('slip'); break
        case 'tool.trackForward': editorActions.setActiveTool('trackForward'); break

        // Transport — panel-aware
        case 'transport.playPause':
          if (focusArea === 'source') {
            editorActions.pause()
            editorActions.stopShuttle()
            refs.sourceDispatchRef.current('transport.playPause')
          } else {
            refs.sourcePauseRef.current()
            const wasPlaying = state.session.transport.isPlaying
            const hadShuttle = state.session.transport.shuttleSpeed !== 0
            editorActions.stopShuttle()
            if (wasPlaying || hadShuttle) editorActions.pause()
            else editorActions.play()
          }
          break

        case 'transport.shuttleReverse':
          if (focusArea === 'source') {
            refs.sourceDispatchRef.current('transport.shuttleReverse')
          } else {
            refs.sourcePauseRef.current()
            if (kHeldRef.current) {
              editorActions.pause()
              editorActions.stopShuttle()
              editorActions.stepCurrentTime(-FRAME_DURATION)
            } else {
              const currentSpeed = selectShuttleSpeed(state)
              const nextSpeed = currentSpeed > 0
                ? -1
                : REVERSE_SPEEDS[Math.max(0, Math.min(REVERSE_SPEEDS.length - 1, REVERSE_SPEEDS.indexOf(currentSpeed) + 1 || 0))]
              editorActions.setShuttleSpeed(nextSpeed)
              editorActions.play()
            }
          }
          break

        case 'transport.shuttleStop':
          if (focusArea === 'source') {
            refs.sourceDispatchRef.current('transport.shuttleStop')
          } else {
            kHeldRef.current = true
            editorActions.pause()
            editorActions.stopShuttle()
          }
          break

        case 'transport.shuttleForward':
          if (focusArea === 'source') {
            refs.sourceDispatchRef.current(kHeldRef.current ? 'transport.stepForward' : 'transport.shuttleForward')
          } else {
            refs.sourcePauseRef.current()
            if (kHeldRef.current) {
              editorActions.pause()
              editorActions.stopShuttle()
              editorActions.stepCurrentTime(FRAME_DURATION)
            } else {
              const currentSpeed = selectShuttleSpeed(state)
              const nextSpeed = currentSpeed < 0
                ? 1
                : FORWARD_SPEEDS[Math.max(0, Math.min(FORWARD_SPEEDS.length - 1, FORWARD_SPEEDS.indexOf(currentSpeed) + 1 || 0))]
              editorActions.setShuttleSpeed(nextSpeed)
              editorActions.play()
            }
          }
          break

        case 'transport.stepBackward':
          if (focusArea === 'source') {
            refs.sourceDispatchRef.current('transport.stepBackward')
          } else {
            editorActions.stepCurrentTime(-FRAME_DURATION)
          }
          break

        case 'transport.stepForward':
          if (focusArea === 'source') {
            refs.sourceDispatchRef.current('transport.stepForward')
          } else {
            const nextTime = Math.min(commandContext.totalDuration, commandContext.currentTime + FRAME_DURATION)
            editorActions.setCurrentTime(nextTime)
          }
          break

        case 'transport.jumpBackward':
          if (focusArea === 'source') {
            refs.sourceDispatchRef.current('transport.jumpBackward')
          } else {
            editorActions.stepCurrentTime(-1)
          }
          break

        case 'transport.jumpForward':
          if (focusArea === 'source') {
            refs.sourceDispatchRef.current('transport.jumpForward')
          } else {
            const nextTime = Math.min(commandContext.totalDuration, commandContext.currentTime + 1)
            editorActions.setCurrentTime(nextTime)
          }
          break

        case 'transport.goToStart':
          editorActions.pause()
          editorActions.stopShuttle()
          editorActions.setCurrentTime(0)
          break
        case 'transport.goToEnd':
          editorActions.pause()
          editorActions.stopShuttle()
          editorActions.setCurrentTime(td)
          break
        case 'transport.goToIn': {
          const { inPoint, clips } = commandContext
          const target = inPoint ?? (clips.length > 0 ? Math.min(...clips.map(c => c.startTime)) : 0)
          editorActions.pause()
          editorActions.stopShuttle()
          editorActions.setCurrentTime(target)
          break
        }
        case 'transport.goToOut': {
          const { outPoint, clips: clipsOut, totalDuration: tdOut } = commandContext
          const target = outPoint ?? (clipsOut.length > 0 ? Math.max(...clipsOut.map(c => c.startTime + c.duration)) : tdOut)
          editorActions.pause()
          editorActions.stopShuttle()
          editorActions.setCurrentTime(target)
          break
        }

        // Editing
        case 'edit.undo':    editorActions.undo(); break
        case 'edit.redo':    editorActions.redo(); break
        case 'edit.cut':     editorActions.cutSelection(); break
        case 'edit.copy':    editorActions.copySelection(); break
        case 'edit.paste':   editorActions.pasteSelection(); break
        case 'edit.selectAll':
          editorActions.selectAllClips()
          break
        case 'edit.deselect':
          if (refs.gapGenerateModeRef.current) {
            refs.clearSelectedGapRef.current()
          } else {
            editorActions.clearClipSelection()
          }
          break
        case 'edit.delete':
          if (sel.size > 0) {
            const deleteIds = new Set<string>()
            for (const id of sel) {
              const clip = selectClips(state).find(cl => cl.id === id)
              if (clip && selectTracks(state)[clip.trackIndex]?.locked) continue
              deleteIds.add(id)
              if (clip?.linkedClipIds) {
                const allLinkedSelected = clip.linkedClipIds.every(lid => sel.has(lid))
                if (allLinkedSelected) clip.linkedClipIds.forEach(lid => deleteIds.add(lid))
              }
            }
            editorActions.deleteClips([...deleteIds])
          } else if (selectSelectedGap(state)) {
            refs.closeSelectedGapRef.current()
          } else if (selectSelectedSubtitleId(state)) {
            editorActions.deleteSubtitle(selectSelectedSubtitleId(state)!)
          } else {
            context.deleteAssetActionRef.current()
          }
          break
        case 'edit.insertEdit':    refs.insertEditRef.current(); break
        case 'edit.overwriteEdit': refs.overwriteEditRef.current(); break
        case 'edit.matchFrame':    refs.matchFrameRef.current(); break

        // Marking — panel-aware
        case 'mark.setIn':
          if (focusArea === 'source') {
            refs.sourceDispatchRef.current('mark.setIn')
          } else {
            const ct = commandContext.currentTime
            const currentIn = selectActiveTimelineInPoint(state)
            editorActions.setTimelineInPoint(currentIn !== null && Math.abs(currentIn - ct) < 0.01 ? null : ct)
          }
          break
        case 'mark.setOut':
          if (focusArea === 'source') {
            refs.sourceDispatchRef.current('mark.setOut')
          } else {
            const ct = commandContext.currentTime
            const currentOut = selectActiveTimelineOutPoint(state)
            editorActions.setTimelineOutPoint(currentOut !== null && Math.abs(currentOut - ct) < 0.01 ? null : ct)
          }
          break
        case 'mark.clearIn':
          if (focusArea === 'source') {
            refs.sourceDispatchRef.current('mark.clearIn')
          } else {
            editorActions.clearTimelineInPoint()
          }
          break
        case 'mark.clearOut':
          if (focusArea === 'source') {
            refs.sourceDispatchRef.current('mark.clearOut')
          } else {
            editorActions.clearTimelineOutPoint()
          }
          break
        case 'mark.clearInOut':
          if (focusArea === 'source') {
            refs.sourceDispatchRef.current('mark.clearInOut')
          } else {
            editorActions.clearTimelineMarks()
          }
          break

        // Timeline
        case 'timeline.zoomIn':
          refs.centerOnPlayheadRef.current = true
          editorActions.setZoom(Math.min(4, +(state.session.tools.zoom + 0.25).toFixed(2)))
          break
        case 'timeline.zoomOut':
          refs.centerOnPlayheadRef.current = true
          editorActions.setZoom(Math.max(refs.getMinZoomRef.current(), +(state.session.tools.zoom - 0.25).toFixed(2)))
          break
        case 'timeline.fitToView':
          refs.fitToViewRef.current()
          break
        case 'timeline.toggleSnap':
          editorActions.toggleSnap()
          break
        case 'nav.prevEdit':
          editorActions.pause()
          editorActions.goToPrevEdit(state.session.transport.isPlaying ? refs.playbackTimeRef.current : state.session.transport.currentTime)
          break
        case 'nav.nextEdit':
          editorActions.pause()
          editorActions.goToNextEdit(state.session.transport.isPlaying ? refs.playbackTimeRef.current : state.session.transport.currentTime)
          break
        case 'view.fullscreen':
          refs.toggleFullscreenRef.current()
          break
      }
    }

    const handleKeyUp = (e: KeyboardEvent) => {
      if (e.key.toLowerCase() === 'k') {
        kHeldRef.current = false
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    window.addEventListener('keyup', handleKeyUp)
    return () => {
      window.removeEventListener('keydown', handleKeyDown)
      window.removeEventListener('keyup', handleKeyUp)
    }
  }, []) // stable - uses refs for latest state
}
