import React, { useEffect, useMemo, useRef, useState } from 'react'
import { Plus, FileUp, Film, Trash2 } from 'lucide-react'
import { Tooltip } from '../../components/ui/tooltip'
import { shallow } from 'zustand/vanilla/shallow'
import {
  selectActiveTimelineId,
  selectOpenTimelineIds,
  selectTimelineRenameState,
  selectTimelines,
} from './editor-selectors'
import { useEditorActions, useEditorStore } from './editor-store'

export interface VideoEditorTimelineControlPanelProps {
  handleTimelineTabContextMenu: (e: React.MouseEvent, timelineId: string) => void
}

export function VideoEditorTimelineControlPanel(props: VideoEditorTimelineControlPanelProps) {
  const {
    handleTimelineTabContextMenu,
  } = props
  const {
    cancelTimelineRename,
    commitTimelineRename,
    createTimeline,
    deleteTimeline,
    openImportTimelineModal,
    setTimelineRenameValue,
    startTimelineRename,
    switchActiveTimeline,
  } = useEditorActions()
  const timelines = useEditorStore(selectTimelines)
  const activeTimelineId = useEditorStore(selectActiveTimelineId)
  const openTimelineIds = useEditorStore(selectOpenTimelineIds)
  const { renamingTimelineId, renameValue, renameSource } = useEditorStore(selectTimelineRenameState, shallow)
  const timelineItems = useMemo(() => (
    timelines.map(timeline => ({
      timeline,
      isActive: timeline.id === activeTimelineId,
      isOpen: openTimelineIds.has(timeline.id),
      isRenaming: renamingTimelineId === timeline.id,
      clipCount: timeline.clips.length,
      duration: timeline.clips.reduce((max, clip) => Math.max(max, clip.startTime + clip.duration), 0),
    }))
  ), [activeTimelineId, openTimelineIds, renamingTimelineId, timelines])

  const [timelineAddMenuOpen, setTimelineAddMenuOpen] = useState(false)
  const addMenuRef = useRef<HTMLDivElement>(null)

  const setRenameValue = (value: string) => {
    setTimelineRenameValue(value)
  }

  const setRenamingTimelineId = (value: string | null) => {
    if (value === null) cancelTimelineRename()
  }

  const handleAddTimeline = () => {
    createTimeline()
  }

  const handleSwitchTimeline = (timelineId: string) => {
    switchActiveTimeline(timelineId)
  }

  const handleDeleteTimeline = (timelineId: string) => {
    deleteTimeline(timelineId)
  }

  const handleStartRename = (timelineId: string, currentName: string, source: 'tab' | 'panel' = 'panel') => {
    void currentName
    startTimelineRename(timelineId, source)
  }

  const handleFinishRename = () => {
    commitTimelineRename()
  }

  useEffect(() => {
    if (!timelineAddMenuOpen) return
    const onClickAway = (e: MouseEvent) => {
      if (!addMenuRef.current?.contains(e.target as Node)) {
        setTimelineAddMenuOpen(false)
      }
    }
    const timer = setTimeout(() => window.addEventListener('click', onClickAway), 0)
    return () => {
      clearTimeout(timer)
      window.removeEventListener('click', onClickAway)
    }
  }, [timelineAddMenuOpen])

  return (
    <div className="flex flex-col min-h-0 h-full">
      <div className="p-3 pb-2 flex items-center justify-between flex-shrink-0">
        <h3 className="text-sm font-semibold text-fg-primary">Timelines</h3>
        <div className="relative" ref={addMenuRef}>
          <Tooltip content="Add timeline" side="right">
            <button
              onClick={() => setTimelineAddMenuOpen(prev => !prev)}
              className="p-1.5 rounded-lg hover:bg-action text-fg-secondary hover:text-fg-primary transition-colors"
            >
              <Plus className="h-4 w-4" />
            </button>
          </Tooltip>
          {timelineAddMenuOpen && (
            <div className="absolute right-0 top-full mt-1 w-48 bg-action border border-separator rounded-lg shadow-xl z-50 py-1 overflow-hidden">
              <button
                onClick={() => { handleAddTimeline(); setTimelineAddMenuOpen(false) }}
                className="w-full text-left px-3 py-2 text-xs text-fg-secondary hover:bg-action-hover flex items-center gap-2"
              >
                <Plus className="h-3.5 w-3.5" />
                New Timeline
              </button>
              <button
                onClick={() => {
                  openImportTimelineModal()
                  setTimelineAddMenuOpen(false)
                }}
                className="w-full text-left px-3 py-2 text-xs text-fg-secondary hover:bg-action-hover flex items-center gap-2"
              >
                <FileUp className="h-3.5 w-3.5" />
                Import from XML
              </button>
            </div>
          )}
        </div>
      </div>

      <div className="flex-1 overflow-auto px-3 pb-3 space-y-1">
        {timelineItems.map(item => {
          const tl = item.timeline
          const isActive = item.isActive
          const clipCount = item.clipCount
          const tlDuration = item.duration
          const formatDur = (s: number) => {
            const m = Math.floor(s / 60)
            const sec = Math.floor(s % 60)
            return m > 0 ? `${m}m ${sec}s` : `${sec}s`
          }

          return (
            <div
              key={tl.id}
              className={`group flex items-center gap-2 px-3 py-2 rounded-lg cursor-pointer transition-colors ${
                isActive
                  ? 'bg-blue-600/20 border border-blue-500/40'
                  : 'hover:bg-action border border-transparent'
              }`}
              draggable={!isActive}
              onDragStart={(e) => {
                if (isActive) { e.preventDefault(); return }
                e.dataTransfer.setData('timeline', JSON.stringify({ id: tl.id, name: tl.name }))
                e.dataTransfer.effectAllowed = 'copy'
              }}
              onClick={() => handleSwitchTimeline(tl.id)}
              onDoubleClick={() => handleStartRename(tl.id, tl.name, 'panel')}
              onContextMenu={(e) => handleTimelineTabContextMenu(e, tl.id)}
            >
              <Film className={`h-4 w-4 flex-shrink-0 ${isActive ? 'text-blue-400' : 'text-fg-tertiary'}`} />
              <div className="flex-1 min-w-0">
                {item.isRenaming && renameSource === 'panel' ? (
                  <input
                    type="text"
                    value={renameValue}
                    onChange={(e) => setRenameValue(e.target.value)}
                    onBlur={handleFinishRename}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') handleFinishRename()
                      if (e.key === 'Escape') { setRenamingTimelineId(null); setRenameValue('') }
                    }}
                    className="bg-surface-primary border border-blue-500 rounded px-1 py-0.5 outline-none text-fg-primary text-xs w-full"
                    autoFocus
                    onClick={(e) => e.stopPropagation()}
                    onDoubleClick={(e) => e.stopPropagation()}
                  />
                ) : (
                  <p className={`text-xs font-medium truncate ${isActive ? 'text-fg-primary' : 'text-fg-secondary'}`}>
                    {tl.name}
                  </p>
                )}
                <div className="flex items-center gap-2 text-[10px] text-fg-tertiary">
                  <span>{clipCount} clip{clipCount !== 1 ? 's' : ''}</span>
                  {clipCount > 0 && (
                    <>
                      <span>·</span>
                      <span>{formatDur(tlDuration)}</span>
                    </>
                  )}
                </div>
              </div>
              {isActive ? (
                <span className="text-[9px] text-blue-400 font-medium uppercase tracking-wider flex-shrink-0">Active</span>
              ) : item.isOpen ? (
                <span className="w-1.5 h-1.5 rounded-full bg-action-active flex-shrink-0" title="Open in tabs" />
              ) : null}
              {timelineItems.length > 1 && (
                <Tooltip content="Delete timeline" side="right">
                  <button
                    onClick={(e) => {
                      e.stopPropagation()
                      handleDeleteTimeline(tl.id)
                    }}
                    className="p-1 rounded hover:bg-danger-soft text-fg-tertiary hover:text-fg-danger opacity-0 group-hover:opacity-100 transition-all flex-shrink-0"
                  >
                    <Trash2 className="h-3 w-3" />
                  </button>
                </Tooltip>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}
