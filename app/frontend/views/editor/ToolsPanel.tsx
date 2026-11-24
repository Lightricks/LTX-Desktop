import React from 'react'
import { Magnet, Type } from 'lucide-react'
import { PRIMARY_TOOLS, TRIM_TOOLS, ToolType } from './video-editor-utils'
import { getShortcutLabel, type KeyboardLayout } from './video-editor-utils'
import { Tooltip } from '@/components/ui/tooltip'

interface ToolsPanelProps {
  activeTool: ToolType
  setActiveTool: (t: ToolType) => void
  lastTrimTool: ToolType
  setLastTrimTool: (t: ToolType) => void
  showTrimFlyout: boolean
  setShowTrimFlyout: (v: boolean) => void
  trimFlyoutOpenedRef: React.MutableRefObject<boolean>
  trimLongPressRef: React.MutableRefObject<ReturnType<typeof setTimeout> | null>
  snapEnabled: boolean
  setSnapEnabled: (v: boolean) => void
  showEffectsBrowser: boolean
  setShowEffectsBrowser: (v: boolean) => void
  addTextClip: () => void
  kbLayout: KeyboardLayout
}

export function ToolsPanel({
  activeTool, setActiveTool,
  lastTrimTool, setLastTrimTool,
  showTrimFlyout, setShowTrimFlyout,
  trimFlyoutOpenedRef, trimLongPressRef,
  snapEnabled, setSnapEnabled,
  showEffectsBrowser: _showEffectsBrowser, setShowEffectsBrowser: _setShowEffectsBrowser, // EFFECTS HIDDEN
  addTextClip, kbLayout,
}: ToolsPanelProps) {
  return (
    <div className="w-10 flex-shrink-0 bg-surface-primary border-r border-separator flex flex-col items-center py-1 gap-0.5 overflow-y-auto">
      {PRIMARY_TOOLS.map(tool => (
        <Tooltip
          key={tool.id}
          side="right"
          content={(() => { const s = getShortcutLabel(kbLayout, tool.actionId); return <>{tool.label}{s && <span className="text-fg-secondary"> ({s})</span>}</>; })()}
        >
          <button
            onClick={() => setActiveTool(tool.id)}
            className={`p-1.5 rounded-lg transition-colors flex-shrink-0 ${
              activeTool === tool.id
                ? 'bg-brand text-fg-white'
                : 'text-fg-secondary hover:bg-action hover:text-fg-primary'
            }`}
          >
            <tool.icon className="h-4 w-4" />
          </button>
        </Tooltip>
      ))}

      {/* Trim tools group button */}
      {(() => {
        const trimToolIds = new Set(TRIM_TOOLS.map(t => t.id))
        const isTrimActive = trimToolIds.has(activeTool)
        const currentTrimTool = TRIM_TOOLS.find(t => t.id === (isTrimActive ? activeTool : lastTrimTool)) || TRIM_TOOLS[0]
        return (
          <div className="relative flex-shrink-0">
            <Tooltip
              side="right"
              content={(() => { const s = getShortcutLabel(kbLayout, currentTrimTool.actionId); return <>{currentTrimTool.label}<span className="text-fg-secondary">{s ? ` (${s}) — ` : ' — '}right-click or hold for more</span></>; })()}
            >
              <button
                onClick={() => {
                  if (trimFlyoutOpenedRef.current) { trimFlyoutOpenedRef.current = false; return }
                  setActiveTool(currentTrimTool.id)
                  setLastTrimTool(currentTrimTool.id)
                }}
                onContextMenu={(e) => {
                  e.preventDefault()
                  e.stopPropagation()
                  if (trimLongPressRef.current) { clearTimeout(trimLongPressRef.current); trimLongPressRef.current = null }
                  trimFlyoutOpenedRef.current = true
                  setShowTrimFlyout(true)
                }}
                onMouseDown={(e) => {
                  if (e.button !== 0) return
                  trimFlyoutOpenedRef.current = false
                  trimLongPressRef.current = setTimeout(() => {
                    trimLongPressRef.current = null
                    trimFlyoutOpenedRef.current = true
                    setShowTrimFlyout(true)
                  }, 400)
                }}
                onMouseUp={() => {
                  if (trimLongPressRef.current) { clearTimeout(trimLongPressRef.current); trimLongPressRef.current = null }
                }}
                onMouseLeave={() => {
                  if (trimLongPressRef.current) { clearTimeout(trimLongPressRef.current); trimLongPressRef.current = null }
                }}
                data-trim-group-btn=""
                className={`p-1.5 rounded-lg transition-colors relative ${
                  isTrimActive
                    ? 'bg-brand text-fg-white'
                    : 'text-fg-secondary hover:bg-action hover:text-fg-primary'
                }`}
              >
                <currentTrimTool.icon className="h-4 w-4" />
                <div className="absolute bottom-0 right-0 w-0 h-0 border-l-[4px] border-l-transparent border-b-[4px] border-b-current opacity-60" />
              </button>
            </Tooltip>
            {showTrimFlyout && (() => {
              const btnEl = document.querySelector('[data-trim-group-btn]')
              const rect = btnEl?.getBoundingClientRect()
              return (
                <>
                  <div className="fixed inset-0 z-[9998]" onMouseDown={() => setShowTrimFlyout(false)} onContextMenu={(e) => { e.preventDefault(); setShowTrimFlyout(false) }} />
                  <div
                    className="fixed bg-action border border-separator rounded-lg shadow-xl py-1 z-[9999] min-w-[160px]"
                    style={{ top: rect?.top ?? 0, left: (rect?.right ?? 44) + 4 }}
                  >
                    {TRIM_TOOLS.map(t => (
                      <button
                        key={t.id}
                        onClick={() => {
                          setActiveTool(t.id)
                          setLastTrimTool(t.id)
                          setShowTrimFlyout(false)
                        }}
                        className={`w-full text-left px-3 py-1.5 text-xs flex items-center gap-2 transition-colors ${
                          activeTool === t.id ? 'bg-blue-600/30 text-fg-primary' : 'text-fg-secondary hover:bg-action-hover'
                        }`}
                      >
                        <t.icon className="h-3.5 w-3.5" />
                        <span className="flex-1">{t.label}</span>
                        <span className="text-fg-tertiary text-[10px]">{getShortcutLabel(kbLayout, t.actionId)}</span>
                      </button>
                    ))}
                  </div>
                </>
              )
            })()}
          </div>
        )
      })()}

      <div className="w-6 h-px bg-action-hover my-1 flex-shrink-0" />

      <Tooltip side="right" content={snapEnabled ? 'Snapping On' : 'Snapping Off'}>
        <button
          onClick={() => setSnapEnabled(!snapEnabled)}
          className={`p-1.5 rounded-lg transition-colors flex-shrink-0 ${
            snapEnabled
              ? 'bg-brand text-fg-white'
              : 'text-fg-secondary hover:bg-action hover:text-fg-primary'
          }`}
        >
          <Magnet className="h-4 w-4" />
        </button>
      </Tooltip>

      {/* EFFECTS HIDDEN - FX button hidden because effects are not applied during export
      <div className="w-6 h-px bg-action-hover my-1 flex-shrink-0" />

      <Tooltip side="right" content="Effects Browser">
        <button
          onClick={() => setShowEffectsBrowser(!showEffectsBrowser)}
          className={`p-1.5 rounded-lg transition-colors flex-shrink-0 text-[10px] font-bold ${
            showEffectsBrowser
              ? 'bg-brand text-fg-white'
              : 'text-fg-secondary hover:bg-action hover:text-fg-primary'
          }`}
        >
          FX
        </button>
      </Tooltip>
      EFFECTS HIDDEN */}

      <div className="w-6 h-px bg-action-hover my-1 flex-shrink-0" />

      <Tooltip side="right" content="Add Text Overlay">
        <button
          onClick={() => addTextClip()}
          className="p-1.5 rounded-lg transition-colors flex-shrink-0 text-ext-teal hover:bg-ext-teal/15 hover:-ext-teal"
        >
          <Type className="h-4 w-4" />
        </button>
      </Tooltip>
    </div>
  )
}
