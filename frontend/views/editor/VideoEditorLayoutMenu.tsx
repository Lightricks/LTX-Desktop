import { useEffect, useRef, useState } from 'react'
import { LayoutGrid, RotateCcw, Save, Trash2 } from 'lucide-react'
import { Tooltip } from '../../components/ui/tooltip'
import {
  type EditorLayout,
  type LayoutPreset,
  loadLayoutPresets,
  saveLayoutPresets,
} from './video-editor-utils'

export interface VideoEditorLayoutMenuProps {
  currentLayout: EditorLayout
  onApplyLayout: (layout: EditorLayout) => void
  onResetLayout: () => void
}

export function VideoEditorLayoutMenu(props: VideoEditorLayoutMenuProps) {
  const { currentLayout, onApplyLayout, onResetLayout } = props

  const [showLayoutMenu, setShowLayoutMenu] = useState(false)
  const [layoutPresets, setLayoutPresets] = useState<LayoutPreset[]>(loadLayoutPresets)
  const [savingPresetName, setSavingPresetName] = useState<string | null>(null)
  const presetNameInputRef = useRef<HTMLInputElement>(null)
  const layoutMenuRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!showLayoutMenu) {
      setSavingPresetName(null)
      return
    }
    const handleClick = (e: MouseEvent) => {
      if (layoutMenuRef.current && !layoutMenuRef.current.contains(e.target as Node)) {
        setShowLayoutMenu(false)
      }
    }
    document.addEventListener('mousedown', handleClick)
    return () => document.removeEventListener('mousedown', handleClick)
  }, [showLayoutMenu])

  const handleSaveLayoutPreset = (name: string) => {
    const preset: LayoutPreset = {
      id: `preset-${Date.now()}`,
      name: name.trim() || 'Untitled',
      layout: { ...currentLayout },
    }
    const updated = [...layoutPresets, preset]
    setLayoutPresets(updated)
    saveLayoutPresets(updated)
  }

  const handleDeleteLayoutPreset = (id: string) => {
    const updated = layoutPresets.filter(p => p.id !== id)
    setLayoutPresets(updated)
    saveLayoutPresets(updated)
  }

  const handleApplyLayoutPreset = (preset: LayoutPreset) => {
    onApplyLayout({ ...preset.layout })
    setShowLayoutMenu(false)
  }

  return (
    <div ref={layoutMenuRef} className="relative">
      <button
        onClick={() => setShowLayoutMenu(v => !v)}
        className={`flex items-center gap-1.5 px-2.5 py-1 rounded text-[12px] transition-colors ${
          showLayoutMenu ? 'bg-zinc-800 text-white' : 'text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/50'
        }`}
      >
        <LayoutGrid className="h-3.5 w-3.5" />
        Layout
      </button>
      {showLayoutMenu && (
        <div className="absolute top-full right-0 mt-1 w-56 bg-zinc-900 border border-zinc-700 rounded-lg shadow-xl shadow-black/50 py-1 z-[60]">
          {savingPresetName !== null ? (
            <div className="px-2 py-1.5">
              <div className="text-[11px] text-zinc-400 mb-1.5 px-1">Name this layout:</div>
              <input
                ref={presetNameInputRef}
                autoFocus
                className="w-full bg-zinc-800 border border-zinc-600 rounded px-2 py-1 text-[13px] text-white outline-none focus:border-blue-500"
                value={savingPresetName}
                onChange={e => setSavingPresetName(e.target.value)}
                onKeyDown={e => {
                  if (e.key === 'Enter' && savingPresetName.trim()) {
                    handleSaveLayoutPreset(savingPresetName)
                    setSavingPresetName(null)
                  } else if (e.key === 'Escape') {
                    setSavingPresetName(null)
                  }
                  e.stopPropagation()
                }}
                placeholder="e.g. Wide Timeline"
              />
              <div className="flex gap-1.5 mt-1.5">
                <button
                  onClick={() => {
                    if (savingPresetName.trim()) {
                      handleSaveLayoutPreset(savingPresetName)
                      setSavingPresetName(null)
                    }
                  }}
                  disabled={!savingPresetName.trim()}
                  className="flex-1 px-2 py-1 rounded bg-blue-600 text-white text-[11px] font-medium hover:bg-blue-500 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                >
                  Save
                </button>
                <button
                  onClick={() => setSavingPresetName(null)}
                  className="px-2 py-1 rounded bg-zinc-800 text-zinc-400 text-[11px] hover:bg-zinc-700 transition-colors"
                >
                  Cancel
                </button>
              </div>
            </div>
          ) : (
            <>
              <button
                onClick={() => {
                  setSavingPresetName('')
                  requestAnimationFrame(() => presetNameInputRef.current?.focus())
                }}
                className="w-full flex items-center gap-2.5 px-3 py-1.5 text-[13px] text-zinc-200 hover:bg-blue-600 hover:text-white transition-colors"
              >
                <Save className="h-3.5 w-3.5" />
                Save Current Layout...
              </button>
              <button
                onClick={() => {
                  onResetLayout()
                  setShowLayoutMenu(false)
                }}
                className="w-full flex items-center gap-2.5 px-3 py-1.5 text-[13px] text-zinc-200 hover:bg-blue-600 hover:text-white transition-colors"
              >
                <RotateCcw className="h-3.5 w-3.5" />
                Reset to Default
              </button>
              {layoutPresets.length > 0 && (
                <>
                  <div className="h-px bg-zinc-700 my-1 mx-2" />
                  <div className="px-3 py-1 text-[10px] text-zinc-500 uppercase tracking-wider">Saved Layouts</div>
                  {layoutPresets.map(preset => (
                    <div
                      key={preset.id}
                      className="flex items-center group hover:bg-blue-600 transition-colors"
                    >
                      <button
                        onClick={() => handleApplyLayoutPreset(preset)}
                        className="flex-1 flex items-center gap-2.5 px-3 py-1.5 text-[13px] text-zinc-200 group-hover:text-white transition-colors text-left"
                      >
                        <LayoutGrid className="h-3.5 w-3.5 text-zinc-500 group-hover:text-white" />
                        {preset.name}
                      </button>
                      <Tooltip content="Delete preset" side="top">
                        <button
                          onClick={(e) => {
                            e.stopPropagation()
                            handleDeleteLayoutPreset(preset.id)
                          }}
                          className="px-2 py-1.5 text-zinc-600 hover:text-red-400 opacity-0 group-hover:opacity-100 transition-all"
                        >
                          <Trash2 className="h-3 w-3" />
                        </button>
                      </Tooltip>
                    </div>
                  ))}
                </>
              )}
            </>
          )}
        </div>
      )}
    </div>
  )
}
