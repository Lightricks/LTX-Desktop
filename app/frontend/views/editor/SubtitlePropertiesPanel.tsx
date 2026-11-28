import { useMemo } from 'react'
import { MessageSquare, Trash2 } from 'lucide-react'
import type { SubtitleStyle } from '../../types/project-model'
import { DEFAULT_SUBTITLE_STYLE } from '../../types/project-model'
import {
  selectSelectedSubtitleId,
  selectSubtitles,
  selectTracks,
} from './editor-selectors'
import { useEditorActions, useEditorStore } from './editor-store'

export function SubtitlePropertiesPanel() {
  const { updateSubtitle, deleteSubtitle } = useEditorActions()
  const selectedSubtitleId = useEditorStore(selectSelectedSubtitleId)
  const subtitles = useEditorStore(selectSubtitles)
  const tracks = useEditorStore(selectTracks)
  const selectedSub = useMemo(
    () => subtitles.find(subtitle => subtitle.id === selectedSubtitleId) ?? null,
    [selectedSubtitleId, subtitles],
  )
  if (!selectedSub) return null
  const trackStyle = useMemo(() => {
    const track = tracks[selectedSub.trackIndex]
    return {
      ...(track?.subtitleStyle || {}),
      ...(selectedSub.style || {}),
    }
  }, [selectedSub, tracks])
  const subStyle = { ...DEFAULT_SUBTITLE_STYLE, ...trackStyle, ...selectedSub.style }

  return (
    <div className="h-full border-l border-separator bg-surface-primary p-4 overflow-auto">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-sm font-semibold text-fg-warning flex items-center gap-2">
            <MessageSquare className="h-4 w-4" />
            Subtitle
          </h3>
          <button
            onClick={() => deleteSubtitle(selectedSub.id)}
            className="p-1 rounded hover:bg-danger-soft text-fg-tertiary hover:text-fg-danger"
            title="Delete subtitle"
          >
            <Trash2 className="h-3.5 w-3.5" />
          </button>
        </div>

        <div className="space-y-4">
          {/* Subtitle text */}
          <div>
            <label className="text-[10px] text-fg-tertiary uppercase tracking-wider font-semibold mb-1.5 block">Text</label>
            <textarea
              value={selectedSub.text}
              onChange={(e) => updateSubtitle(selectedSub.id, { text: e.target.value })}
              onKeyDown={(e) => e.stopPropagation()}
              className="w-full bg-action border border-separator rounded-lg p-2.5 text-sm text-fg-primary resize-none focus:outline-none focus:border-warning/50 focus:ring-1 focus:ring-warning/30"
              rows={3}
              placeholder="Enter subtitle text..."
            />
          </div>

          {/* Timing */}
          <div>
            <label className="text-[10px] text-fg-tertiary uppercase tracking-wider font-semibold mb-1.5 block">Timing</label>
            <div className="grid grid-cols-2 gap-2">
              <div>
                <span className="text-[9px] text-fg-tertiary block mb-1">Start</span>
                <input
                  type="number"
                  step={0.1}
                  min={0}
                  value={parseFloat(selectedSub.startTime.toFixed(2))}
                  onChange={(e) => {
                    const v = parseFloat(e.target.value)
                    if (!isNaN(v) && v >= 0 && v < selectedSub.endTime) {
                      updateSubtitle(selectedSub.id, { startTime: v })
                    }
                  }}
                  onKeyDown={(e) => e.stopPropagation()}
                  className="w-full bg-action border border-separator rounded px-2 py-1 text-xs text-fg-primary font-mono focus:outline-none focus:border-warning/50"
                />
              </div>
              <div>
                <span className="text-[9px] text-fg-tertiary block mb-1">End</span>
                <input
                  type="number"
                  step={0.1}
                  min={0}
                  value={parseFloat(selectedSub.endTime.toFixed(2))}
                  onChange={(e) => {
                    const v = parseFloat(e.target.value)
                    if (!isNaN(v) && v > selectedSub.startTime) {
                      updateSubtitle(selectedSub.id, { endTime: v })
                    }
                  }}
                  onKeyDown={(e) => e.stopPropagation()}
                  className="w-full bg-action border border-separator rounded px-2 py-1 text-xs text-fg-primary font-mono focus:outline-none focus:border-warning/50"
                />
              </div>
            </div>
            <span className="text-[9px] text-fg-tertiary mt-1 block">
              Duration: {(selectedSub.endTime - selectedSub.startTime).toFixed(2)}s
            </span>
          </div>

          {/* Style */}
          <div>
            <label className="text-[10px] text-fg-tertiary uppercase tracking-wider font-semibold mb-1.5 block">Style</label>
            <div className="space-y-2">
              {/* Font size */}
              <div className="flex items-center justify-between">
                <span className="text-[10px] text-fg-secondary">Font Size</span>
                <input
                  type="number"
                  min={12}
                  max={96}
                  value={subStyle.fontSize}
                  onChange={(e) => updateSubtitle(selectedSub.id, { style: { ...selectedSub.style, fontSize: parseInt(e.target.value) || 32 } })}
                  onKeyDown={(e) => e.stopPropagation()}
                  className="w-16 bg-action border border-separator rounded px-2 py-0.5 text-[10px] text-fg-primary text-center focus:outline-none focus:border-warning/50"
                />
              </div>

              {/* Bold / Italic */}
              <div className="flex items-center gap-2">
                <button
                  onClick={() => updateSubtitle(selectedSub.id, { style: { ...selectedSub.style, fontWeight: subStyle.fontWeight === 'bold' ? 'normal' : 'bold' } })}
                  className={`px-2.5 py-1 rounded text-[10px] font-bold ${subStyle.fontWeight === 'bold' ? 'bg-warning-soft text-fg-warning border border-warning/40' : 'bg-action text-fg-secondary border border-separator'}`}
                >
                  B
                </button>
                <button
                  onClick={() => updateSubtitle(selectedSub.id, { style: { ...selectedSub.style, italic: !subStyle.italic } })}
                  className={`px-2.5 py-1 rounded text-[10px] italic ${subStyle.italic ? 'bg-warning-soft text-fg-warning border border-warning/40' : 'bg-action text-fg-secondary border border-separator'}`}
                >
                  I
                </button>
              </div>

              {/* Text color */}
              <div className="flex items-center justify-between">
                <span className="text-[10px] text-fg-secondary">Text Color</span>
                <input
                  type="color"
                  value={subStyle.color}
                  onChange={(e) => updateSubtitle(selectedSub.id, { style: { ...selectedSub.style, color: e.target.value } })}
                  className="w-7 h-6 rounded cursor-pointer border border-separator"
                />
              </div>

              {/* Background toggle + color */}
              <div className="flex items-center justify-between">
                <span className="text-[10px] text-fg-secondary">Background</span>
                <div className="flex items-center gap-1.5">
                  <button
                    onClick={() => updateSubtitle(selectedSub.id, {
                      style: { ...selectedSub.style, backgroundColor: subStyle.backgroundColor === 'transparent' ? '#000000AA' : 'transparent' }
                    })}
                    className={`px-2 py-0.5 rounded text-[9px] border ${
                      subStyle.backgroundColor !== 'transparent'
                        ? 'bg-warning-soft text-fg-warning border-warning/40'
                        : 'bg-action text-fg-tertiary border-separator'
                    }`}
                  >
                    {subStyle.backgroundColor !== 'transparent' ? 'On' : 'Off'}
                  </button>
                  {subStyle.backgroundColor !== 'transparent' && (
                    <input
                      type="color"
                      value={subStyle.backgroundColor.slice(0, 7)}
                      onChange={(e) => updateSubtitle(selectedSub.id, { style: { ...selectedSub.style, backgroundColor: e.target.value + 'CC' } })}
                      className="w-7 h-6 rounded cursor-pointer border border-separator"
                    />
                  )}
                </div>
              </div>

              {/* Position */}
              <div className="flex items-center justify-between">
                <span className="text-[10px] text-fg-secondary">Position</span>
                <select
                  value={subStyle.position}
                  onChange={(e) => updateSubtitle(selectedSub.id, { style: { ...selectedSub.style, position: e.target.value as SubtitleStyle['position'] } })}
                  className="bg-action border border-separator rounded px-2 py-0.5 text-[10px] text-fg-primary focus:outline-none focus:border-warning/50"
                >
                  <option value="bottom">Bottom</option>
                  <option value="center">Center</option>
                  <option value="top">Top</option>
                </select>
              </div>
            </div>
          </div>
        </div>
      </div>
  )
}
