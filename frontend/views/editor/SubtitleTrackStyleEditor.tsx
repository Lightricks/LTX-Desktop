import { useMemo } from 'react'
import { X, Palette } from 'lucide-react'
import type { SubtitleStyle } from '../../types/project'
import { DEFAULT_SUBTITLE_STYLE } from '../../types/project'
import { selectSubtitleTrackStyleIdx, selectTracks } from './editor-selectors'
import { useEditorActions, useEditorStore } from './editor-store'

export function SubtitleTrackStyleEditor() {
  const {
    clearSubtitleOverridesForTrack,
    setSubtitleTrackStyleEditorTrack,
    updateSubtitleTrackStyle,
  } = useEditorActions()
  const trackIndex = useEditorStore(selectSubtitleTrackStyleIdx)
  const tracks = useEditorStore(selectTracks)
  const editorModel = useMemo(() => {
    if (trackIndex === null) return null
    const track = tracks[trackIndex]
    if (!track || track.type !== 'subtitle') return null
    return {
      track,
      trackIndex,
      style: {
        ...DEFAULT_SUBTITLE_STYLE,
        ...(track.subtitleStyle || {}),
      },
    }
  }, [trackIndex, tracks])
  if (!editorModel) return null
  const { track, trackIndex: activeTrackIndex, style } = editorModel

  const closeEditor = () => {
    setSubtitleTrackStyleEditorTrack()
  }

  const updateTrackStyle = (patch: Partial<SubtitleStyle>) => {
    updateSubtitleTrackStyle(activeTrackIndex, patch)
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm" onClick={closeEditor}>
      <div className="bg-zinc-900 border border-zinc-700 rounded-xl shadow-2xl w-[380px] max-h-[80vh] flex flex-col overflow-hidden" onClick={e => e.stopPropagation()}>
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-3 border-b border-zinc-800">
          <div className="flex items-center gap-3">
            <div className="w-7 h-7 rounded-lg flex items-center justify-center bg-amber-600/20">
              <Palette className="h-3.5 w-3.5 text-amber-400" />
            </div>
            <div>
              <h2 className="text-sm font-semibold text-white">Track Style</h2>
              <p className="text-[10px] text-zinc-500">{track.name} - applies to all subtitles on this track</p>
            </div>
          </div>
          <button onClick={closeEditor} className="p-1.5 rounded-lg hover:bg-zinc-800 text-zinc-500 hover:text-zinc-300">
            <X className="h-4 w-4" />
          </button>
        </div>

        {/* Body */}
        <div className="flex-1 overflow-auto p-5 space-y-3">
          {/* Preview */}
          <div className="bg-zinc-950 rounded-lg p-4 flex items-center justify-center border border-zinc-800 min-h-[60px]">
            <span
              className="inline-block text-center rounded px-3 py-1.5 leading-snug"
              style={{
                fontSize: `${Math.min(style.fontSize, 28)}px`,
                fontFamily: style.fontFamily,
                fontWeight: style.fontWeight,
                fontStyle: style.italic ? 'italic' : 'normal',
                color: style.color,
                backgroundColor: style.backgroundColor,
                textShadow: '1px 1px 3px rgba(0,0,0,0.8)',
              }}
            >
              Preview subtitle
            </span>
          </div>

          {/* Font size */}
          <div className="flex items-center justify-between">
            <span className="text-[10px] text-zinc-400">Font Size</span>
            <div className="flex items-center gap-2">
              <input
                type="range" min={16} max={72} value={style.fontSize}
                onChange={e => updateTrackStyle({ fontSize: parseInt(e.target.value) })}
                className="w-24 accent-amber-500"
              />
              <span className="text-[10px] text-zinc-300 w-8 text-right tabular-nums">{style.fontSize}px</span>
            </div>
          </div>

          {/* Font family */}
          <div className="flex items-center justify-between">
            <span className="text-[10px] text-zinc-400">Font</span>
            <select
              value={style.fontFamily}
              onChange={e => updateTrackStyle({ fontFamily: e.target.value })}
              className="bg-zinc-800 border border-zinc-700 rounded px-2 py-0.5 text-[10px] text-white focus:outline-none focus:border-amber-500/50"
            >
              <option value="sans-serif">Sans-Serif</option>
              <option value="serif">Serif</option>
              <option value="monospace">Monospace</option>
              <option value="'Arial', sans-serif">Arial</option>
              <option value="'Helvetica Neue', sans-serif">Helvetica</option>
              <option value="'Georgia', serif">Georgia</option>
              <option value="'Courier New', monospace">Courier New</option>
              <option value="'Times New Roman', serif">Times New Roman</option>
            </select>
          </div>

          {/* Bold / Italic */}
          <div className="flex items-center justify-between">
            <span className="text-[10px] text-zinc-400">Style</span>
            <div className="flex items-center gap-2">
              <button
                onClick={() => updateTrackStyle({ fontWeight: style.fontWeight === 'bold' ? 'normal' : 'bold' })}
                className={`px-2.5 py-1 rounded text-[10px] font-bold ${style.fontWeight === 'bold' ? 'bg-amber-600/30 text-amber-300 border border-amber-500/40' : 'bg-zinc-800 text-zinc-400 border border-zinc-700'}`}
              >
                B
              </button>
              <button
                onClick={() => updateTrackStyle({ italic: !style.italic })}
                className={`px-2.5 py-1 rounded text-[10px] italic ${style.italic ? 'bg-amber-600/30 text-amber-300 border border-amber-500/40' : 'bg-zinc-800 text-zinc-400 border border-zinc-700'}`}
              >
                I
              </button>
            </div>
          </div>

          {/* Text color */}
          <div className="flex items-center justify-between">
            <span className="text-[10px] text-zinc-400">Text Color</span>
            <input type="color" value={style.color} onChange={e => updateTrackStyle({ color: e.target.value })} className="w-7 h-6 rounded cursor-pointer border border-zinc-700" />
          </div>

          {/* Background toggle + color */}
          <div className="flex items-center justify-between">
            <span className="text-[10px] text-zinc-400">Background</span>
            <div className="flex items-center gap-1.5">
              <button
                onClick={() => updateTrackStyle({ backgroundColor: style.backgroundColor === 'transparent' ? '#000000AA' : 'transparent' })}
                className={`px-2 py-0.5 rounded text-[9px] border ${style.backgroundColor !== 'transparent' ? 'bg-amber-600/20 text-amber-300 border-amber-500/40' : 'bg-zinc-800 text-zinc-500 border-zinc-700'}`}
              >
                {style.backgroundColor !== 'transparent' ? 'On' : 'Off'}
              </button>
              {style.backgroundColor !== 'transparent' && (
                <input type="color" value={style.backgroundColor.slice(0, 7)} onChange={e => updateTrackStyle({ backgroundColor: e.target.value + 'CC' })} className="w-7 h-6 rounded cursor-pointer border border-zinc-700" />
              )}
            </div>
          </div>

          {/* Position */}
          <div className="flex items-center justify-between">
            <span className="text-[10px] text-zinc-400">Position</span>
            <select
              value={style.position}
              onChange={e => updateTrackStyle({ position: e.target.value as SubtitleStyle['position'] })}
              className="bg-zinc-800 border border-zinc-700 rounded px-2 py-0.5 text-[10px] text-white focus:outline-none focus:border-amber-500/50"
            >
              <option value="bottom">Bottom</option>
              <option value="center">Center</option>
              <option value="top">Top</option>
            </select>
          </div>

          <div className="border-t border-zinc-800 pt-3 mt-3">
            <button
              onClick={() => {
                clearSubtitleOverridesForTrack(activeTrackIndex)
                setSubtitleTrackStyleEditorTrack()
              }}
              className="w-full px-3 py-2 rounded-lg bg-zinc-800 border border-zinc-700 text-zinc-300 text-xs hover:bg-zinc-700 transition-colors text-center"
            >
              Apply to all &amp; reset overrides
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
