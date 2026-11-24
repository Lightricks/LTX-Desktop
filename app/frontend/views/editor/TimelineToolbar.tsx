import React from 'react'
import { Plus, Gauge, Download, Maximize2, Sparkles, FileUp, FileDown, ZoomOut, ZoomIn } from 'lucide-react'
import { Button } from '../../components/ui/button'
import { Tooltip } from '../../components/ui/tooltip'
import type { TimelineClip, Track, SubtitleClip } from '../../types/project-model'

interface TimelineToolbarProps {
  selectedClip: TimelineClip | null
  updateClip: (id: string, updates: Partial<TimelineClip>) => void
  getMaxClipDuration: (clip: TimelineClip) => number
  setShowExportModal: (v: boolean) => void
  handleResetLayout: () => void
  canUseIcLora: boolean
  onICLoraClip: (clip: TimelineClip) => void
  tracks: Track[]
  subtitleFileInputRef: React.RefObject<HTMLInputElement>
  handleImportSrt: (e: React.ChangeEvent<HTMLInputElement>) => void
  handleExportSrt: () => void
  subtitles: SubtitleClip[]
  zoom: number
  setZoom: (z: number) => void
  getMinZoom: () => number
  centerOnPlayheadRef: React.MutableRefObject<boolean>
  handleFitToView: () => void
}

export function TimelineToolbar({
  selectedClip, updateClip, getMaxClipDuration,
  setShowExportModal, handleResetLayout,
  canUseIcLora, onICLoraClip,
  tracks, subtitleFileInputRef, handleImportSrt, handleExportSrt, subtitles,
  zoom, setZoom, getMinZoom, centerOnPlayheadRef, handleFitToView,
}: TimelineToolbarProps) {
  return (
    <div className="h-9 bg-surface-primary border-t border-separator flex items-center px-3 gap-2 flex-shrink-0">
      <Button variant="outline" size="sm" className="h-6 border-separator text-fg-secondary text-[10px] px-2">
        <Plus className="h-3 w-3 mr-1" />
        Add Clip
      </Button>
      
      {selectedClip && (
        <>
          <div className="w-px h-4 bg-action-hover" />
          <div className="flex items-center gap-1.5 text-[10px] text-fg-secondary">
            <Gauge className="h-3 w-3" />
            <select
              value={selectedClip.speed}
              onChange={(e) => {
                const newSpeed = parseFloat(e.target.value)
                const oldSpeed = selectedClip.speed
                let newDuration = selectedClip.duration * (oldSpeed / newSpeed)
                const maxDur = getMaxClipDuration({ ...selectedClip, speed: newSpeed })
                newDuration = Math.min(newDuration, maxDur)
                newDuration = Math.max(0.5, newDuration)
                updateClip(selectedClip.id, { speed: newSpeed, duration: newDuration })
              }}
              className="bg-action border border-separator rounded px-1.5 py-0.5 text-[10px] text-fg-primary"
            >
              <option value={0.25}>0.25x</option>
              <option value={0.5}>0.5x</option>
              <option value={0.75}>0.75x</option>
              <option value={1}>1x</option>
              <option value={1.25}>1.25x</option>
              <option value={1.5}>1.5x</option>
              <option value={2}>2x</option>
              <option value={4}>4x</option>
            </select>
          </div>
        </>
      )}
      
      <div className="w-px h-4 bg-action-hover" />
      
      <Button
        variant="outline"
        size="sm"
        className="h-6 border-separator text-fg-secondary text-[10px] px-2"
        onClick={() => setShowExportModal(true)}
      >
        <Download className="h-3 w-3 mr-1" />
        Export
      </Button>
      
      <Button
        variant="outline"
        size="sm"
        className="h-6 border-separator text-fg-secondary text-[10px] px-2"
        onClick={handleResetLayout}
        title="Reset panel sizes to default"
      >
        <Maximize2 className="h-3 w-3 mr-1" />
        Layout
      </Button>
      
      <div className="w-px h-4 bg-action-hover" />

      {canUseIcLora && (
        <Button
          variant="outline"
          size="sm"
          className="h-6 border-warning/50 text-fg-warning text-[10px] px-2 hover:bg-warning-soft"
          onClick={() => {
            if (selectedClip?.type === 'video') {
              onICLoraClip(selectedClip)
            }
          }}
          disabled={selectedClip?.type !== 'video'}
          title="Open IC-LoRA style transfer panel"
        >
          <Sparkles className="h-3 w-3 mr-1" />
          IC-LoRA
        </Button>
      )}
      
      
      {/* Subtitle import/export */}
      {tracks.some(t => t.type === 'subtitle') && (
        <>
          <div className="w-px h-4 bg-action-hover" />
          <div className="flex items-center gap-1">
            <button
              onClick={() => subtitleFileInputRef.current?.click()}
              className="h-6 px-2 rounded bg-warning-soft border border-warning/30 text-fg-warning hover:bg-warning-soft text-[10px] flex items-center gap-1 transition-colors"
              title="Import SRT subtitles"
            >
              <FileUp className="h-3 w-3" />
              Import SRT
            </button>
            <button
              onClick={handleExportSrt}
              disabled={subtitles.length === 0}
              className="h-6 px-2 rounded bg-warning-soft border border-warning/30 text-fg-warning hover:bg-warning-soft text-[10px] flex items-center gap-1 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
              title="Export SRT subtitles"
            >
              <FileDown className="h-3 w-3" />
              Export SRT
            </button>
          </div>
          <input
            ref={subtitleFileInputRef}
            type="file"
            accept=".srt"
            onChange={handleImportSrt}
            className="hidden"
          />
        </>
      )}
      
      {/* Spacer */}
      <div className="flex-1" />
      
      {/* Zoom slider bar */}
      <div className="flex items-center gap-2">
        <Tooltip content="Zoom out (-)">
          <button
            onClick={() => { centerOnPlayheadRef.current = true; setZoom(Math.max(getMinZoom(), +(zoom - 0.25).toFixed(2))) }}
            className="p-0.5 rounded hover:bg-action text-fg-tertiary hover:text-fg-secondary transition-colors"
          >
            <ZoomOut className="h-3.5 w-3.5" />
          </button>
        </Tooltip>
        <input
          type="range"
          min={Math.max(1, Math.round(getMinZoom() * 100))}
          max={400}
          step={5}
          value={Math.round(zoom * 100)}
          onChange={(e) => { centerOnPlayheadRef.current = true; setZoom(Math.max(getMinZoom(), +(parseInt(e.target.value) / 100).toFixed(2))) }}
          className="w-28 h-1 accent-blue-500 cursor-pointer"
          title={`Zoom: ${Math.round(zoom * 100)}%`}
        />
        <Tooltip content="Zoom in (+)">
          <button
            onClick={() => { centerOnPlayheadRef.current = true; setZoom(Math.min(4, +(zoom + 0.25).toFixed(2))) }}
            className="p-0.5 rounded hover:bg-action text-fg-tertiary hover:text-fg-secondary transition-colors"
          >
            <ZoomIn className="h-3.5 w-3.5" />
          </button>
        </Tooltip>
        <span className="text-[10px] text-fg-tertiary tabular-nums w-8 text-right">{Math.round(zoom * 100)}%</span>
        <Tooltip content="Fit to view (Ctrl+0)">
          <button
            onClick={handleFitToView}
            className="p-0.5 rounded hover:bg-action text-fg-tertiary hover:text-fg-secondary transition-colors ml-0.5"
          >
            <Maximize2 className="h-3.5 w-3.5" />
          </button>
        </Tooltip>
      </div>
    </div>
  )
}
