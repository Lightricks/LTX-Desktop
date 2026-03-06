import { type MenuDefinition } from '../../components/MenuBar'
import type { TimelineClip } from '../../types/project'
import { TEXT_PRESETS } from '../../types/project'
import { getShortcutLabel, type ToolType } from './video-editor-utils'
import type { KeyboardLayout } from '../../lib/keyboard-shortcuts'

export interface MenuDepsParams {
  selectedClip: TimelineClip | null | undefined
  selectedClipIds: Set<string>
  clips: TimelineClip[]
  tracks: any[]
  subtitles: any[]
  snapEnabled: boolean
  showEffectsBrowser: boolean
  showSourceMonitor: boolean
  showPropertiesPanel: boolean
  showICLoraPanel: boolean
  sourceAsset: any
  activeTool: ToolType
  activeTimeline: any
  timelines: any[]
  kbLayout: KeyboardLayout
  fileInputRef: React.RefObject<HTMLInputElement>
  subtitleFileInputRef: React.RefObject<HTMLInputElement>
  setShowImportTimelineModal: (v: boolean) => void
  setShowExportModal: (v: boolean) => void
  handleExportTimelineXml: () => void
  handleExportSrt: () => void
  undoRef: React.RefObject<() => void>
  redoRef: React.RefObject<() => void>
  cutRef: React.RefObject<() => void>
  copyRef: React.RefObject<() => void>
  pasteRef: React.RefObject<() => void>
  setSelectedClipIds: (v: Set<string>) => void
  handleInsertEdit: () => void
  handleOverwriteEdit: () => void
  matchFrameRef: React.RefObject<() => void>
  setKbEditorOpen: (v: boolean) => void
  splitClipAtPlayhead: (id: string, atTime?: number, batchClipIds?: string[]) => void
  duplicateClip: (id: string) => void
  pushUndo: () => void
  setClips: React.Dispatch<React.SetStateAction<TimelineClip[]>>
  updateClip: (id: string, patch: Partial<TimelineClip>) => void
  setTracks: React.Dispatch<React.SetStateAction<any[]>>
  addTextClip: (style?: any) => void
  addSubtitleTrack: () => void
  createAdjustmentLayerAsset: () => void
  setSnapEnabled: (v: boolean) => void
  fitToViewRef: React.RefObject<() => void>
  setZoom: React.Dispatch<React.SetStateAction<number>>
  setShowSourceMonitor: (v: boolean) => void
  setShowEffectsBrowser: (v: boolean) => void
  setShowPropertiesPanel: (v: boolean) => void
  setShowICLoraPanel: (v: boolean) => void
  setIcLoraSourceClipId: (v: string | null) => void
  setActiveTool: (v: ToolType) => void
  setLastTrimTool: (v: ToolType) => void
  setShowProjectSettings: (v: boolean) => void
  handleAddTimeline: () => void
  handleDuplicateTimeline: (id: string) => void
  handleResetLayout: () => void
  openProjectFolder: () => void
}

export function buildMenuDefinitions(p: MenuDepsParams): MenuDefinition[] {
  return [
    // ── File ──
    // Import/export, timeline management, project settings
    {
      id: 'file',
      label: 'File',
      items: [
        { id: 'new-timeline', label: 'New Timeline', action: () => p.handleAddTimeline() },
        { id: 'duplicate-timeline', label: 'Duplicate Active Timeline', action: () => { if (p.activeTimeline) p.handleDuplicateTimeline(p.activeTimeline.id) }, disabled: !p.activeTimeline },
        { id: 'sep-0', label: '', separator: true },
        { id: 'import-media', label: 'Import Media...', shortcut: 'Ctrl+I', action: () => p.fileInputRef.current?.click() },
        { id: 'import-timeline', label: 'Import Timeline (XML)...', action: () => p.setShowImportTimelineModal(true) },
        { id: 'import-srt', label: 'Import Subtitles (SRT)...', action: () => p.subtitleFileInputRef.current?.click() },
        { id: 'sep-1', label: '', separator: true },
        { id: 'export-timeline', label: 'Export Timeline...', shortcut: 'Ctrl+E', action: () => p.setShowExportModal(true) },
        { id: 'export-xml', label: 'Export FCP7 XML...', action: () => p.handleExportTimelineXml() },
        { id: 'export-srt', label: 'Export Subtitles (SRT)...', action: () => p.handleExportSrt(), disabled: p.subtitles.length === 0 },
        { id: 'sep-2', label: '', separator: true },
        { id: 'project-settings', label: 'Project Settings...', action: () => p.setShowProjectSettings(true) },
      ],
    },

    // ── Edit ──
    // Undo/redo, clipboard, selection, source monitor edits
    {
      id: 'edit',
      label: 'Edit',
      items: [
        { id: 'undo', label: 'Undo', shortcut: getShortcutLabel(p.kbLayout, 'edit.undo'), action: () => p.undoRef.current!() },
        { id: 'redo', label: 'Redo', shortcut: getShortcutLabel(p.kbLayout, 'edit.redo'), action: () => p.redoRef.current!() },
        { id: 'sep-1', label: '', separator: true },
        { id: 'cut', label: 'Cut', shortcut: getShortcutLabel(p.kbLayout, 'edit.cut'), action: () => p.cutRef.current!() },
        { id: 'copy', label: 'Copy', shortcut: getShortcutLabel(p.kbLayout, 'edit.copy'), action: () => p.copyRef.current!() },
        { id: 'paste', label: 'Paste', shortcut: getShortcutLabel(p.kbLayout, 'edit.paste'), action: () => p.pasteRef.current!() },
        { id: 'sep-2', label: '', separator: true },
        { id: 'select-all', label: 'Select All', shortcut: getShortcutLabel(p.kbLayout, 'edit.selectAll'), action: () => p.setSelectedClipIds(new Set(p.clips.map(c => c.id))) },
        { id: 'deselect-all', label: 'Deselect All', shortcut: getShortcutLabel(p.kbLayout, 'edit.deselect'), action: () => p.setSelectedClipIds(new Set()) },
        { id: 'sep-3', label: '', separator: true },
        { id: 'insert-edit', label: 'Insert Edit', shortcut: getShortcutLabel(p.kbLayout, 'edit.insertEdit'), action: () => p.handleInsertEdit(), disabled: !p.sourceAsset },
        { id: 'overwrite-edit', label: 'Overwrite Edit', shortcut: getShortcutLabel(p.kbLayout, 'edit.overwriteEdit'), action: () => p.handleOverwriteEdit(), disabled: !p.sourceAsset },
        { id: 'match-frame', label: 'Match Frame', shortcut: getShortcutLabel(p.kbLayout, 'edit.matchFrame'), action: () => p.matchFrameRef.current!() },
        { id: 'sep-4', label: '', separator: true },
        { id: 'keyboard-shortcuts', label: 'Keyboard Shortcuts...', action: () => p.setKbEditorOpen(true) },
      ],
    },

    // ── Clip ──
    // Operations on selected clip(s): split, duplicate, delete, transform, audio, speed
    {
      id: 'clip',
      label: 'Clip',
      items: [
        { id: 'split', label: 'Split at Playhead', shortcut: getShortcutLabel(p.kbLayout, 'tool.blade'), action: () => { if (p.selectedClip) p.splitClipAtPlayhead(p.selectedClip.id) }, disabled: !p.selectedClip },
        { id: 'duplicate', label: 'Duplicate Clip', action: () => { if (p.selectedClip) p.duplicateClip(p.selectedClip.id) }, disabled: !p.selectedClip },
        { id: 'delete', label: 'Delete', shortcut: getShortcutLabel(p.kbLayout, 'edit.delete'), action: () => { if (p.selectedClipIds.size > 0) { p.pushUndo(); p.setClips(prev => prev.filter(c => !p.selectedClipIds.has(c.id))); p.setSelectedClipIds(new Set()) } }, disabled: p.selectedClipIds.size === 0 },
        { id: 'sep-1', label: '', separator: true },
        { id: 'flip-h', label: 'Flip Horizontal', action: () => { if (p.selectedClip) p.updateClip(p.selectedClip.id, { flipH: !p.selectedClip.flipH }) }, disabled: !p.selectedClip },
        { id: 'flip-v', label: 'Flip Vertical', action: () => { if (p.selectedClip) p.updateClip(p.selectedClip.id, { flipV: !p.selectedClip.flipV }) }, disabled: !p.selectedClip },
        { id: 'reverse', label: 'Reverse', action: () => { if (p.selectedClip) p.updateClip(p.selectedClip.id, { reversed: !p.selectedClip.reversed }) }, disabled: !p.selectedClip },
        { id: 'sep-2', label: '', separator: true },
        { id: 'mute', label: p.selectedClip?.muted ? 'Unmute Clip' : 'Mute Clip', action: () => { if (p.selectedClip) p.updateClip(p.selectedClip.id, { muted: !p.selectedClip.muted }) }, disabled: !p.selectedClip },
        { id: 'link-audio', label: p.selectedClip?.linkedClipIds?.length ? 'Unlink Audio' : 'Link Audio', action: () => {
          if (!p.selectedClip) return
          p.pushUndo()
          if (p.selectedClip.linkedClipIds?.length) {
            const linkedIds = p.selectedClip.linkedClipIds
            p.setClips(prev => prev.map(c => {
              if (c.id === p.selectedClip!.id) return { ...c, linkedClipIds: undefined }
              if (linkedIds.includes(c.id)) return { ...c, linkedClipIds: c.linkedClipIds?.filter(lid => lid !== p.selectedClip!.id) }
              return c
            }))
          }
        }, disabled: !p.selectedClip },
        { id: 'sep-3', label: '', separator: true },
        { id: 'speed-025', label: 'Speed: 0.25x', action: () => { if (p.selectedClip) p.updateClip(p.selectedClip.id, { speed: 0.25 }) }, disabled: !p.selectedClip },
        { id: 'speed-050', label: 'Speed: 0.5x', action: () => { if (p.selectedClip) p.updateClip(p.selectedClip.id, { speed: 0.5 }) }, disabled: !p.selectedClip },
        { id: 'speed-100', label: 'Speed: 1x (Normal)', action: () => { if (p.selectedClip) p.updateClip(p.selectedClip.id, { speed: 1 }) }, disabled: !p.selectedClip },
        { id: 'speed-150', label: 'Speed: 1.5x', action: () => { if (p.selectedClip) p.updateClip(p.selectedClip.id, { speed: 1.5 }) }, disabled: !p.selectedClip },
        { id: 'speed-200', label: 'Speed: 2x', action: () => { if (p.selectedClip) p.updateClip(p.selectedClip.id, { speed: 2 }) }, disabled: !p.selectedClip },
        { id: 'speed-400', label: 'Speed: 4x', action: () => { if (p.selectedClip) p.updateClip(p.selectedClip.id, { speed: 4 }) }, disabled: !p.selectedClip },
      ],
    },

    // ── Sequence ──
    // Timeline-level: add tracks, add layers, add text/captions, snapping
    {
      id: 'sequence',
      label: 'Sequence',
      items: [
        { id: 'add-video-track', label: 'Add Video Track', action: () => { p.pushUndo(); p.setTracks(prev => { const vTracks = prev.filter((t: any) => t.kind === 'video'); const name = `V${vTracks.length + 1}`; return [...prev, { id: `track-${Date.now()}`, name, muted: false, locked: false, kind: 'video' as const }] }) } },
        { id: 'add-audio-track', label: 'Add Audio Track', action: () => { p.pushUndo(); p.setTracks(prev => { const aTracks = prev.filter((t: any) => t.kind === 'audio'); const name = `A${aTracks.length + 1}`; return [...prev, { id: `track-${Date.now()}`, name, muted: false, locked: false, kind: 'audio' as const }] }) } },
        { id: 'add-subtitle-track', label: 'Add Subtitle Track', action: () => p.addSubtitleTrack() },
        { id: 'sep-1', label: '', separator: true },
        { id: 'add-adjustment', label: 'Add Adjustment Layer', action: () => p.createAdjustmentLayerAsset() },
        { id: 'sep-2', label: '', separator: true },
        { id: 'add-text', label: 'Add Text Overlay', action: () => p.addTextClip() },
        { id: 'add-text-lower', label: 'Add Lower Third', action: () => p.addTextClip(TEXT_PRESETS.find((pr: any) => pr.id === 'lower-third-basic')?.style) },
        { id: 'add-text-subtitle', label: 'Add Caption', action: () => p.addTextClip(TEXT_PRESETS.find((pr: any) => pr.id === 'subtitle-style')?.style) },
        { id: 'sep-3', label: '', separator: true },
        { id: 'snap-toggle', label: p.snapEnabled ? 'Disable Snapping' : 'Enable Snapping', shortcut: getShortcutLabel(p.kbLayout, 'timeline.toggleSnap'), action: () => p.setSnapEnabled(!p.snapEnabled) },
      ],
    },

    // ── Tools ──
    // Timeline editing tools (selection, trim, blade, etc.)
    {
      id: 'tools',
      label: 'Tools',
      items: [
        { id: 'tool-select', label: 'Selection Tool', shortcut: getShortcutLabel(p.kbLayout, 'tool.select'), action: () => p.setActiveTool('select') },
        { id: 'tool-blade', label: 'Blade Tool', shortcut: getShortcutLabel(p.kbLayout, 'tool.blade'), action: () => p.setActiveTool('blade') },
        { id: 'sep-1', label: '', separator: true },
        { id: 'tool-ripple', label: 'Ripple Trim', shortcut: getShortcutLabel(p.kbLayout, 'tool.ripple'), action: () => { p.setActiveTool('ripple'); p.setLastTrimTool('ripple') } },
        { id: 'tool-roll', label: 'Roll Trim', shortcut: getShortcutLabel(p.kbLayout, 'tool.roll'), action: () => { p.setActiveTool('roll'); p.setLastTrimTool('roll') } },
        { id: 'tool-slip', label: 'Slip Tool', shortcut: getShortcutLabel(p.kbLayout, 'tool.slip'), action: () => { p.setActiveTool('slip'); p.setLastTrimTool('slip') } },
        { id: 'tool-slide', label: 'Slide Tool', shortcut: getShortcutLabel(p.kbLayout, 'tool.slide'), action: () => { p.setActiveTool('slide'); p.setLastTrimTool('slide') } },
        { id: 'sep-2', label: '', separator: true },
        // IC-LORA HIDDEN - IC-LoRA menu item hidden because IC-LoRA is broken on server
        // { id: 'ic-lora', label: 'IC-LoRA Style Transfer...', action: () => {
        //   p.setIcLoraSourceClipId(p.selectedClip?.type === 'video' ? p.selectedClip.id : null)
        //   p.setShowICLoraPanel(true)
        // }},
      ],
    },

    // ── View ──
    // Panel visibility, timeline zoom, layout
    {
      id: 'view',
      label: 'View',
      items: [
        { id: 'clip-viewer', label: p.showSourceMonitor ? 'Hide Clip Viewer' : 'Show Clip Viewer', action: () => p.setShowSourceMonitor(!p.showSourceMonitor) },
        // EFFECTS HIDDEN - effects-browser menu item hidden because effects are not applied during export
        // { id: 'effects-browser', label: p.showEffectsBrowser ? 'Hide Effects Browser' : 'Show Effects Browser', action: () => p.setShowEffectsBrowser(!p.showEffectsBrowser) },
        { id: 'properties-panel', label: p.showPropertiesPanel ? 'Hide Properties Panel' : 'Show Properties Panel', action: () => p.setShowPropertiesPanel(!p.showPropertiesPanel) },
        // IC-LORA HIDDEN - IC-LoRA panel toggle hidden because IC-LoRA is broken on server
        // { id: 'ic-lora-panel', label: p.showICLoraPanel ? 'Hide IC-LoRA Panel' : 'Show IC-LoRA Panel', action: () => p.setShowICLoraPanel(!p.showICLoraPanel) },
        { id: 'sep-1', label: '', separator: true },
        { id: 'fit-to-view', label: 'Zoom to Fit', shortcut: getShortcutLabel(p.kbLayout, 'timeline.fitToView'), action: () => p.fitToViewRef.current!() },
        { id: 'zoom-in', label: 'Zoom In', shortcut: getShortcutLabel(p.kbLayout, 'timeline.zoomIn'), action: () => p.setZoom(z => Math.min(z * 1.25, 10)) },
        { id: 'zoom-out', label: 'Zoom Out', shortcut: getShortcutLabel(p.kbLayout, 'timeline.zoomOut'), action: () => p.setZoom(z => Math.max(z / 1.25, 0.1)) },
        { id: 'sep-2', label: '', separator: true },
        { id: 'reset-layout', label: 'Reset Layout', action: () => p.handleResetLayout() },
        { id: 'sep-3', label: '', separator: true },
        { id: 'open-project-folder', label: 'Open Project Folder', action: () => p.openProjectFolder() },
      ],
    },

    // ── Help ──
    {
      id: 'help',
      label: 'Help',
      items: [
        { id: 'shortcuts', label: 'Keyboard Shortcuts...', action: () => p.setKbEditorOpen(true) },
        { id: 'about', label: 'About LTX Desktop', action: () => window.dispatchEvent(new CustomEvent('open-settings', { detail: { tab: 'about' } })) },
      ],
    },
  ]
}
