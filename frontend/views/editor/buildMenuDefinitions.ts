import { type MenuDefinition } from '../../components/MenuBar'
import { TEXT_PRESETS, type TimelineClip } from '../../types/project'
import type { KeyboardLayout } from '../../lib/keyboard-shortcuts'
import { useMemo } from 'react'
import { useShallow } from 'zustand/react/shallow'
import {
  selectCanInsertEdit,
  selectCanRedo,
  selectCanUndo,
  selectCanOverwriteEdit,
  selectCanUseClipboard,
  selectCurrentTime,
  selectMenuState,
} from './editor-selectors'
import { useEditorActions, useEditorStore } from './editor-store'
import { getShortcutLabel } from './video-editor-utils'

export interface MenuDepsParams {
  kbLayout: KeyboardLayout
  fileInputRef: React.RefObject<HTMLInputElement>
  subtitleFileInputRef: React.RefObject<HTMLInputElement>
  handleExportTimelineXml: () => void
  handleExportSrt: () => void
  handleInsertEdit: () => void
  handleOverwriteEdit: () => void
  handleMatchFrame: () => void
  setKbEditorOpen: (v: boolean) => void
  fitToViewRef: React.RefObject<() => void>
  handleResetLayout: () => void
  canUseIcLora: boolean
  onICLoraClip: (clip: TimelineClip) => void
}

export function useBuildMenuDefinitions(p: MenuDepsParams): MenuDefinition[] {
  const actions = useEditorActions()
  const menuState = useEditorStore(useShallow(selectMenuState))
  const canUseClipboard = useEditorStore(selectCanUseClipboard)
  const canUndo = useEditorStore(selectCanUndo)
  const canRedo = useEditorStore(selectCanRedo)
  const canInsertEdit = useEditorStore(selectCanInsertEdit)
  const canOverwriteEdit = useEditorStore(selectCanOverwriteEdit)
  const currentTime = useEditorStore(selectCurrentTime)

  return useMemo(() => ([
    {
      id: 'file',
      label: 'File',
      items: [
        { id: 'new-timeline', label: 'New Timeline', action: () => actions.createTimeline() },
        {
          id: 'duplicate-timeline',
          label: 'Duplicate Active Timeline',
          action: () => {
            if (menuState.activeTimeline) {
              actions.duplicateTimeline(menuState.activeTimeline.id)
            }
          },
          disabled: !menuState.activeTimeline,
        },
        { id: 'sep-0', label: '', separator: true },
        { id: 'import-media', label: 'Import Media...', shortcut: 'Ctrl+I', action: () => p.fileInputRef.current?.click() },
        { id: 'import-timeline', label: 'Import Timeline (XML)...', action: () => actions.openImportTimelineModal() },
        { id: 'import-srt', label: 'Import Subtitles (SRT)...', action: () => p.subtitleFileInputRef.current?.click() },
        { id: 'sep-1', label: '', separator: true },
        { id: 'export-timeline', label: 'Export Timeline...', shortcut: 'Ctrl+E', action: () => actions.openExportModal() },
        { id: 'export-xml', label: 'Export FCP7 XML...', action: () => p.handleExportTimelineXml() },
        { id: 'export-srt', label: 'Export Subtitles (SRT)...', action: () => p.handleExportSrt(), disabled: menuState.subtitles.length === 0 },
      ],
    },
    {
      id: 'edit',
      label: 'Edit',
      items: [
        { id: 'undo', label: 'Undo', shortcut: getShortcutLabel(p.kbLayout, 'edit.undo'), action: () => actions.undo(), disabled: !canUndo },
        { id: 'redo', label: 'Redo', shortcut: getShortcutLabel(p.kbLayout, 'edit.redo'), action: () => actions.redo(), disabled: !canRedo },
        { id: 'sep-1', label: '', separator: true },
        { id: 'cut', label: 'Cut', shortcut: getShortcutLabel(p.kbLayout, 'edit.cut'), action: () => actions.cutSelection() },
        { id: 'copy', label: 'Copy', shortcut: getShortcutLabel(p.kbLayout, 'edit.copy'), action: () => actions.copySelection() },
        { id: 'paste', label: 'Paste', shortcut: getShortcutLabel(p.kbLayout, 'edit.paste'), action: () => actions.pasteSelection(), disabled: !canUseClipboard },
        { id: 'sep-2', label: '', separator: true },
        { id: 'select-all', label: 'Select All', shortcut: getShortcutLabel(p.kbLayout, 'edit.selectAll'), action: () => actions.selectAllClips() },
        { id: 'deselect-all', label: 'Deselect All', shortcut: getShortcutLabel(p.kbLayout, 'edit.deselect'), action: () => actions.clearClipSelection() },
        { id: 'sep-3', label: '', separator: true },
        { id: 'insert-edit', label: 'Insert Edit', shortcut: getShortcutLabel(p.kbLayout, 'edit.insertEdit'), action: () => p.handleInsertEdit(), disabled: !canInsertEdit },
        { id: 'overwrite-edit', label: 'Overwrite Edit', shortcut: getShortcutLabel(p.kbLayout, 'edit.overwriteEdit'), action: () => p.handleOverwriteEdit(), disabled: !canOverwriteEdit },
        { id: 'match-frame', label: 'Match Frame', shortcut: getShortcutLabel(p.kbLayout, 'edit.matchFrame'), action: () => p.handleMatchFrame() },
        { id: 'sep-4', label: '', separator: true },
        { id: 'keyboard-shortcuts', label: 'Keyboard Shortcuts...', action: () => p.setKbEditorOpen(true) },
      ],
    },
    {
      id: 'clip',
      label: 'Clip',
      items: [
        {
          id: 'split',
          label: 'Split at Playhead',
          shortcut: getShortcutLabel(p.kbLayout, 'tool.blade'),
          action: () => {
            if (menuState.selectedClip) {
              actions.splitClipsAtTime([menuState.selectedClip.id], currentTime)
            }
          },
          disabled: !menuState.selectedClip,
        },
        {
          id: 'duplicate',
          label: 'Duplicate Clip',
          action: () => {
            if (menuState.selectedClip) {
              actions.duplicateClips([menuState.selectedClip.id])
            }
          },
          disabled: !menuState.selectedClip,
        },
        {
          id: 'delete',
          label: 'Delete',
          shortcut: getShortcutLabel(p.kbLayout, 'edit.delete'),
          action: () => actions.deleteClips([...menuState.selectedClipIds]),
          disabled: menuState.selectedClipIds.size === 0,
        },
        { id: 'sep-1', label: '', separator: true },
        {
          id: 'flip-h',
          label: 'Flip Horizontal',
          action: () => {
            if (menuState.selectedClip) {
              actions.updateClip(menuState.selectedClip.id, { flipH: !menuState.selectedClip.flipH })
            }
          },
          disabled: !menuState.selectedClip,
        },
        {
          id: 'flip-v',
          label: 'Flip Vertical',
          action: () => {
            if (menuState.selectedClip) {
              actions.updateClip(menuState.selectedClip.id, { flipV: !menuState.selectedClip.flipV })
            }
          },
          disabled: !menuState.selectedClip,
        },
        {
          id: 'reverse',
          label: 'Reverse',
          action: () => {
            if (menuState.selectedClip) {
              actions.toggleClipReverse(menuState.selectedClip.id)
            }
          },
          disabled: !menuState.selectedClip,
        },
        { id: 'sep-2', label: '', separator: true },
        {
          id: 'mute',
          label: menuState.selectedClip?.muted ? 'Unmute Clip' : 'Mute Clip',
          action: () => {
            if (menuState.selectedClip) {
              actions.toggleClipMute(menuState.selectedClip.id)
            }
          },
          disabled: !menuState.selectedClip,
        },
        {
          id: 'link-audio',
          label: menuState.selectedClip?.linkedClipIds?.length ? 'Unlink Audio' : 'Link Audio',
          action: () => {
            if (menuState.selectedClip?.linkedClipIds?.length) {
              actions.unlinkClipGroup(menuState.selectedClip.id)
            }
          },
          disabled: !menuState.selectedClip,
        },
        { id: 'sep-3', label: '', separator: true },
        { id: 'speed-025', label: 'Speed: 0.25x', action: () => menuState.selectedClip && actions.updateClip(menuState.selectedClip.id, { speed: 0.25 }), disabled: !menuState.selectedClip },
        { id: 'speed-050', label: 'Speed: 0.5x', action: () => menuState.selectedClip && actions.updateClip(menuState.selectedClip.id, { speed: 0.5 }), disabled: !menuState.selectedClip },
        { id: 'speed-100', label: 'Speed: 1x (Normal)', action: () => menuState.selectedClip && actions.updateClip(menuState.selectedClip.id, { speed: 1 }), disabled: !menuState.selectedClip },
        { id: 'speed-150', label: 'Speed: 1.5x', action: () => menuState.selectedClip && actions.updateClip(menuState.selectedClip.id, { speed: 1.5 }), disabled: !menuState.selectedClip },
        { id: 'speed-200', label: 'Speed: 2x', action: () => menuState.selectedClip && actions.updateClip(menuState.selectedClip.id, { speed: 2 }), disabled: !menuState.selectedClip },
        { id: 'speed-400', label: 'Speed: 4x', action: () => menuState.selectedClip && actions.updateClip(menuState.selectedClip.id, { speed: 4 }), disabled: !menuState.selectedClip },
      ],
    },
    {
      id: 'sequence',
      label: 'Sequence',
      items: [
        { id: 'add-video-track', label: 'Add Video Track', action: () => actions.addTrack('video') },
        { id: 'add-audio-track', label: 'Add Audio Track', action: () => actions.addTrack('audio') },
        { id: 'add-subtitle-track', label: 'Add Subtitle Track', action: () => actions.addSubtitleTrack() },
        { id: 'sep-1', label: '', separator: true },
        { id: 'add-adjustment', label: 'Add Adjustment Layer', action: () => actions.createAdjustmentLayerAsset() },
        { id: 'sep-2', label: '', separator: true },
        { id: 'add-text', label: 'Add Text Overlay', action: () => actions.addTextClip() },
        { id: 'add-text-lower', label: 'Add Lower Third', action: () => actions.addTextClip({ style: TEXT_PRESETS.find(pr => pr.id === 'lower-third-basic')?.style }) },
        { id: 'add-text-subtitle', label: 'Add Caption', action: () => actions.addTextClip({ style: TEXT_PRESETS.find(pr => pr.id === 'subtitle-style')?.style }) },
        { id: 'sep-3', label: '', separator: true },
        { id: 'snap-toggle', label: menuState.snapEnabled ? 'Disable Snapping' : 'Enable Snapping', shortcut: getShortcutLabel(p.kbLayout, 'timeline.toggleSnap'), action: () => actions.toggleSnap() },
      ],
    },
    {
      id: 'tools',
      label: 'Tools',
      items: [
        { id: 'tool-select', label: 'Selection Tool', shortcut: getShortcutLabel(p.kbLayout, 'tool.select'), action: () => actions.setActiveTool('select') },
        { id: 'tool-blade', label: 'Blade Tool', shortcut: getShortcutLabel(p.kbLayout, 'tool.blade'), action: () => actions.setActiveTool('blade') },
        { id: 'sep-1', label: '', separator: true },
        { id: 'tool-ripple', label: 'Ripple Trim', shortcut: getShortcutLabel(p.kbLayout, 'tool.ripple'), action: () => { actions.setActiveTool('ripple'); actions.setLastTrimTool('ripple') } },
        { id: 'tool-roll', label: 'Roll Trim', shortcut: getShortcutLabel(p.kbLayout, 'tool.roll'), action: () => { actions.setActiveTool('roll'); actions.setLastTrimTool('roll') } },
        { id: 'tool-slip', label: 'Slip Tool', shortcut: getShortcutLabel(p.kbLayout, 'tool.slip'), action: () => { actions.setActiveTool('slip'); actions.setLastTrimTool('slip') } },
        { id: 'tool-slide', label: 'Slide Tool', shortcut: getShortcutLabel(p.kbLayout, 'tool.slide'), action: () => { actions.setActiveTool('slide'); actions.setLastTrimTool('slide') } },
        { id: 'sep-2', label: '', separator: true },
        ...(p.canUseIcLora ? [{
          id: 'ic-lora',
          label: 'IC-LoRA Style Transfer...',
          action: () => {
            if (menuState.selectedClip?.type === 'video') {
              p.onICLoraClip(menuState.selectedClip)
            }
          },
          disabled: menuState.selectedClip?.type !== 'video',
        }] : []),
      ],
    },
    {
      id: 'view',
      label: 'View',
      items: [
        { id: 'clip-viewer', label: menuState.showSourceMonitor ? 'Hide Clip Viewer' : 'Show Clip Viewer', action: () => actions.setShowSourceMonitor(!menuState.showSourceMonitor) },
        { id: 'properties-panel', label: menuState.showPropertiesPanel ? 'Hide Properties Panel' : 'Show Properties Panel', action: () => actions.setShowPropertiesPanel(!menuState.showPropertiesPanel) },
        { id: 'sep-1', label: '', separator: true },
        { id: 'fit-to-view', label: 'Zoom to Fit', shortcut: getShortcutLabel(p.kbLayout, 'timeline.fitToView'), action: () => p.fitToViewRef.current?.() },
        { id: 'zoom-in', label: 'Zoom In', shortcut: getShortcutLabel(p.kbLayout, 'timeline.zoomIn'), action: () => actions.zoomIn() },
        { id: 'zoom-out', label: 'Zoom Out', shortcut: getShortcutLabel(p.kbLayout, 'timeline.zoomOut'), action: () => actions.zoomOut() },
        { id: 'sep-2', label: '', separator: true },
        { id: 'reset-layout', label: 'Reset Layout', action: () => p.handleResetLayout() },
      ],
    },
    {
      id: 'help',
      label: 'Help',
      items: [
        { id: 'shortcuts', label: 'Keyboard Shortcuts...', action: () => p.setKbEditorOpen(true) },
        { id: 'about', label: 'About LTX Desktop', action: () => window.dispatchEvent(new CustomEvent('open-settings', { detail: { tab: 'about' } })) },
      ],
    },
  ]), [
    actions,
    canInsertEdit,
    canOverwriteEdit,
    canUseClipboard,
    currentTime,
    menuState,
    p.canUseIcLora,
    p.fileInputRef,
    p.fitToViewRef,
    p.handleExportSrt,
    p.handleExportTimelineXml,
    p.handleInsertEdit,
    p.handleMatchFrame,
    p.handleOverwriteEdit,
    p.handleResetLayout,
    p.kbLayout,
    p.onICLoraClip,
    p.setKbEditorOpen,
    p.subtitleFileInputRef,
  ])
}
