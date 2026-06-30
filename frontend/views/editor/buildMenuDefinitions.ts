import { type MenuDefinition } from '../../components/MenuBar'
import { TEXT_PRESETS } from '../../types/project'
import type { TimelineClip } from '../../types/project-model'
import type { KeyboardLayout } from '../../lib/keyboard-shortcuts'
import { useMemo } from 'react'
import { useT } from '../../lib/i18n'
import { shallow } from 'zustand/vanilla/shallow'
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
  const menuState = useEditorStore(selectMenuState, shallow)
  const canUseClipboard = useEditorStore(selectCanUseClipboard)
  const canUndo = useEditorStore(selectCanUndo)
  const canRedo = useEditorStore(selectCanRedo)
  const canInsertEdit = useEditorStore(selectCanInsertEdit)
  const canOverwriteEdit = useEditorStore(selectCanOverwriteEdit)
  const currentTime = useEditorStore(selectCurrentTime)
  const { t } = useT()

  return useMemo(() => ([
    {
      id: 'file',
      label: t('menu.file'),
      items: [
        { id: 'new-timeline', label: t('menu.newTimeline'), action: () => actions.createTimeline() },
        {
          id: 'duplicate-timeline',
          label: t('menu.duplicateTimeline'),
          action: () => {
            if (menuState.activeTimeline) {
              actions.duplicateTimeline(menuState.activeTimeline.id)
            }
          },
          disabled: !menuState.activeTimeline,
        },
        { id: 'sep-0', label: '', separator: true },
        { id: 'import-media', label: t('menu.importMedia'), shortcut: 'Ctrl+I', action: () => p.fileInputRef.current?.click() },
        { id: 'import-timeline', label: t('menu.importTimeline'), action: () => actions.openImportTimelineModal() },
        { id: 'import-srt', label: t('menu.importSubtitles'), action: () => p.subtitleFileInputRef.current?.click() },
        { id: 'sep-1', label: '', separator: true },
        { id: 'export-timeline', label: t('menu.exportTimeline'), shortcut: 'Ctrl+E', action: () => actions.openExportModal() },
        { id: 'export-xml', label: t('menu.exportFcp7Xml'), action: () => p.handleExportTimelineXml() },
        { id: 'export-srt', label: t('menu.exportSubtitles'), action: () => p.handleExportSrt(), disabled: menuState.subtitles.length === 0 },
      ],
    },
    {
      id: 'edit',
      label: t('menu.edit'),
      items: [
        { id: 'undo', label: t('editor.undo'), shortcut: getShortcutLabel(p.kbLayout, 'edit.undo'), action: () => actions.undo(), disabled: !canUndo },
        { id: 'redo', label: t('editor.redo'), shortcut: getShortcutLabel(p.kbLayout, 'edit.redo'), action: () => actions.redo(), disabled: !canRedo },
        { id: 'sep-1', label: '', separator: true },
        { id: 'cut', label: t('editor.cut'), shortcut: getShortcutLabel(p.kbLayout, 'edit.cut'), action: () => actions.cutSelection() },
        { id: 'copy', label: t('editor.copy'), shortcut: getShortcutLabel(p.kbLayout, 'edit.copy'), action: () => actions.copySelection() },
        { id: 'paste', label: t('editor.paste'), shortcut: getShortcutLabel(p.kbLayout, 'edit.paste'), action: () => actions.pasteSelection(), disabled: !canUseClipboard },
        { id: 'sep-2', label: '', separator: true },
        { id: 'select-all', label: t('editor.selectAll'), shortcut: getShortcutLabel(p.kbLayout, 'edit.selectAll'), action: () => actions.selectAllClips() },
        { id: 'deselect-all', label: t('editor.deselectAll'), shortcut: getShortcutLabel(p.kbLayout, 'edit.deselect'), action: () => actions.clearClipSelection() },
        { id: 'sep-3', label: '', separator: true },
        { id: 'insert-edit', label: t('editor.insertEdit'), shortcut: getShortcutLabel(p.kbLayout, 'edit.insertEdit'), action: () => p.handleInsertEdit(), disabled: !canInsertEdit },
        { id: 'overwrite-edit', label: t('editor.overwriteEdit'), shortcut: getShortcutLabel(p.kbLayout, 'edit.overwriteEdit'), action: () => p.handleOverwriteEdit(), disabled: !canOverwriteEdit },
        { id: 'match-frame', label: t('editor.matchFrame'), shortcut: getShortcutLabel(p.kbLayout, 'edit.matchFrame'), action: () => p.handleMatchFrame() },
        { id: 'sep-4', label: '', separator: true },
        { id: 'keyboard-shortcuts', label: t('keyboard.title') + '...', action: () => p.setKbEditorOpen(true) },
      ],
    },
    {
      id: 'clip',
      label: t('menu.clip'),
      items: [
        {
          id: 'split',
          label: t('menu.splitAtPlayhead'),
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
          label: t('menu.duplicateClip'),
          action: () => {
            if (menuState.selectedClip) {
              actions.duplicateClips([menuState.selectedClip.id])
            }
          },
          disabled: !menuState.selectedClip,
        },
        {
          id: 'delete',
          label: t('editor.delete'),
          shortcut: getShortcutLabel(p.kbLayout, 'edit.delete'),
          action: () => actions.deleteClips([...menuState.selectedClipIds]),
          disabled: menuState.selectedClipIds.size === 0,
        },
        { id: 'sep-1', label: '', separator: true },
        {
          id: 'flip-h',
          label: t('menu.flipHorizontal'),
          action: () => {
            if (menuState.selectedClip) {
              actions.updateClip(menuState.selectedClip.id, { flipH: !menuState.selectedClip.flipH })
            }
          },
          disabled: !menuState.selectedClip,
        },
        {
          id: 'flip-v',
          label: t('menu.flipVertical'),
          action: () => {
            if (menuState.selectedClip) {
              actions.updateClip(menuState.selectedClip.id, { flipV: !menuState.selectedClip.flipV })
            }
          },
          disabled: !menuState.selectedClip,
        },
        {
          id: 'reverse',
          label: t('menu.reverse'),
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
          label: menuState.selectedClip?.muted ? t('menu.unmuteClip') : t('menu.muteClip'),
          action: () => {
            if (menuState.selectedClip) {
              actions.toggleClipMute(menuState.selectedClip.id)
            }
          },
          disabled: !menuState.selectedClip,
        },
        {
          id: 'link-audio',
          label: menuState.selectedClip?.linkedClipIds?.length ? t('menu.unlinkAudio') : t('menu.linkAudio'),
          action: () => {
            if (menuState.selectedClip?.linkedClipIds?.length) {
              actions.unlinkClipGroup(menuState.selectedClip.id)
            }
          },
          disabled: !menuState.selectedClip,
        },
        { id: 'sep-3', label: '', separator: true },
        { id: 'speed-025', label: t('menu.speed025'), action: () => menuState.selectedClip && actions.updateClip(menuState.selectedClip.id, { speed: 0.25 }), disabled: !menuState.selectedClip },
        { id: 'speed-050', label: t('menu.speed050'), action: () => menuState.selectedClip && actions.updateClip(menuState.selectedClip.id, { speed: 0.5 }), disabled: !menuState.selectedClip },
        { id: 'speed-100', label: t('menu.speed100'), action: () => menuState.selectedClip && actions.updateClip(menuState.selectedClip.id, { speed: 1 }), disabled: !menuState.selectedClip },
        { id: 'speed-150', label: t('menu.speed150'), action: () => menuState.selectedClip && actions.updateClip(menuState.selectedClip.id, { speed: 1.5 }), disabled: !menuState.selectedClip },
        { id: 'speed-200', label: t('menu.speed200'), action: () => menuState.selectedClip && actions.updateClip(menuState.selectedClip.id, { speed: 2 }), disabled: !menuState.selectedClip },
        { id: 'speed-400', label: t('menu.speed400'), action: () => menuState.selectedClip && actions.updateClip(menuState.selectedClip.id, { speed: 4 }), disabled: !menuState.selectedClip },
      ],
    },
    {
      id: 'sequence',
      label: t('menu.sequence'),
      items: [
        { id: 'add-video-track', label: t('menu.addVideoTrack'), action: () => actions.addTrack('video') },
        { id: 'add-audio-track', label: t('menu.addAudioTrack'), action: () => actions.addTrack('audio') },
        { id: 'add-subtitle-track', label: t('menu.addSubtitleTrack'), action: () => actions.addSubtitleTrack() },
        { id: 'sep-1', label: '', separator: true },
        { id: 'add-adjustment', label: t('menu.addAdjustmentLayer'), action: () => actions.createAdjustmentLayerAsset() },
        { id: 'sep-2', label: '', separator: true },
        { id: 'add-text', label: t('menu.addTextOverlay'), action: () => actions.addTextClip() },
        { id: 'add-text-lower', label: t('menu.addLowerThird'), action: () => actions.addTextClip({ style: TEXT_PRESETS.find(pr => pr.id === 'lower-third-basic')?.style }) },
        { id: 'add-text-subtitle', label: t('menu.addCaption'), action: () => actions.addTextClip({ style: TEXT_PRESETS.find(pr => pr.id === 'subtitle-style')?.style }) },
        { id: 'sep-3', label: '', separator: true },
        { id: 'snap-toggle', label: menuState.snapEnabled ? t('menu.disableSnapping') : t('menu.enableSnapping'), shortcut: getShortcutLabel(p.kbLayout, 'timeline.toggleSnap'), action: () => actions.toggleSnap() },
      ],
    },
    {
      id: 'tools',
      label: t('menu.tools'),
      items: [
        { id: 'tool-select', label: t('menu.selectionTool'), shortcut: getShortcutLabel(p.kbLayout, 'tool.select'), action: () => actions.setActiveTool('select') },
        { id: 'tool-blade', label: t('menu.bladeTool'), shortcut: getShortcutLabel(p.kbLayout, 'tool.blade'), action: () => actions.setActiveTool('blade') },
        { id: 'sep-1', label: '', separator: true },
        { id: 'tool-ripple', label: t('menu.rippleTrim'), shortcut: getShortcutLabel(p.kbLayout, 'tool.ripple'), action: () => { actions.setActiveTool('ripple'); actions.setLastTrimTool('ripple') } },
        { id: 'tool-roll', label: t('menu.rollTrim'), shortcut: getShortcutLabel(p.kbLayout, 'tool.roll'), action: () => { actions.setActiveTool('roll'); actions.setLastTrimTool('roll') } },
        { id: 'tool-slip', label: t('menu.slipTool'), shortcut: getShortcutLabel(p.kbLayout, 'tool.slip'), action: () => { actions.setActiveTool('slip'); actions.setLastTrimTool('slip') } },
        { id: 'tool-slide', label: t('menu.slideTool'), shortcut: getShortcutLabel(p.kbLayout, 'tool.slide'), action: () => { actions.setActiveTool('slide'); actions.setLastTrimTool('slide') } },
        { id: 'sep-2', label: '', separator: true },
        ...(p.canUseIcLora ? [{
          id: 'ic-lora',
          label: t('menu.icLoraStyle'),
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
      label: t('menu.view'),
      items: [
        { id: 'clip-viewer', label: menuState.showSourceMonitor ? t('menu.hideClipViewer') : t('menu.showClipViewer'), action: () => actions.setShowSourceMonitor(!menuState.showSourceMonitor) },
        { id: 'properties-panel', label: menuState.showPropertiesPanel ? t('menu.hideProperties') : t('menu.showProperties'), action: () => actions.setShowPropertiesPanel(!menuState.showPropertiesPanel) },
        { id: 'sep-1', label: '', separator: true },
        { id: 'fit-to-view', label: t('menu.zoomToFit'), shortcut: getShortcutLabel(p.kbLayout, 'timeline.fitToView'), action: () => p.fitToViewRef.current?.() },
        { id: 'zoom-in', label: t('editor.zoomIn'), shortcut: getShortcutLabel(p.kbLayout, 'timeline.zoomIn'), action: () => actions.zoomIn() },
        { id: 'zoom-out', label: t('editor.zoomOut'), shortcut: getShortcutLabel(p.kbLayout, 'timeline.zoomOut'), action: () => actions.zoomOut() },
        { id: 'sep-2', label: '', separator: true },
        { id: 'reset-layout', label: t('menu.resetLayout'), action: () => p.handleResetLayout() },
      ],
    },
    {
      id: 'help',
      label: t('menu.help'),
      items: [
        { id: 'shortcuts', label: t('keyboard.title') + '...', action: () => p.setKbEditorOpen(true) },
        { id: 'about', label: t('menu.aboutLtx'), action: () => window.dispatchEvent(new CustomEvent('open-settings', { detail: { tab: 'about' } })) },
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
    t,
  ])
}
