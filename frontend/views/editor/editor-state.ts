import type {
  Asset,
  AssetTake,
  GenerationParams,
  SubtitleClip,
  SubtitleStyle,
  Timeline,
  TimelineClip,
  Track,
} from '../../types/project'
import {
  DEFAULT_LAYOUT,
  type EditorLayout,
  type ToolType,
} from './video-editor-utils'

export interface EditorModel {
  assets: Asset[]
  timelines: Timeline[]
  activeTimelineId: string | null
}

export interface TimelineGapSelection {
  trackIndex: number
  startTime: number
  endTime: number
}

export interface TimelineInOutRange {
  inPoint: number | null
  outPoint: number | null
}

export interface EditorUndoSnapshot {
  assets: Asset[]
  timelines: Timeline[]
}

export interface EditorClipboardState {
  kind: 'clips' | null
  clips: TimelineClip[]
  copiedFromTimelineId: string | null
}

export interface EditorHistoryState {
  undoStack: EditorUndoSnapshot[]
  redoStack: EditorUndoSnapshot[]
}

export interface EditorProjectSyncState {
  dirty: boolean
}

export interface EditorSelectionState {
  clipIds: Set<string>
  subtitleId: string | null
  editingSubtitleId: string | null
  gap: TimelineGapSelection | null
}

export interface EditorTransportState {
  currentTime: number
  isPlaying: boolean
  shuttleSpeed: number
  playingInOut: boolean
  timelineInOutMap: Record<string, TimelineInOutRange>
}

export interface EditorToolsState {
  zoom: number
  snapEnabled: boolean
  activeTool: ToolType
  lastTrimTool: ToolType
}

export interface EditorUiState {
  showImportTimelineModal: boolean
  showExportModal: boolean
  showSourceMonitor: boolean
  showPropertiesPanel: boolean
  showEffectsBrowser: boolean
  activeFocusArea: 'source' | 'timeline'
  sourceSplitPercent: number
  hasSourceAsset: boolean
  openTimelineIds: Set<string>
  renamingTimelineId: string | null
  renameValue: string
  renameSource: 'tab' | 'panel'
  layout: EditorLayout
  subtitleTrackStyleIdx: number | null
  gapGenerateMode: 'text-to-video' | 'image-to-video' | 'text-to-image' | null
}

export interface EditorRegenerationState {
  regeneratingAssetId: string | null
  regeneratingClipId: string | null
  preError: string | null
}

export interface EditorSessionState {
  selection: EditorSelectionState
  transport: EditorTransportState
  tools: EditorToolsState
  ui: EditorUiState
  regeneration: EditorRegenerationState
  clipboard: EditorClipboardState
}

export interface EditorState {
  editorModel: EditorModel
  session: EditorSessionState
  history: EditorHistoryState
  projectSync: EditorProjectSyncState
}

export interface TimelineListItem {
  timeline: Timeline
  isActive: boolean
  isOpen: boolean
  isRenaming: boolean
  clipCount: number
  duration: number
}

export interface ClipDimensions {
  width: number
  height: number
}

export interface ClipResolutionInfo {
  label: string
  color: string
  height: number
}

export interface OrderedTrackEntry {
  track: Track
  realIndex: number
  displayRow: number
}

export interface TimelineCutPoint {
  leftClip: TimelineClip
  rightClip: TimelineClip
  time: number
  trackIndex: number
  hasDissolve: boolean
}

export interface ClipMetadata {
  liveAsset: Asset | null | undefined
  dimensions: ClipDimensions | null
  resolution: ClipResolutionInfo | null
  generationParams: GenerationParams | undefined
  totalTakes: number
  currentTakeIdx: number
  displayTakeNum: number
  filePath: string
  isUpscaled: boolean
}

export interface ClipCapabilities {
  isVideo: boolean
  isImage: boolean
  isAudio: boolean
  isAdjustment: boolean
  isText: boolean
  canCreateVideoFromImage: boolean
  canCreateVideoFromAudio: boolean
  canRegenerate: boolean
  canRetake: boolean
  canUseIcLora: boolean
}

export interface SelectedClipPropertiesModel {
  clip: TimelineClip
  metadata: ClipMetadata
  capabilities: ClipCapabilities
}

export interface SelectedSubtitleEditorModel {
  subtitle: SubtitleClip
  track: Track | undefined
  effectiveStyle: Partial<SubtitleStyle>
}

export interface SubtitleTrackStyleEditorModel {
  trackIndex: number
  track: Track
  style: SubtitleStyle
}

export interface KeyboardCommandContext {
  clips: TimelineClip[]
  selectedClipIds: Set<string>
  totalDuration: number
  currentTime: number
  inPoint: number | null
  outPoint: number | null
}

export interface MenuState {
  selectedClip: TimelineClip | null
  selectedClipIds: Set<string>
  clips: TimelineClip[]
  tracks: Track[]
  subtitles: SubtitleClip[]
  snapEnabled: boolean
  showEffectsBrowser: boolean
  showSourceMonitor: boolean
  showPropertiesPanel: boolean
  hasSourceAsset: boolean
  activeTool: ToolType
  activeTimeline: Timeline | null
  timelines: Timeline[]
}

export interface AssetListFilters {
  assetFilter?: Asset['type'] | 'all'
  selectedBin?: string | null
  assetViewMode?: 'grid' | 'list'
  listSortCol?: 'name' | 'type' | 'duration' | 'resolution' | 'date' | 'color'
  listSortDir?: 'asc' | 'desc'
}

export interface AssetTakeView {
  asset: Asset | undefined
  takes: AssetTake[]
  activeTakeIndex: number
}

export const EMPTY_IN_OUT_RANGE: TimelineInOutRange = {
  inPoint: null,
  outPoint: null,
}

export function createInitialEditorState(
  editorModel: EditorModel,
  layout: EditorLayout = DEFAULT_LAYOUT,
): EditorState {
  return {
    editorModel,
    session: {
      selection: {
        clipIds: new Set(),
        subtitleId: null,
        editingSubtitleId: null,
        gap: null,
      },
      transport: {
        currentTime: 0,
        isPlaying: false,
        shuttleSpeed: 0,
        playingInOut: false,
        timelineInOutMap: {},
      },
      tools: {
        zoom: 1,
        snapEnabled: true,
        activeTool: 'select',
        lastTrimTool: 'ripple',
      },
      ui: {
        showImportTimelineModal: false,
        showExportModal: false,
        showSourceMonitor: false,
        showPropertiesPanel: false,
        showEffectsBrowser: false,
        activeFocusArea: 'timeline',
        sourceSplitPercent: 50,
        hasSourceAsset: false,
        openTimelineIds: editorModel.activeTimelineId ? new Set([editorModel.activeTimelineId]) : new Set(),
        renamingTimelineId: null,
        renameValue: '',
        renameSource: 'tab',
        layout,
        subtitleTrackStyleIdx: null,
        gapGenerateMode: null,
      },
      regeneration: {
        regeneratingAssetId: null,
        regeneratingClipId: null,
        preError: null,
      },
      clipboard: {
        kind: null,
        clips: [],
        copiedFromTimelineId: null,
      },
    },
    history: {
      undoStack: [],
      redoStack: [],
    },
    projectSync: {
      dirty: false,
    },
  }
}

export function getUndoSnapshot(state: EditorState): EditorUndoSnapshot {
  return {
    assets: state.editorModel.assets,
    timelines: state.editorModel.timelines,
  }
}

export function applyUndoSnapshot(
  state: EditorState,
  snapshot: EditorUndoSnapshot,
): EditorState {
  return {
    ...state,
    editorModel: {
      ...state.editorModel,
      assets: snapshot.assets,
      timelines: snapshot.timelines,
    },
  }
}

export function equalUndoSnapshot(
  left: EditorUndoSnapshot,
  right: EditorUndoSnapshot,
): boolean {
  return left.assets === right.assets && left.timelines === right.timelines
}
