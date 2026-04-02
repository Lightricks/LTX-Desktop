import type { SetStateAction } from 'react'
import type { ParsedTimeline } from '../../lib/timeline-import'
import type { SrtCue } from '../../lib/srt'
import type {
  Asset,
  AssetTake,
  ColorCorrection,
  LetterboxSettings,
  Project,
  SubtitleClip,
  SubtitleStyle,
  TextOverlayStyle,
  Timeline,
  TimelineClip,
  Track,
} from '../../types/project'
import {
  createDefaultTimeline,
  DEFAULT_COLOR_CORRECTION,
  DEFAULT_LETTERBOX,
  DEFAULT_TEXT_STYLE,
} from '../../types/project'
import { resolveOverlaps, type EditorLayout, type ToolType } from './video-editor-utils'
import {
  applyUndoSnapshot,
  createInitialEditorState,
  equalUndoSnapshot,
  getUndoSnapshot,
  type EditorModel,
  type EditorState,
  type TimelineGapSelection,
} from './editor-state'
import {
  getActiveTimelineFromEditorModel,
  selectActiveTimeline,
  selectActiveTimelineInPoint,
  selectActiveTimelineOutPoint,
  selectActiveTimelineId,
  selectAssetById,
  selectCanUseClipboard,
  selectClipById,
  selectClips,
  selectCurrentTime,
  selectLiveAssetForClip,
  selectSelectedClipIds,
  selectTracks,
} from './editor-selectors'
import { getEditorModel, updatedProject } from './editor-project-bridging'

export interface PendingClipTakeUpdate {
  assetId: string
  clipIds: string[]
  newTakeIndex: number
}

export interface InsertAssetsToTimelineParams {
  assets: Asset[]
  trackIndex?: number
  startTime?: number
}

export interface AddTextClipParams {
  style?: Partial<TextOverlayStyle>
  startTime?: number
  trackIndex?: number
}

export interface AddAdjustmentLayerParams {
  duration?: number
  trackIndex?: number
  startTime?: number
}

export interface MoveClipsParams {
  clipIds: string[]
  deltaTime?: number
  targetTrackIndex?: number
}

export interface ResizeClipParams {
  clipId: string
  edge: 'start' | 'end'
  deltaTime: number
}

export interface SlipClipParams {
  clipId: string
  deltaTime: number
}

export interface SlideClipParams {
  clipId: string
  deltaTime: number
}

export interface AddSubtitleParams {
  trackIndex: number
  text?: string
  startTime?: number
  endTime?: number
}

export interface InsertGeneratedGapAssetParams {
  gap: TimelineGapSelection
  asset: Asset
  createAudio: boolean
}

export interface SourceEditParams {
  asset: Asset
  sourceIn: number | null
  sourceOut: number | null
  sourceTime: number
}

export interface SelectClipMode {
  mode?: 'replace' | 'toggle' | 'add'
}

function makeId(prefix: string): string {
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 11)}`
}

function markEditorModelDirty(state: EditorState): EditorState {
  if (state.projectSync.dirty) return state
  return {
    ...state,
    projectSync: {
      ...state.projectSync,
      dirty: true,
    },
  }
}

function cloneClipIds(ids: Set<string>): Set<string> {
  return new Set(ids)
}

export function applyStateAction<T>(value: SetStateAction<T>, current: T): T {
  return typeof value === 'function'
    ? (value as (prevState: T) => T)(current)
    : value
}

function updateEditorModel(state: EditorState, updater: (editorModel: EditorModel) => EditorModel): EditorState {
  const nextEditorModel = updater(state.editorModel)
  if (nextEditorModel === state.editorModel) return state
  return markEditorModelDirty({
    ...state,
    editorModel: nextEditorModel,
  })
}

function updateSession(state: EditorState, updater: (session: EditorState['session']) => EditorState['session']): EditorState {
  const nextSession = updater(state.session)
  if (nextSession === state.session) return state
  return {
    ...state,
    session: nextSession,
  }
}

function withActiveTimeline(editorModel: EditorModel, updater: (timeline: Timeline) => Timeline): EditorModel {
  const activeTimeline = getActiveTimelineFromEditorModel(editorModel)
  if (!activeTimeline) return editorModel
  return {
    ...editorModel,
    timelines: editorModel.timelines.map(timeline => (
      timeline.id === activeTimeline.id ? updater(timeline) : timeline
    )),
  }
}

function activeTrackStartTime(state: EditorState, trackIndex: number): number {
  return selectClips(state)
    .filter(clip => clip.trackIndex === trackIndex)
    .reduce((max, clip) => Math.max(max, clip.startTime + clip.duration), 0)
}

function createTimelineClipFromAsset(asset: Asset, trackIndex: number, startTime: number): TimelineClip {
  return {
    id: makeId('clip'),
    assetId: asset.id,
    type: asset.type === 'adjustment' ? 'adjustment' : asset.type,
    startTime,
    duration: asset.duration || (asset.type === 'adjustment' ? 10 : 5),
    trimStart: 0,
    trimEnd: 0,
    speed: 1,
    reversed: false,
    muted: false,
    volume: 1,
    trackIndex,
    asset,
    flipH: false,
    flipV: false,
    transitionIn: { type: 'none', duration: 0 },
    transitionOut: { type: 'none', duration: 0 },
    colorCorrection: { ...DEFAULT_COLOR_CORRECTION },
    opacity: 100,
  }
}

function buildDroppedAssetInsertion(
  asset: Asset,
  trackIndex: number,
  startTime: number,
  tracks: Track[],
): {
  tracks: Track[]
  clips: TimelineClip[]
  duration: number
} {
  const track = tracks[trackIndex]
  if (!track || track.locked) {
    return { tracks, clips: [], duration: 0 }
  }

  const trackPatched = track.sourcePatched !== false
  const isAdjustment = asset.type === 'adjustment'
  const isVideoAsset = asset.type === 'video'
  const isAudioAsset = asset.type === 'audio'
  const isImageAsset = asset.type === 'image'

  if (isAudioAsset && !trackPatched) {
    return { tracks, clips: [], duration: 0 }
  }

  const createVideoClip = (isVideoAsset || isImageAsset || isAdjustment) && trackPatched
  const needsAudioClip = isVideoAsset && !isAdjustment
  let nextTracks = tracks
  let audioTrackIndex = -1

  if (needsAudioClip) {
    audioTrackIndex = nextTracks.findIndex(
      (candidate, index) =>
        index > trackIndex &&
        candidate.kind === 'audio' &&
        !candidate.locked &&
        candidate.sourcePatched !== false,
    )
    if (audioTrackIndex < 0) {
      audioTrackIndex = nextTracks.findIndex(
        candidate => candidate.kind === 'audio' && !candidate.locked && candidate.sourcePatched !== false,
      )
    }
    if (audioTrackIndex < 0) {
      const audioTrackCount = nextTracks.filter(candidate => candidate.kind === 'audio').length
      nextTracks = [
        ...nextTracks,
        {
          id: makeId('track-audio'),
          name: `A${audioTrackCount + 1}`,
          muted: false,
          locked: false,
          kind: 'audio',
        },
      ]
      audioTrackIndex = nextTracks.length - 1
    }
  }

  const createAudioClip = needsAudioClip && audioTrackIndex >= 0
  if (!createVideoClip && !createAudioClip) {
    return { tracks: nextTracks, clips: [], duration: 0 }
  }

  const duration = asset.duration || (isAdjustment ? 10 : 5)
  const videoClipId = makeId('clip')
  const audioClipId = makeId('clip-audio')
  const clips: TimelineClip[] = []

  if (createVideoClip) {
    clips.push({
      id: videoClipId,
      assetId: asset.id,
      type: isAdjustment ? 'adjustment' : isVideoAsset ? 'video' : isAudioAsset ? 'audio' : 'image',
      startTime,
      duration,
      trimStart: 0,
      trimEnd: 0,
      speed: 1,
      reversed: false,
      muted: false,
      volume: 1,
      trackIndex,
      asset,
      flipH: false,
      flipV: false,
      transitionIn: { type: 'none', duration: isAdjustment ? 0 : 0.5 },
      transitionOut: { type: 'none', duration: isAdjustment ? 0 : 0.5 },
      colorCorrection: { ...DEFAULT_COLOR_CORRECTION },
      opacity: 100,
      ...(isAdjustment ? { letterbox: { ...DEFAULT_LETTERBOX } } : {}),
      ...(createAudioClip ? { linkedClipIds: [audioClipId] } : {}),
    })
  }

  if (createAudioClip && audioTrackIndex >= 0) {
    clips.push({
      id: audioClipId,
      assetId: asset.id,
      type: 'audio',
      startTime,
      duration,
      trimStart: 0,
      trimEnd: 0,
      speed: 1,
      reversed: false,
      muted: false,
      volume: 1,
      trackIndex: audioTrackIndex,
      asset,
      flipH: false,
      flipV: false,
      transitionIn: { type: 'none', duration: 0.5 },
      transitionOut: { type: 'none', duration: 0.5 },
      colorCorrection: { ...DEFAULT_COLOR_CORRECTION },
      opacity: 100,
      ...(createVideoClip ? { linkedClipIds: [videoClipId] } : {}),
    })
  }

  return {
    tracks: nextTracks,
    clips,
    duration,
  }
}

function createTextClip(style?: Partial<TextOverlayStyle>, startTime = 0, trackIndex = 0): TimelineClip {
  return {
    id: makeId('clip-text'),
    assetId: null,
    type: 'text',
    startTime,
    duration: 5,
    trimStart: 0,
    trimEnd: 0,
    speed: 1,
    reversed: false,
    muted: true,
    volume: 1,
    trackIndex,
    asset: null,
    flipH: false,
    flipV: false,
    transitionIn: { type: 'none', duration: 0 },
    transitionOut: { type: 'none', duration: 0 },
    colorCorrection: { ...DEFAULT_COLOR_CORRECTION },
    opacity: 100,
    textStyle: {
      ...DEFAULT_TEXT_STYLE,
      ...(style || {}),
    },
  }
}

function getNextAdjustmentLayerName(assets: Asset[]): string {
  const count = assets.filter(asset => asset.type === 'adjustment').length
  return count > 0 ? `Adjustment Layer ${count + 1}` : 'Adjustment Layer'
}

function createAdjustmentAsset(name = 'Adjustment Layer'): Asset {
  return {
    id: makeId('asset-adjustment'),
    type: 'adjustment',
    path: '',
    prompt: name,
    resolution: '',
    duration: 10,
    createdAt: Date.now(),
  }
}

function createAdjustmentClip(asset: Asset, startTime = 0, trackIndex = 0, duration = 10): TimelineClip {
  return {
    ...createTimelineClipFromAsset(asset, trackIndex, startTime),
    type: 'adjustment',
    duration,
    letterbox: { ...DEFAULT_LETTERBOX },
  }
}

function buildSourceRequestClips(state: EditorState, params: SourceEditParams): {
  newClips: TimelineClip[]
  insertDuration: number
  targetTrackIndices: number[]
  time: number
} | null {
  const { asset } = params
  const sourceIn = params.sourceIn ?? 0
  const sourceDuration = asset.duration || 5
  const sourceOut = params.sourceOut ?? sourceDuration
  const insertDuration = sourceOut - sourceIn
  if (insertDuration <= 0) return null

  const time = selectCurrentTime(state)
  const tracks = selectTracks(state)
  const isAudio = asset.type === 'audio'
  const videoTrack = !isAudio
    ? tracks.find(track => !track.locked && track.sourcePatched !== false && track.kind === 'video')
    : undefined
  const audioTrack = tracks.find(track => !track.locked && track.sourcePatched !== false && track.kind === 'audio')

  if (!videoTrack && !audioTrack) return null
  if (isAudio && !audioTrack) return null
  if (!isAudio && !videoTrack) return null

  const videoTrackIndex = videoTrack ? tracks.indexOf(videoTrack) : -1
  const audioTrackIndex = audioTrack ? tracks.indexOf(audioTrack) : -1
  const videoClipId = makeId('clip')
  const audioClipId = makeId('clip-audio')

  const baseClip = {
    assetId: asset.id,
    startTime: time,
    duration: insertDuration,
    trimStart: sourceIn,
    trimEnd: sourceDuration - sourceOut,
    speed: 1,
    reversed: false,
    muted: false,
    volume: 1,
    asset,
    flipH: false as const,
    flipV: false as const,
    transitionIn: { type: 'none' as const, duration: 0.5 },
    transitionOut: { type: 'none' as const, duration: 0.5 },
    colorCorrection: { ...DEFAULT_COLOR_CORRECTION },
    opacity: 100,
  }

  const newClips: TimelineClip[] = []
  const targetTrackIndices: number[] = []

  if (isAudio) {
    newClips.push({
      ...baseClip,
      id: audioClipId,
      type: 'audio',
      trackIndex: audioTrackIndex,
    })
    targetTrackIndices.push(audioTrackIndex)
  } else {
    const needsAudio = asset.type === 'video' && audioTrackIndex >= 0
    newClips.push({
      ...baseClip,
      id: videoClipId,
      type: asset.type === 'video' ? 'video' : 'image',
      trackIndex: videoTrackIndex,
      ...(needsAudio ? { linkedClipIds: [audioClipId] } : {}),
    })
    targetTrackIndices.push(videoTrackIndex)
    if (needsAudio) {
      newClips.push({
        ...baseClip,
        id: audioClipId,
        type: 'audio',
        trackIndex: audioTrackIndex,
        linkedClipIds: [videoClipId],
      })
      targetTrackIndices.push(audioTrackIndex)
    }
  }

  return { newClips, insertDuration, targetTrackIndices, time }
}

function replaceActiveTimeline(state: EditorState, updater: (timeline: Timeline) => Timeline): EditorState {
  return updateEditorModel(state, editorModel => withActiveTimeline(editorModel, updater))
}

function mapClips(state: EditorState, mapper: (clip: TimelineClip) => TimelineClip): EditorState {
  return replaceActiveTimeline(state, timeline => ({
    ...timeline,
    clips: timeline.clips.map(mapper),
  }))
}

function mapTracks(state: EditorState, mapper: (track: Track, index: number) => Track): EditorState {
  return replaceActiveTimeline(state, timeline => ({
    ...timeline,
    tracks: timeline.tracks.map(mapper),
  }))
}

export function loadEditorDocument(state: EditorState, snapshot: EditorModel): EditorState {
  return markEditorModelDirty({
    ...state,
    editorModel: snapshot,
  })
}

export function replaceActiveTimelineDocument(
  state: EditorState,
  snapshot: Partial<Pick<Timeline, 'tracks' | 'clips' | 'subtitles'>>,
): EditorState {
  return replaceActiveTimeline(state, timeline => ({
    ...timeline,
    ...snapshot,
  }))
}

export function setTimelineClips(state: EditorState, value: SetStateAction<TimelineClip[]>): EditorState {
  return replaceActiveTimeline(state, timeline => ({
    ...timeline,
    clips: applyStateAction(value, timeline.clips),
  }))
}

export function setTimelineTracks(state: EditorState, value: SetStateAction<Track[]>): EditorState {
  return replaceActiveTimeline(state, timeline => ({
    ...timeline,
    tracks: applyStateAction(value, timeline.tracks),
  }))
}

export function setTimelineSubtitles(state: EditorState, value: SetStateAction<SubtitleClip[]>): EditorState {
  return replaceActiveTimeline(state, timeline => ({
    ...timeline,
    subtitles: applyStateAction(value, timeline.subtitles || []),
  }))
}

export function commitEditorDocument(state: EditorState): EditorState {
  return state
}

export function switchActiveTimeline(state: EditorState, timelineId: string | null): EditorState {
  return {
    ...markEditorModelDirty({
      ...state,
      editorModel: {
        ...state.editorModel,
        activeTimelineId: timelineId,
      },
    }),
    session: {
      ...state.session,
      selection: {
        ...state.session.selection,
        clipIds: new Set(),
        subtitleId: null,
      },
      transport: {
        ...state.session.transport,
        currentTime: 0,
        isPlaying: false,
        playingInOut: false,
      },
      ui: {
        ...state.session.ui,
        openTimelineIds: timelineId
          ? new Set([...state.session.ui.openTimelineIds, timelineId])
          : new Set(state.session.ui.openTimelineIds),
      },
    },
  }
}

export function createTimeline(state: EditorState, name?: string): EditorState {
  const timeline = createDefaultTimeline(name)
  return {
    ...updateEditorModel(state, editorModel => ({
      ...editorModel,
      timelines: [...editorModel.timelines, timeline],
      activeTimelineId: timeline.id,
    })),
    session: {
      ...state.session,
      ui: {
        ...state.session.ui,
        openTimelineIds: new Set([...state.session.ui.openTimelineIds, timeline.id]),
      },
      selection: {
        ...state.session.selection,
        clipIds: new Set(),
        subtitleId: null,
      },
      transport: {
        ...state.session.transport,
        currentTime: 0,
        isPlaying: false,
        playingInOut: false,
      },
    },
  }
}

export function duplicateTimeline(state: EditorState, timelineId: string): EditorState {
  const source = state.editorModel.timelines.find(timeline => timeline.id === timelineId)
  if (!source) return state
  const duplicate: Timeline = {
    ...source,
    id: makeId('timeline'),
    name: `${source.name} Copy`,
    createdAt: Date.now(),
    tracks: source.tracks.map(track => ({ ...track })),
    clips: source.clips.map(clip => ({ ...clip, id: makeId('clip') })),
    subtitles: source.subtitles?.map(subtitle => ({ ...subtitle, id: makeId('sub') })),
  }
  return {
    ...updateEditorModel(state, editorModel => ({
      ...editorModel,
      timelines: [...editorModel.timelines, duplicate],
      activeTimelineId: duplicate.id,
    })),
    session: {
      ...state.session,
      ui: {
        ...state.session.ui,
        openTimelineIds: new Set([...state.session.ui.openTimelineIds, duplicate.id]),
      },
      selection: {
        ...state.session.selection,
        clipIds: new Set(),
        subtitleId: null,
      },
      transport: {
        ...state.session.transport,
        currentTime: 0,
        isPlaying: false,
        playingInOut: false,
      },
    },
  }
}

export function renameTimeline(state: EditorState, timelineId: string, name: string): EditorState {
  return updateEditorModel(state, editorModel => ({
    ...editorModel,
    timelines: editorModel.timelines.map(timeline => (
      timeline.id === timelineId ? { ...timeline, name } : timeline
    )),
  }))
}

export function deleteTimeline(state: EditorState, timelineId: string): EditorState {
  const nextTimelines = state.editorModel.timelines.filter(timeline => timeline.id !== timelineId)
  const nextActiveTimelineId = state.editorModel.activeTimelineId === timelineId
    ? nextTimelines[0]?.id ?? null
    : state.editorModel.activeTimelineId

  return {
    ...markEditorModelDirty({
      ...state,
      editorModel: {
        ...state.editorModel,
        timelines: nextTimelines,
        activeTimelineId: nextActiveTimelineId,
      },
    }),
    session: {
      ...state.session,
      ui: {
        ...state.session.ui,
        openTimelineIds: new Set([...state.session.ui.openTimelineIds].filter(id => nextTimelines.some(timeline => timeline.id === id))),
      },
    },
  }
}

export function importParsedTimeline(state: EditorState, parsed: ParsedTimeline): EditorState {
  const importedAssets: Asset[] = parsed.mediaRefs.map(ref => ({
    id: makeId('asset'),
    type: ref.type,
    path: ref.path,
    bigThumbnailPath: ref.bigThumbnailPath,
    smallThumbnailPath: ref.smallThumbnailPath,
    width: ref.width,
    height: ref.height,
    prompt: ref.name || ref.path.split(/[/\\]/).pop() || 'Imported media',
    resolution: ref.width && ref.height ? `${ref.width}x${ref.height}` : 'Unknown',
    duration: ref.duration || undefined,
    bin: 'Imported',
    createdAt: Date.now(),
  }))

  const assetByParsedId = new Map<string, Asset>()
  parsed.mediaRefs.forEach((ref, index) => {
    assetByParsedId.set(ref.id, importedAssets[index])
  })

  const totalTracks = Math.max(parsed.videoTrackCount + parsed.audioTrackCount, 1)
  const tracks: Track[] = []
  for (let index = 0; index < totalTracks; index++) {
    const isAudio = index >= parsed.videoTrackCount
    tracks.push({
      id: makeId('track'),
      name: isAudio ? `A${index - parsed.videoTrackCount + 1}` : `V${index + 1}`,
      muted: false,
      locked: false,
      kind: isAudio ? 'audio' : 'video',
    })
  }

  const clips: TimelineClip[] = []
  const parsedIndexToClipId = new Map<number, string>()
  for (let parsedIndex = 0; parsedIndex < parsed.clips.length; parsedIndex++) {
    const parsedClip = parsed.clips[parsedIndex]
    const asset = assetByParsedId.get(parsedClip.mediaRefId)
    if (!asset) continue

    const clipId = makeId('clip')
    parsedIndexToClipId.set(parsedIndex, clipId)
    clips.push({
      id: clipId,
      assetId: asset.id,
      type: parsedClip.trackType === 'audio' ? 'audio' : asset.type === 'image' ? 'image' : 'video',
      startTime: parsedClip.startTime,
      duration: parsedClip.duration,
      trimStart: parsedClip.sourceIn || 0,
      trimEnd: 0,
      speed: parsedClip.speed || 1,
      reversed: parsedClip.reversed || false,
      muted: parsedClip.muted || false,
      volume: parsedClip.volume !== undefined ? Math.min(1, Math.max(0, parsedClip.volume)) : 1,
      trackIndex: Math.min(parsedClip.trackIndex, totalTracks - 1),
      asset,
      importedName: parsedClip.name,
      flipH: parsedClip.flipH || false,
      flipV: parsedClip.flipV || false,
      transitionIn: { type: 'none', duration: 0 },
      transitionOut: { type: 'none', duration: 0 },
      colorCorrection: { ...DEFAULT_COLOR_CORRECTION },
      opacity: parsedClip.opacity !== undefined ? parsedClip.opacity : 100,
    })
  }

  for (let parsedIndex = 0; parsedIndex < parsed.clips.length; parsedIndex++) {
    const parsedClip = parsed.clips[parsedIndex]
    if (parsedClip.linkedVideoClipIndex === undefined) continue
    const audioClipId = parsedIndexToClipId.get(parsedIndex)
    const videoClipId = parsedIndexToClipId.get(parsedClip.linkedVideoClipIndex)
    if (!audioClipId || !videoClipId) continue
    const audioClip = clips.find(clip => clip.id === audioClipId)
    const videoClip = clips.find(clip => clip.id === videoClipId)
    if (!audioClip || !videoClip) continue
    if (!audioClip.linkedClipIds) audioClip.linkedClipIds = []
    if (!audioClip.linkedClipIds.includes(videoClipId)) audioClip.linkedClipIds.push(videoClipId)
    if (!videoClip.linkedClipIds) videoClip.linkedClipIds = []
    if (!videoClip.linkedClipIds.includes(audioClipId)) videoClip.linkedClipIds.push(audioClipId)
  }

  const timeline: Timeline = {
    id: makeId('timeline'),
    name: parsed.name || 'Imported Timeline',
    createdAt: Date.now(),
    tracks,
    clips,
    subtitles: [],
  }

  return {
    ...updateEditorModel(state, editorModel => ({
      ...editorModel,
      assets: [...importedAssets, ...editorModel.assets],
      timelines: [...editorModel.timelines, timeline],
      activeTimelineId: timeline.id,
    })),
    session: {
      ...state.session,
      selection: {
        ...state.session.selection,
        clipIds: new Set(),
        subtitleId: null,
      },
      transport: {
        ...state.session.transport,
        currentTime: 0,
        isPlaying: false,
        playingInOut: false,
      },
      ui: {
        ...state.session.ui,
        openTimelineIds: new Set([...state.session.ui.openTimelineIds, timeline.id]),
      },
    },
  }
}

export function insertAssetsToTimeline(state: EditorState, params: InsertAssetsToTimelineParams): EditorState {
  const trackIndex = params.trackIndex ?? 0
  const activeTimeline = selectActiveTimeline(state)
  if (!activeTimeline) return state

  let cursor = params.startTime ?? activeTrackStartTime(state, trackIndex)
  let nextTracks = activeTimeline.tracks
  const insertedClips: TimelineClip[] = []

  for (const asset of params.assets) {
    const insertion = buildDroppedAssetInsertion(asset, trackIndex, cursor, nextTracks)
    nextTracks = insertion.tracks
    insertedClips.push(...insertion.clips)
    cursor += insertion.duration
  }

  if (insertedClips.length === 0 && nextTracks === activeTimeline.tracks) {
    return state
  }

  const insertedIds = new Set(insertedClips.map(clip => clip.id))
  return replaceActiveTimeline(state, timeline => ({
    ...timeline,
    tracks: nextTracks,
    clips: resolveOverlaps([...timeline.clips, ...insertedClips], insertedIds),
  }))
}

export function overwriteAssetsOnTimeline(state: EditorState, params: InsertAssetsToTimelineParams): EditorState {
  const afterInsert = insertAssetsToTimeline(state, params)
  const activeTimeline = selectActiveTimeline(afterInsert)
  if (!activeTimeline) return afterInsert
  const insertedIds = new Set<string>(
    activeTimeline.clips
      .slice(-params.assets.length * 2)
      .map((clip: TimelineClip) => clip.id),
  )
  return replaceActiveTimeline(afterInsert, timeline => ({
    ...timeline,
    clips: resolveOverlaps(timeline.clips, insertedIds),
  }))
}

export function insertSourceEdit(state: EditorState, params: SourceEditParams): EditorState {
  const result = buildSourceRequestClips(state, params)
  if (!result) return state
  return replaceActiveTimeline(state, timeline => ({
    ...timeline,
    clips: [
      ...timeline.clips.map(clip => (
        result.targetTrackIndices.includes(clip.trackIndex) && clip.startTime >= result.time
          ? { ...clip, startTime: clip.startTime + result.insertDuration }
          : clip
      )),
      ...result.newClips,
    ],
  }))
}

export function overwriteSourceEdit(state: EditorState, params: SourceEditParams): EditorState {
  const result = buildSourceRequestClips(state, params)
  if (!result) return state
  const insertedIds = new Set(result.newClips.map(clip => clip.id))
  return replaceActiveTimeline(state, timeline => ({
    ...timeline,
    clips: resolveOverlaps([...timeline.clips, ...result.newClips], insertedIds),
  }))
}

export function addTextClip(state: EditorState, params: AddTextClipParams = {}): EditorState {
  const insertTime = params.startTime ?? selectCurrentTime(state)
  const tracks = selectTracks(state)
  const videoTrackIndices = tracks
    .map((track, index) => ({ track, index }))
    .filter(({ track }) => track.kind === 'video' && track.type !== 'subtitle')
    .map(({ index }) => index)
  const targetTrack = params.trackIndex ?? (
    videoTrackIndices.length > 0 ? videoTrackIndices[videoTrackIndices.length - 1] : 0
  )
  const clip = createTextClip(params.style, insertTime, targetTrack)

  let next = replaceActiveTimeline(state, timeline => ({
    ...timeline,
    clips: resolveOverlaps([...timeline.clips, clip], new Set([clip.id])),
  }))
  next = setSelectedClipIds(next, new Set([clip.id]))
  next = setCurrentTime(next, insertTime + 0.1)
  return next
}

export function createAdjustmentLayerAsset(state: EditorState): EditorState {
  const name = getNextAdjustmentLayerName(state.editorModel.assets)
  const asset = createAdjustmentAsset(name)
  return updateEditorModel(state, editorModel => ({
    ...editorModel,
    assets: [asset, ...editorModel.assets],
  }))
}

export function addAdjustmentLayer(state: EditorState, params: AddAdjustmentLayerParams = {}): EditorState {
  const name = getNextAdjustmentLayerName(state.editorModel.assets)
  const asset = createAdjustmentAsset(name)
  const clip = createAdjustmentClip(asset, params.startTime ?? selectCurrentTime(state), params.trackIndex ?? 0, params.duration ?? 10)
  return updateEditorModel(replaceActiveTimeline(state, timeline => ({
    ...timeline,
    clips: [...timeline.clips, clip],
  })), editorModel => ({
    ...editorModel,
    assets: [asset, ...editorModel.assets],
  }))
}

export function duplicateClips(state: EditorState, clipIds: string[]): EditorState {
  const clipSet = new Set(clipIds)
  return replaceActiveTimeline(state, timeline => {
    const duplicates = timeline.clips
      .filter(clip => clipSet.has(clip.id))
      .map(clip => ({
        ...clip,
        id: makeId('clip'),
        startTime: clip.startTime + clip.duration,
      }))
    return {
      ...timeline,
      clips: [...timeline.clips, ...duplicates],
    }
  })
}

export function unlinkClipGroup(state: EditorState, clipId: string): EditorState {
  const clip = selectClips(state).find(candidate => candidate.id === clipId)
  if (!clip?.linkedClipIds?.length) return state
  const linkedIds = new Set(clip.linkedClipIds)
  return replaceActiveTimeline(state, timeline => ({
    ...timeline,
    clips: timeline.clips.map(candidate => {
      if (candidate.id === clipId) {
        return { ...candidate, linkedClipIds: undefined }
      }
      if (!linkedIds.has(candidate.id) || !candidate.linkedClipIds?.length) {
        return candidate
      }
      const remaining = candidate.linkedClipIds.filter(id => id !== clipId)
      return { ...candidate, linkedClipIds: remaining.length > 0 ? remaining : undefined }
    }),
  }))
}

export function deleteClips(state: EditorState, clipIds: string[]): EditorState {
  const deleteSet = new Set(clipIds)
  let next = replaceActiveTimeline(state, timeline => ({
    ...timeline,
    clips: timeline.clips
      .filter(clip => !deleteSet.has(clip.id))
      .map(clip => {
        if (!clip.linkedClipIds) return clip
        const remaining = clip.linkedClipIds.filter(id => !deleteSet.has(id))
        return { ...clip, linkedClipIds: remaining.length > 0 ? remaining : undefined }
      }),
  }))
  next = updateSession(next, session => ({
    ...session,
    selection: {
      ...session.selection,
      clipIds: new Set([...session.selection.clipIds].filter(id => !deleteSet.has(id))),
    },
  }))
  return next
}

export function splitClipsAtTime(state: EditorState, clipIds: string[], time: number): EditorState {
  const clips = selectClips(state)
  const tracks = selectTracks(state)
  const splittable = clipIds.filter(id => {
    const clip = clips.find(candidate => candidate.id === id)
    if (!clip) return false
    if (tracks[clip.trackIndex]?.locked) return false
    const splitPoint = time - clip.startTime
    return splitPoint > 0.1 && splitPoint < clip.duration - 0.1
  })
  if (splittable.length === 0) return state

  return replaceActiveTimeline(state, timeline => {
    const alreadySplit = new Set<string>()
    let newClips = [...timeline.clips]

    for (const splitId of splittable) {
      if (alreadySplit.has(splitId)) continue

      const clip = newClips.find(candidate => candidate.id === splitId)
      if (!clip) continue

      const splitPoint = time - clip.startTime
      if (splitPoint <= 0.1 || splitPoint >= clip.duration - 0.1) continue

      alreadySplit.add(splitId)

      const firstHalfId = clip.id
      const secondHalfId = makeId('clip')
      const linkedClips = (clip.linkedClipIds || [])
        .map(linkedId => newClips.find(candidate => candidate.id === linkedId))
        .filter((linkedClip): linkedClip is TimelineClip => linkedClip != null)

      const firstHalf: TimelineClip = {
        ...clip,
        duration: splitPoint,
        trimEnd: clip.trimEnd + (clip.duration - splitPoint),
      }
      const secondHalf: TimelineClip = {
        ...clip,
        id: secondHalfId,
        startTime: clip.startTime + splitPoint,
        duration: clip.duration - splitPoint,
        trimStart: clip.trimStart + splitPoint,
      }

      newClips = newClips.map(candidate => candidate.id === splitId ? firstHalf : candidate).concat(secondHalf)

      const firstHalfLinkedIds: string[] = []
      const secondHalfLinkedIds: string[] = []

      for (const linkedClip of linkedClips) {
        alreadySplit.add(linkedClip.id)

        const linkedSplitPoint = time - linkedClip.startTime
        if (linkedSplitPoint <= 0.01 || linkedSplitPoint >= linkedClip.duration - 0.01) {
          firstHalfLinkedIds.push(linkedClip.id)
          continue
        }

        const linkedSecondId = makeId('clip')
        firstHalfLinkedIds.push(linkedClip.id)
        secondHalfLinkedIds.push(linkedSecondId)

        const linkedFirstHalf: TimelineClip = {
          ...linkedClip,
          duration: linkedSplitPoint,
          trimEnd: linkedClip.trimEnd + (linkedClip.duration - linkedSplitPoint),
          linkedClipIds: [firstHalfId],
        }
        const linkedSecondHalf: TimelineClip = {
          ...linkedClip,
          id: linkedSecondId,
          startTime: linkedClip.startTime + linkedSplitPoint,
          duration: linkedClip.duration - linkedSplitPoint,
          trimStart: linkedClip.trimStart + linkedSplitPoint,
          linkedClipIds: [secondHalfId],
        }

        newClips = newClips
          .map(candidate => candidate.id === linkedClip.id ? linkedFirstHalf : candidate)
          .concat(linkedSecondHalf)
      }

      firstHalf.linkedClipIds = firstHalfLinkedIds.length > 0 ? firstHalfLinkedIds : undefined
      secondHalf.linkedClipIds = secondHalfLinkedIds.length > 0 ? secondHalfLinkedIds : undefined
      newClips = newClips.map(candidate => (
        candidate.id === firstHalfId ? firstHalf : candidate.id === secondHalfId ? secondHalf : candidate
      ))
    }

    return {
      ...timeline,
      clips: newClips,
    }
  })
}

export function moveClips(state: EditorState, params: MoveClipsParams): EditorState {
  const clipSet = new Set(params.clipIds)
  return replaceActiveTimeline(state, timeline => ({
    ...timeline,
    clips: timeline.clips.map(clip => (
      clipSet.has(clip.id)
        ? {
            ...clip,
            startTime: Math.max(0, clip.startTime + (params.deltaTime ?? 0)),
            trackIndex: params.targetTrackIndex ?? clip.trackIndex,
          }
        : clip
    )),
  }))
}

export function resizeClip(state: EditorState, params: ResizeClipParams): EditorState {
  return mapClips(state, clip => {
    if (clip.id !== params.clipId) return clip
    if (params.edge === 'start') {
      const nextStart = Math.max(0, clip.startTime + params.deltaTime)
      const delta = nextStart - clip.startTime
      const nextDuration = Math.max(0.1, clip.duration - delta)
      return {
        ...clip,
        startTime: nextStart,
        duration: nextDuration,
        trimStart: Math.max(0, clip.trimStart + delta * clip.speed),
      }
    }
    return {
      ...clip,
      duration: Math.max(0.1, clip.duration + params.deltaTime),
    }
  })
}

export function slipClip(state: EditorState, params: SlipClipParams): EditorState {
  return mapClips(state, clip => {
    if (clip.id !== params.clipId) return clip
    return {
      ...clip,
      trimStart: Math.max(0, clip.trimStart + params.deltaTime * clip.speed),
      trimEnd: Math.max(0, clip.trimEnd - params.deltaTime * clip.speed),
    }
  })
}

export function slideClip(state: EditorState, params: SlideClipParams): EditorState {
  return moveClips(state, { clipIds: [params.clipId], deltaTime: params.deltaTime })
}

export function updateClip(state: EditorState, clipId: string, patch: Partial<TimelineClip>): EditorState {
  return mapClips(state, clip => (clip.id === clipId ? { ...clip, ...patch } : clip))
}

export function setClipStartTime(state: EditorState, clipId: string, startTime: number): EditorState {
  return updateClip(state, clipId, { startTime: Math.max(0, startTime) })
}

export function setClipDuration(state: EditorState, clipId: string, duration: number): EditorState {
  return updateClip(state, clipId, { duration: Math.max(0.1, duration) })
}

export function setClipSpeed(state: EditorState, clipId: string, speed: number): EditorState {
  return updateClip(state, clipId, { speed: Math.max(0.01, speed) })
}

function resolveClipAudioTargetId(state: EditorState, clipId: string): string | null {
  const clip = selectClipById(state, clipId)
  if (!clip) return null
  if (clip.type === 'audio') return clip.id

  const linkedAudioClip = (clip.linkedClipIds || [])
    .map(linkedId => selectClipById(state, linkedId))
    .find(candidate => candidate?.type === 'audio')

  return linkedAudioClip?.id ?? clip.id
}

export function setClipAudioLevel(state: EditorState, clipId: string, volume: number): EditorState {
  const targetClipId = resolveClipAudioTargetId(state, clipId)
  if (!targetClipId) return state
  const clampedVolume = Math.max(0, Math.min(1, volume))
  return updateClip(state, targetClipId, { volume: clampedVolume, muted: false })
}

export function setClipAudioMuted(state: EditorState, clipId: string, muted: boolean): EditorState {
  const targetClipId = resolveClipAudioTargetId(state, clipId)
  if (!targetClipId) return state
  return updateClip(state, targetClipId, { muted })
}

export function setClipVolume(state: EditorState, clipId: string, volume: number): EditorState {
  return updateClip(state, clipId, { volume })
}

export function toggleClipMute(state: EditorState, clipId: string): EditorState {
  const clip = selectClips(state).find(candidate => candidate.id === clipId)
  if (!clip) return state
  return updateClip(state, clipId, { muted: !clip.muted })
}

export function toggleClipReverse(state: EditorState, clipId: string): EditorState {
  const clip = selectClips(state).find(candidate => candidate.id === clipId)
  if (!clip) return state
  return updateClip(state, clipId, { reversed: !clip.reversed })
}

export function setClipOpacity(state: EditorState, clipId: string, opacity: number): EditorState {
  return updateClip(state, clipId, { opacity })
}

export function setClipFlipH(state: EditorState, clipId: string, value: boolean): EditorState {
  return updateClip(state, clipId, { flipH: value })
}

export function setClipFlipV(state: EditorState, clipId: string, value: boolean): EditorState {
  return updateClip(state, clipId, { flipV: value })
}

export function setClipColorLabel(state: EditorState, clipId: string, colorLabel?: string): EditorState {
  return updateClip(state, clipId, { colorLabel })
}

export function setClipTakeIndex(state: EditorState, clipId: string, takeIndex?: number): EditorState {
  return updateClip(state, clipId, { takeIndex })
}

export function stepClipTake(state: EditorState, clipId: string, direction: 'prev' | 'next'): EditorState {
  const clip = selectClips(state).find(candidate => candidate.id === clipId)
  const asset = clip ? selectLiveAssetForClip(state, clip) : null
  if (!clip || !asset?.takes || asset.takes.length <= 1) return state
  const currentTakeIndex = clip.takeIndex ?? asset.activeTakeIndex ?? 0
  const delta = direction === 'prev' ? -1 : 1
  const nextTakeIndex = Math.max(0, Math.min(asset.takes.length - 1, currentTakeIndex + delta))
  if (nextTakeIndex === currentTakeIndex) return state
  return setClipTakeIndex(state, clipId, nextTakeIndex)
}

export function setClipColorCorrectionField<K extends keyof ColorCorrection>(
  state: EditorState,
  clipId: string,
  field: K,
  value: ColorCorrection[K],
): EditorState {
  const clip = selectClips(state).find(candidate => candidate.id === clipId)
  if (!clip) return state
  return updateClip(state, clipId, {
    colorCorrection: {
      ...(clip.colorCorrection || DEFAULT_COLOR_CORRECTION),
      [field]: value,
    },
  })
}

export function resetClipColorCorrection(state: EditorState, clipId: string): EditorState {
  return updateClip(state, clipId, { colorCorrection: { ...DEFAULT_COLOR_CORRECTION } })
}

export function setClipLetterbox(state: EditorState, clipId: string, patch: Partial<LetterboxSettings>): EditorState {
  const clip = selectClips(state).find(candidate => candidate.id === clipId)
  if (!clip) return state
  return updateClip(state, clipId, {
    letterbox: {
      ...(clip.letterbox || DEFAULT_LETTERBOX),
      ...patch,
    },
  })
}

export function setClipTextStyleField<K extends keyof TextOverlayStyle>(
  state: EditorState,
  clipId: string,
  field: K,
  value: TextOverlayStyle[K],
): EditorState {
  const clip = selectClips(state).find(candidate => candidate.id === clipId)
  if (!clip) return state
  return updateClip(state, clipId, {
    textStyle: {
      ...(clip.textStyle || DEFAULT_TEXT_STYLE),
      [field]: value,
    },
  })
}

export function setClipTextPosition(
  state: EditorState,
  clipId: string,
  positionX: number,
  positionY: number,
): EditorState {
  const clip = selectClips(state).find(candidate => candidate.id === clipId)
  if (!clip?.textStyle) return state
  const clamp = (value: number) => Math.max(0, Math.min(100, value))
  const round1 = (value: number) => Math.round(value * 10) / 10
  return updateClip(state, clipId, {
    textStyle: {
      ...clip.textStyle,
      positionX: round1(clamp(positionX)),
      positionY: round1(clamp(positionY)),
    },
  })
}

export function addCrossDissolve(state: EditorState, leftClipId: string, rightClipId: string): EditorState {
  return mapClips(state, clip => {
    if (clip.id === leftClipId) return { ...clip, transitionOut: { type: 'dissolve', duration: 0.5 } }
    if (clip.id === rightClipId) return { ...clip, transitionIn: { type: 'dissolve', duration: 0.5 } }
    return clip
  })
}

export function removeCrossDissolve(state: EditorState, leftClipId: string, rightClipId: string): EditorState {
  return mapClips(state, clip => {
    if (clip.id === leftClipId) return { ...clip, transitionOut: { type: 'none', duration: 0.5 } }
    if (clip.id === rightClipId) return { ...clip, transitionIn: { type: 'none', duration: 0.5 } }
    return clip
  })
}

export function addTrack(state: EditorState, kind: 'video' | 'audio'): EditorState {
  const tracks = selectTracks(state)
  const sameKindCount = tracks.filter(track => track.kind === kind && track.type !== 'subtitle').length
  const newTrack: Track = {
    id: makeId('track'),
    name: kind === 'audio' ? `A${sameKindCount + 1}` : `V${sameKindCount + 1}`,
    muted: false,
    locked: false,
    kind,
  }
  return replaceActiveTimeline(state, timeline => ({
    ...timeline,
    tracks: [...timeline.tracks, newTrack],
  }))
}

export function deleteTrack(state: EditorState, trackId: string): EditorState {
  const tracks = selectTracks(state)
  const trackIndex = tracks.findIndex(track => track.id === trackId)
  if (trackIndex < 0 || tracks.length <= 1) return state
  return replaceActiveTimeline(state, timeline => ({
    ...timeline,
    clips: timeline.clips
      .filter(clip => clip.trackIndex !== trackIndex)
      .map(clip => clip.trackIndex > trackIndex ? { ...clip, trackIndex: clip.trackIndex - 1 } : clip),
    subtitles: (timeline.subtitles || [])
      .filter(subtitle => subtitle.trackIndex !== trackIndex)
      .map(subtitle => subtitle.trackIndex > trackIndex ? { ...subtitle, trackIndex: subtitle.trackIndex - 1 } : subtitle),
    tracks: timeline.tracks.filter((_, index) => index !== trackIndex),
  }))
}

export function renameTrack(state: EditorState, trackId: string, name: string): EditorState {
  return mapTracks(state, track => (track.id === trackId ? { ...track, name } : track))
}

export function toggleTrackLock(state: EditorState, trackId: string): EditorState {
  return mapTracks(state, track => (track.id === trackId ? { ...track, locked: !track.locked } : track))
}

export function toggleTrackMute(state: EditorState, trackId: string): EditorState {
  return mapTracks(state, track => (track.id === trackId ? { ...track, muted: !track.muted } : track))
}

export function toggleTrackEnabled(state: EditorState, trackId: string): EditorState {
  return mapTracks(state, track => (track.id === trackId ? { ...track, enabled: !(track.enabled ?? true) } : track))
}

export function toggleTrackSolo(state: EditorState, trackId: string): EditorState {
  return mapTracks(state, track => (track.id === trackId ? { ...track, solo: !track.solo } : track))
}

export function toggleTrackSourcePatched(state: EditorState, trackId: string): EditorState {
  return mapTracks(state, track => (track.id === trackId ? { ...track, sourcePatched: !(track.sourcePatched ?? true) } : track))
}

export function addSubtitleTrack(state: EditorState): EditorState {
  const count = selectTracks(state).filter(track => track.type === 'subtitle').length
  const track: Track = {
    id: makeId('track-sub'),
    name: count > 0 ? `Subtitles ${count + 1}` : 'Subtitles',
    muted: false,
    locked: false,
    kind: 'video',
    type: 'subtitle',
  }
  return replaceActiveTimeline(state, timeline => ({
    ...timeline,
    clips: timeline.clips.map(clip => ({ ...clip, trackIndex: clip.trackIndex + 1 })),
    subtitles: (timeline.subtitles || []).map(subtitle => ({ ...subtitle, trackIndex: subtitle.trackIndex + 1 })),
    tracks: [track, ...timeline.tracks],
  }))
}

export function importSrtCues(state: EditorState, cues: SrtCue[]): EditorState {
  if (cues.length === 0) return state

  let next = state
  let subtitleTrackIndex = selectTracks(next).findIndex(track => track.type === 'subtitle')
  if (subtitleTrackIndex === -1) {
    next = addSubtitleTrack(next)
    subtitleTrackIndex = 0
  }

  const importedSubtitles: SubtitleClip[] = cues.map(cue => ({
    id: `${makeId('sub')}-${cue.index}`,
    text: cue.text,
    startTime: cue.startTime,
    endTime: cue.endTime,
    trackIndex: subtitleTrackIndex,
    ...(cue.color ? { style: { color: cue.color } } : {}),
  }))

  return setTimelineSubtitles(next, prev => [
    ...prev.filter(subtitle => subtitle.trackIndex !== subtitleTrackIndex),
    ...importedSubtitles,
  ])
}

export function setSubtitleTrackStyle(state: EditorState, trackId: string, patch: Partial<SubtitleStyle>): EditorState {
  return mapTracks(state, track => (
    track.id === trackId
      ? { ...track, subtitleStyle: { ...(track.subtitleStyle || {}), ...patch } }
      : track
  ))
}

export function addSubtitle(state: EditorState, params: AddSubtitleParams): EditorState {
  const subtitle: SubtitleClip = {
    id: makeId('sub'),
    text: params.text || 'New subtitle',
    startTime: params.startTime ?? selectCurrentTime(state),
    endTime: params.endTime ?? (selectCurrentTime(state) + 3),
    trackIndex: params.trackIndex,
  }
  return replaceActiveTimeline(state, timeline => ({
    ...timeline,
    subtitles: [...(timeline.subtitles || []), subtitle],
  }))
}

export function deleteSubtitle(state: EditorState, subtitleId: string): EditorState {
  const next = replaceActiveTimeline(state, timeline => ({
    ...timeline,
    subtitles: (timeline.subtitles || []).filter(subtitle => subtitle.id !== subtitleId),
  }))
  return updateSession(next, session => ({
    ...session,
    selection: {
      ...session.selection,
      subtitleId: session.selection.subtitleId === subtitleId ? null : session.selection.subtitleId,
      editingSubtitleId: session.selection.editingSubtitleId === subtitleId ? null : session.selection.editingSubtitleId,
    },
  }))
}

export function updateSubtitle(state: EditorState, subtitleId: string, patch: Partial<SubtitleClip>): EditorState {
  return replaceActiveTimeline(state, timeline => ({
    ...timeline,
    subtitles: (timeline.subtitles || []).map(subtitle => (
      subtitle.id === subtitleId ? { ...subtitle, ...patch } : subtitle
    )),
  }))
}

export function setSubtitleText(state: EditorState, subtitleId: string, text: string): EditorState {
  return updateSubtitle(state, subtitleId, { text })
}

export function setSubtitleStart(state: EditorState, subtitleId: string, startTime: number): EditorState {
  return updateSubtitle(state, subtitleId, { startTime })
}

export function setSubtitleEnd(state: EditorState, subtitleId: string, endTime: number): EditorState {
  return updateSubtitle(state, subtitleId, { endTime })
}

export function setSubtitleStyleField<K extends keyof SubtitleStyle>(
  state: EditorState,
  subtitleId: string,
  field: K,
  value: SubtitleStyle[K],
): EditorState {
  const subtitle = state.editorModel.timelines.flatMap(timeline => timeline.subtitles || []).find(candidate => candidate.id === subtitleId)
  if (!subtitle) return state
  return updateSubtitle(state, subtitleId, {
    style: {
      ...(subtitle.style || {}),
      [field]: value,
    },
  })
}

export function addAssetToEditor(state: EditorState, asset: Asset): EditorState {
  return updateEditorModel(state, editorModel => ({
    ...editorModel,
    assets: [asset, ...editorModel.assets],
  }))
}

export function addAssetsToEditor(state: EditorState, assets: Asset[]): EditorState {
  return updateEditorModel(state, editorModel => ({
    ...editorModel,
    assets: [...assets, ...editorModel.assets],
  }))
}

export function insertGeneratedGapAsset(state: EditorState, params: InsertGeneratedGapAssetParams): EditorState {
  let next = addAssetToEditor(state, params.asset)

  let audioTrackIndex = -1
  if (params.createAudio) {
    audioTrackIndex = selectTracks(next).findIndex(
      track => track.kind === 'audio' && !track.locked && track.sourcePatched !== false,
    )
    if (audioTrackIndex < 0) {
      next = addTrack(next, 'audio')
      audioTrackIndex = selectTracks(next).length - 1
    }
  }

  const gapDuration = params.gap.endTime - params.gap.startTime
  const videoClipId = makeId('clip')
  const audioClipId = makeId('clip-audio')
  const newClips: TimelineClip[] = [{
    id: videoClipId,
    assetId: params.asset.id,
    type: params.asset.type === 'image' ? 'image' : 'video',
    startTime: params.gap.startTime,
    duration: gapDuration,
    trimStart: 0,
    trimEnd: 0,
    speed: 1,
    reversed: false,
    muted: false,
    volume: 1,
    trackIndex: params.gap.trackIndex,
    asset: params.asset,
    flipH: false,
    flipV: false,
    transitionIn: { type: 'none', duration: 0 },
    transitionOut: { type: 'none', duration: 0 },
    colorCorrection: { ...DEFAULT_COLOR_CORRECTION },
    opacity: 100,
    ...(params.createAudio && audioTrackIndex >= 0 ? { linkedClipIds: [audioClipId] } : {}),
  }]

  if (params.createAudio && audioTrackIndex >= 0) {
    newClips.push({
      id: audioClipId,
      assetId: params.asset.id,
      type: 'audio',
      startTime: params.gap.startTime,
      duration: gapDuration,
      trimStart: 0,
      trimEnd: 0,
      speed: 1,
      reversed: false,
      muted: false,
      volume: 1,
      trackIndex: audioTrackIndex,
      asset: params.asset,
      flipH: false,
      flipV: false,
      transitionIn: { type: 'none', duration: 0 },
      transitionOut: { type: 'none', duration: 0 },
      colorCorrection: { ...DEFAULT_COLOR_CORRECTION },
      opacity: 100,
      linkedClipIds: [videoClipId],
    })
  }

  return setTimelineClips(next, prev => [...prev, ...newClips])
}

export function deleteAssets(state: EditorState, assetIds: string[]): EditorState {
  const deleteSet = new Set(assetIds)
  return updateEditorModel(state, editorModel => ({
    ...editorModel,
    assets: editorModel.assets.filter(asset => !deleteSet.has(asset.id)),
    timelines: editorModel.timelines.map(timeline => ({
      ...timeline,
      clips: timeline.clips.filter(clip => !clip.assetId || !deleteSet.has(clip.assetId)),
    })),
  }))
}

export function updateAsset(state: EditorState, assetId: string, patch: Partial<Asset>): EditorState {
  return updateEditorModel(state, editorModel => ({
    ...editorModel,
    assets: editorModel.assets.map(asset => (asset.id === assetId ? { ...asset, ...patch } : asset)),
  }))
}

export function setAssetBin(state: EditorState, assetId: string, bin?: string): EditorState {
  return updateAsset(state, assetId, { bin })
}

export function assignAssetsToBin(state: EditorState, assetIds: string[], bin?: string): EditorState {
  const assetSet = new Set(assetIds)
  return updateEditorModel(state, editorModel => ({
    ...editorModel,
    assets: editorModel.assets.map(asset => (assetSet.has(asset.id) ? { ...asset, bin } : asset)),
  }))
}

export function renameBin(state: EditorState, oldName: string, newName: string): EditorState {
  return updateEditorModel(state, editorModel => ({
    ...editorModel,
    assets: editorModel.assets.map(asset => (asset.bin === oldName ? { ...asset, bin: newName } : asset)),
  }))
}

export function clearBin(state: EditorState, binName: string): EditorState {
  return updateEditorModel(state, editorModel => ({
    ...editorModel,
    assets: editorModel.assets.map(asset => (asset.bin === binName ? { ...asset, bin: undefined } : asset)),
  }))
}

export function toggleAssetFavorite(state: EditorState, assetId: string): EditorState {
  const asset = selectAssetById(state, assetId)
  if (!asset) return state
  return updateAsset(state, assetId, { favorite: !asset.favorite })
}

export function setAssetColorLabel(state: EditorState, assetId: string, colorLabel?: string): EditorState {
  const next = updateAsset(state, assetId, { colorLabel })
  return replaceActiveTimeline(next, timeline => ({
    ...timeline,
    clips: timeline.clips.map(clip => (
      clip.assetId === assetId
        ? { ...clip, colorLabel }
        : clip
    )),
  }))
}

export function addAssetTake(state: EditorState, assetId: string, take: AssetTake): EditorState {
  return updateEditorModel(state, editorModel => ({
    ...editorModel,
    assets: editorModel.assets.map(asset => {
      if (asset.id !== assetId) return asset
      const existingTakes: AssetTake[] = asset.takes || [{
        path: asset.path,
        bigThumbnailPath: asset.bigThumbnailPath,
        smallThumbnailPath: asset.smallThumbnailPath,
        width: asset.width,
        height: asset.height,
        createdAt: asset.createdAt,
      }]
      const nextTakes = [...existingTakes, take]
      const newIndex = nextTakes.length - 1
      return {
        ...asset,
        takes: nextTakes,
        activeTakeIndex: newIndex,
        path: take.path,
        bigThumbnailPath: take.bigThumbnailPath,
        smallThumbnailPath: take.smallThumbnailPath,
        width: take.width,
        height: take.height,
      }
    }),
  }))
}

export function deleteAssetTake(state: EditorState, assetId: string, takeIndex: number): EditorState {
  const next = updateEditorModel(state, editorModel => ({
    ...editorModel,
    assets: editorModel.assets.map(asset => {
      if (asset.id !== assetId || !asset.takes || asset.takes.length <= 1) return asset
      const nextTakes = asset.takes.filter((_, index) => index !== takeIndex)
      const nextActiveTakeIndex = Math.max(0, Math.min(asset.activeTakeIndex ?? (asset.takes.length - 1), nextTakes.length - 1))
      const activeTake = nextTakes[nextActiveTakeIndex]
      return {
        ...asset,
        takes: nextTakes,
        activeTakeIndex: nextActiveTakeIndex,
        path: activeTake.path,
        bigThumbnailPath: activeTake.bigThumbnailPath,
        smallThumbnailPath: activeTake.smallThumbnailPath,
        width: activeTake.width,
        height: activeTake.height,
      }
    }),
  }))
  return repointClipsAfterTakeDelete(next, assetId, takeIndex)
}

export function deleteClipDisplayedTake(state: EditorState, clipId: string): EditorState {
  const clip = selectClips(state).find(candidate => candidate.id === clipId)
  const asset = clip ? selectLiveAssetForClip(state, clip) : null
  if (!clip || !asset?.takes || asset.takes.length <= 1) return state
  const takeIndex = clip.takeIndex ?? asset.activeTakeIndex ?? 0
  return deleteAssetTake(state, asset.id, takeIndex)
}

export function setAssetActiveTake(state: EditorState, assetId: string, takeIndex: number): EditorState {
  return updateEditorModel(state, editorModel => ({
    ...editorModel,
    assets: editorModel.assets.map(asset => {
      if (asset.id !== assetId || !asset.takes || !asset.takes[takeIndex]) return asset
      const take = asset.takes[takeIndex]
      return {
        ...asset,
        activeTakeIndex: takeIndex,
        path: take.path,
        bigThumbnailPath: take.bigThumbnailPath,
        smallThumbnailPath: take.smallThumbnailPath,
        width: take.width,
        height: take.height,
      }
    }),
  }))
}

export function repointClipsAfterTakeDelete(state: EditorState, assetId: string, deletedTakeIndex: number): EditorState {
  return updateEditorModel(state, editorModel => ({
    ...editorModel,
    timelines: editorModel.timelines.map(timeline => ({
      ...timeline,
      clips: timeline.clips.map(clip => {
        if (clip.assetId !== assetId) return clip
        const currentIndex = clip.takeIndex
        if (currentIndex === undefined) return clip
        if (currentIndex === deletedTakeIndex) return { ...clip, takeIndex: Math.max(0, deletedTakeIndex - 1) }
        if (currentIndex > deletedTakeIndex) return { ...clip, takeIndex: currentIndex - 1 }
        return clip
      }),
    })),
  }))
}

export function selectClip(state: EditorState, clipId: string, options: SelectClipMode = {}): EditorState {
  const mode = options.mode ?? 'replace'
  const current = state.session.selection.clipIds
  const next = cloneClipIds(current)
  if (mode === 'replace') {
    next.clear()
    next.add(clipId)
  } else if (mode === 'toggle') {
    if (next.has(clipId)) next.delete(clipId)
    else next.add(clipId)
  } else {
    next.add(clipId)
  }
  return updateSession(state, session => ({
    ...session,
    selection: {
      ...session.selection,
      clipIds: next,
      subtitleId: null,
    },
  }))
}

export function selectClipsById(state: EditorState, clipIds: string[], options: SelectClipMode = {}): EditorState {
  const mode = options.mode ?? 'replace'
  const next = mode === 'replace' ? new Set<string>() : cloneClipIds(state.session.selection.clipIds)
  for (const clipId of clipIds) {
    if (mode === 'toggle') {
      if (next.has(clipId)) next.delete(clipId)
      else next.add(clipId)
    } else {
      next.add(clipId)
    }
  }
  return updateSession(state, session => ({
    ...session,
    selection: {
      ...session.selection,
      clipIds: next,
      subtitleId: null,
    },
  }))
}

export { selectClipsById as selectClips }

export function selectLinkedGroup(state: EditorState, clipId: string): EditorState {
  const clips = selectClips(state)
  const first = clips.find(clip => clip.id === clipId)
  if (!first) return state
  const linkedIds = new Set([first.id])
  const queue = [first]
  while (queue.length > 0) {
    const clip = queue.pop()!
    if (!clip.linkedClipIds) continue
    for (const linkedId of clip.linkedClipIds) {
      if (linkedIds.has(linkedId)) continue
      const linkedClip = clips.find(candidate => candidate.id === linkedId)
      if (!linkedClip) continue
      linkedIds.add(linkedId)
      queue.push(linkedClip)
    }
  }
  return selectClipsById(state, [...linkedIds])
}

export function selectAllClips(state: EditorState): EditorState {
  return updateSession(state, session => ({
    ...session,
    selection: {
      ...session.selection,
      clipIds: new Set(selectClips(state).map(clip => clip.id)),
      subtitleId: null,
    },
  }))
}

export function clearClipSelection(state: EditorState): EditorState {
  return updateSession(state, session => ({
    ...session,
    selection: {
      ...session.selection,
      clipIds: new Set(),
    },
  }))
}

export function setSelectedClipIds(state: EditorState, value: SetStateAction<Set<string>>): EditorState {
  return updateSession(state, session => ({
    ...session,
    selection: {
      ...session.selection,
      clipIds: applyStateAction(value, session.selection.clipIds),
      subtitleId: null,
    },
  }))
}

export function setSelectedSubtitle(state: EditorState, subtitleId?: string): EditorState {
  return updateSession(state, session => ({
    ...session,
    selection: {
      ...session.selection,
      subtitleId: subtitleId ?? null,
      clipIds: new Set(),
    },
  }))
}

export function clearSelectedSubtitle(state: EditorState): EditorState {
  return updateSession(state, session => ({
    ...session,
    selection: {
      ...session.selection,
      subtitleId: null,
      editingSubtitleId: null,
    },
  }))
}

export function setEditingSubtitleId(state: EditorState, value: SetStateAction<string | null>): EditorState {
  return updateSession(state, session => ({
    ...session,
    selection: {
      ...session.selection,
      editingSubtitleId: applyStateAction(value, session.selection.editingSubtitleId),
    },
  }))
}

export function setSelectedGap(state: EditorState, gap?: TimelineGapSelection): EditorState {
  return updateSession(state, session => ({
    ...session,
    selection: {
      ...session.selection,
      gap: gap ?? null,
    },
  }))
}

export function clearSelectedGap(state: EditorState): EditorState {
  return setSelectedGap(state, undefined)
}

export function setCurrentTime(state: EditorState, time: number): EditorState {
  return updateSession(state, session => ({
    ...session,
    transport: {
      ...session.transport,
      currentTime: Math.max(0, time),
    },
  }))
}

export function stepCurrentTime(state: EditorState, delta: number): EditorState {
  return setCurrentTime(state, Math.max(0, selectCurrentTime(state) + delta))
}

export function play(state: EditorState): EditorState {
  return updateSession(state, session => ({
    ...session,
    transport: {
      ...session.transport,
      isPlaying: true,
    },
  }))
}

export function pause(state: EditorState): EditorState {
  return updateSession(state, session => ({
    ...session,
    transport: {
      ...session.transport,
      isPlaying: false,
    },
  }))
}

export function togglePlayPause(state: EditorState): EditorState {
  return updateSession(state, session => ({
    ...session,
    transport: {
      ...session.transport,
      isPlaying: !session.transport.isPlaying,
    },
  }))
}

export function setShuttleSpeed(state: EditorState, speed: number): EditorState {
  return updateSession(state, session => ({
    ...session,
    transport: {
      ...session.transport,
      shuttleSpeed: speed,
    },
  }))
}

export function stopShuttle(state: EditorState): EditorState {
  return setShuttleSpeed(pause(state), 0)
}

export function setTimelineInPoint(state: EditorState, time?: number | null): EditorState {
  const activeTimelineId = selectActiveTimelineId(state) || ''
  if (!activeTimelineId) return state
  const current = state.session.transport.timelineInOutMap[activeTimelineId] || { inPoint: null, outPoint: null }
  let inPoint = time ?? null
  if (inPoint !== null && current.outPoint !== null && inPoint >= current.outPoint) {
    inPoint = current.outPoint - 0.01
  }
  return updateSession(state, session => ({
    ...session,
    transport: {
      ...session.transport,
      timelineInOutMap: {
        ...session.transport.timelineInOutMap,
        [activeTimelineId]: {
          ...current,
          inPoint,
        },
      },
    },
  }))
}

export function setTimelineOutPoint(state: EditorState, time?: number | null): EditorState {
  const activeTimelineId = selectActiveTimelineId(state) || ''
  if (!activeTimelineId) return state
  const current = state.session.transport.timelineInOutMap[activeTimelineId] || { inPoint: null, outPoint: null }
  let outPoint = time ?? null
  if (outPoint !== null && current.inPoint !== null && outPoint <= current.inPoint) {
    outPoint = current.inPoint + 0.01
  }
  return updateSession(state, session => ({
    ...session,
    transport: {
      ...session.transport,
      timelineInOutMap: {
        ...session.transport.timelineInOutMap,
        [activeTimelineId]: {
          ...current,
          outPoint,
        },
      },
    },
  }))
}

export function clearTimelineInPoint(state: EditorState): EditorState {
  return setTimelineInPoint(state, null)
}

export function clearTimelineOutPoint(state: EditorState): EditorState {
  return setTimelineOutPoint(state, null)
}

export function clearTimelineMarks(state: EditorState): EditorState {
  let next = clearTimelineInPoint(state)
  next = clearTimelineOutPoint(next)
  return updateSession(next, session => ({
    ...session,
    transport: {
      ...session.transport,
      playingInOut: false,
    },
  }))
}

export function goToInPoint(state: EditorState): EditorState {
  const inPoint = selectActiveTimelineInPoint(state)
  const target = inPoint ?? (selectClips(state).length > 0 ? Math.min(...selectClips(state).map(clip => clip.startTime)) : 0)
  return stopShuttle(setCurrentTime(state, target))
}

export function goToOutPoint(state: EditorState): EditorState {
  const outPoint = selectActiveTimelineOutPoint(state)
  const total = selectClips(state).length > 0
    ? Math.max(...selectClips(state).map(clip => clip.startTime + clip.duration))
    : 0
  return stopShuttle(setCurrentTime(state, outPoint ?? total))
}

export function goToPrevEdit(state: EditorState, anchorTime?: number): EditorState {
  const current = Math.round((anchorTime ?? selectCurrentTime(state)) * 1000) / 1000
  const points = new Set<number>([0])
  for (const clip of selectClips(state)) {
    points.add(Math.round(clip.startTime * 1000) / 1000)
    points.add(Math.round((clip.startTime + clip.duration) * 1000) / 1000)
  }
  const sorted = Array.from(points).sort((a, b) => a - b)
  let target = sorted[0] ?? 0
  for (const point of sorted) {
    if (point < current - 0.01) target = point
    else break
  }
  return setCurrentTime(state, target)
}

export function goToNextEdit(state: EditorState, anchorTime?: number): EditorState {
  const current = Math.round((anchorTime ?? selectCurrentTime(state)) * 1000) / 1000
  const points = new Set<number>()
  for (const clip of selectClips(state)) {
    points.add(Math.round(clip.startTime * 1000) / 1000)
    points.add(Math.round((clip.startTime + clip.duration) * 1000) / 1000)
  }
  const sorted = Array.from(points).sort((a, b) => a - b)
  let target = sorted.length > 0 ? sorted[sorted.length - 1] : selectCurrentTime(state)
  for (const point of sorted) {
    if (point > current + 0.01) {
      target = point
      break
    }
  }
  return setCurrentTime(state, target)
}

export function togglePlayInOut(state: EditorState): EditorState {
  return updateSession(state, session => ({
    ...session,
    transport: {
      ...session.transport,
      playingInOut: !session.transport.playingInOut,
    },
  }))
}

export function setPlayingInOut(state: EditorState, value: boolean): EditorState {
  return updateSession(state, session => ({
    ...session,
    transport: {
      ...session.transport,
      playingInOut: value,
    },
  }))
}

export function setSnapEnabled(state: EditorState, enabled: boolean): EditorState {
  return updateSession(state, session => ({
    ...session,
    tools: {
      ...session.tools,
      snapEnabled: enabled,
    },
  }))
}

export function setZoom(state: EditorState, zoom: number): EditorState {
  return updateSession(state, session => ({
    ...session,
    tools: {
      ...session.tools,
      zoom,
    },
  }))
}

export function zoomIn(state: EditorState): EditorState {
  return setZoom(state, Math.min(selectClips(state).length > 0 ? state.session.tools.zoom * 1.25 : 1.25, 10))
}

export function zoomOut(state: EditorState): EditorState {
  return setZoom(state, Math.max(state.session.tools.zoom / 1.25, 0.1))
}

export function fitTimelineToView(state: EditorState): EditorState {
  return state
}

export function toggleSnap(state: EditorState): EditorState {
  return updateSession(state, session => ({
    ...session,
    tools: {
      ...session.tools,
      snapEnabled: !session.tools.snapEnabled,
    },
  }))
}

export function setActiveTool(state: EditorState, tool: ToolType): EditorState {
  return updateSession(state, session => ({
    ...session,
    tools: {
      ...session.tools,
      activeTool: tool,
    },
  }))
}

export function setLastTrimTool(state: EditorState, tool: ToolType): EditorState {
  return updateSession(state, session => ({
    ...session,
    tools: {
      ...session.tools,
      lastTrimTool: tool,
    },
  }))
}

export function setShowSourceMonitor(state: EditorState, value: boolean): EditorState {
  return updateSession(state, session => ({
    ...session,
    ui: {
      ...session.ui,
      showSourceMonitor: value,
    },
  }))
}

export function closeSourceMonitor(state: EditorState): EditorState {
  return updateSession(state, session => ({
    ...session,
    ui: {
      ...session.ui,
      showSourceMonitor: false,
      activeFocusArea: 'timeline',
      hasSourceAsset: false,
    },
  }))
}

export function setShowPropertiesPanel(state: EditorState, value: boolean): EditorState {
  return updateSession(state, session => ({
    ...session,
    ui: {
      ...session.ui,
      showPropertiesPanel: value,
    },
  }))
}

export function setShowEffectsBrowser(state: EditorState, value: boolean): EditorState {
  return updateSession(state, session => ({
    ...session,
    ui: {
      ...session.ui,
      showEffectsBrowser: value,
    },
  }))
}

export function setActiveFocusArea(state: EditorState, area: 'source' | 'timeline'): EditorState {
  return updateSession(state, session => ({
    ...session,
    ui: {
      ...session.ui,
      activeFocusArea: area,
    },
  }))
}

export function setSourceSplitPercent(state: EditorState, percent: number): EditorState {
  return updateSession(state, session => ({
    ...session,
    ui: {
      ...session.ui,
      sourceSplitPercent: percent,
    },
  }))
}

export function setHasSourceAsset(state: EditorState, value: boolean): EditorState {
  return updateSession(state, session => ({
    ...session,
    ui: {
      ...session.ui,
      hasSourceAsset: value,
    },
  }))
}

export function openImportTimelineModal(state: EditorState): EditorState {
  return updateSession(state, session => ({
    ...session,
    ui: {
      ...session.ui,
      showImportTimelineModal: true,
    },
  }))
}

export function closeImportTimelineModal(state: EditorState): EditorState {
  return updateSession(state, session => ({
    ...session,
    ui: {
      ...session.ui,
      showImportTimelineModal: false,
    },
  }))
}

export function openExportModal(state: EditorState): EditorState {
  return updateSession(state, session => ({
    ...session,
    ui: {
      ...session.ui,
      showExportModal: true,
    },
  }))
}

export function closeExportModal(state: EditorState): EditorState {
  return updateSession(state, session => ({
    ...session,
    ui: {
      ...session.ui,
      showExportModal: false,
    },
  }))
}

export function setOpenTimelineIds(state: EditorState, ids: Set<string>): EditorState {
  return updateSession(state, session => ({
    ...session,
    ui: {
      ...session.ui,
      openTimelineIds: new Set(ids),
    },
  }))
}

export function openTimelineTab(state: EditorState, timelineId: string): EditorState {
  return setOpenTimelineIds(state, new Set([...state.session.ui.openTimelineIds, timelineId]))
}

export function closeTimelineTab(state: EditorState, timelineId: string): EditorState {
  return setOpenTimelineIds(state, new Set([...state.session.ui.openTimelineIds].filter(id => id !== timelineId)))
}

export function startTimelineRename(state: EditorState, timelineId: string, source: 'tab' | 'panel'): EditorState {
  const timeline = state.editorModel.timelines.find(candidate => candidate.id === timelineId)
  return updateSession(state, session => ({
    ...session,
    ui: {
      ...session.ui,
      renamingTimelineId: timelineId,
      renameValue: timeline?.name || '',
      renameSource: source,
    },
  }))
}

export function setTimelineRenameValue(state: EditorState, value: string): EditorState {
  return updateSession(state, session => ({
    ...session,
    ui: {
      ...session.ui,
      renameValue: value,
    },
  }))
}

export function commitTimelineRename(state: EditorState): EditorState {
  const { renamingTimelineId, renameValue } = state.session.ui
  if (!renamingTimelineId || !renameValue.trim()) return cancelTimelineRename(state)
  return cancelTimelineRename(renameTimeline(state, renamingTimelineId, renameValue.trim()))
}

export function cancelTimelineRename(state: EditorState): EditorState {
  return updateSession(state, session => ({
    ...session,
    ui: {
      ...session.ui,
      renamingTimelineId: null,
      renameValue: '',
    },
  }))
}

export function setLayout(state: EditorState, layout: EditorLayout): EditorState {
  return updateSession(state, session => ({
    ...session,
    ui: {
      ...session.ui,
      layout,
    },
  }))
}

export function resetLayout(state: EditorState): EditorState {
  return setLayout(state, {
    leftPanelWidth: 288,
    rightPanelWidth: 256,
    timelineHeight: 224,
    assetsHeight: 0,
  })
}

export function setSubtitleTrackStyleEditorTrack(state: EditorState, trackIdx?: number): EditorState {
  return updateSession(state, session => ({
    ...session,
    ui: {
      ...session.ui,
      subtitleTrackStyleIdx: trackIdx ?? null,
    },
  }))
}

export function updateSubtitleTrackStyle(
  state: EditorState,
  trackIndex: number,
  patch: Partial<SubtitleStyle>,
): EditorState {
  return mapTracks(state, (track, index) => (
    index === trackIndex
      ? {
          ...track,
          subtitleStyle: {
            ...track.subtitleStyle,
            ...patch,
          },
        }
      : track
  ))
}

export function clearSubtitleOverridesForTrack(state: EditorState, trackIndex: number): EditorState {
  return replaceActiveTimeline(state, timeline => ({
    ...timeline,
    subtitles: (timeline.subtitles || []).map(subtitle => (
      subtitle.trackIndex === trackIndex
        ? { ...subtitle, style: undefined }
        : subtitle
    )),
  }))
}

export function copySelection(state: EditorState): EditorState {
  const selectedIds = selectSelectedClipIds(state)
  const clips = selectClips(state).filter(clip => selectedIds.has(clip.id))
  return updateSession(state, session => ({
    ...session,
    clipboard: {
      kind: 'clips',
      clips,
      copiedFromTimelineId: selectActiveTimelineId(state),
    },
  }))
}

export function cutSelection(state: EditorState): EditorState {
  const copied = copySelection(state)
  return deleteClips(copied, [...selectSelectedClipIds(copied)])
}

export function pasteSelection(state: EditorState, atTime?: number): EditorState {
  if (!selectCanUseClipboard(state)) return state
  const clipboard = state.session.clipboard
  const earliestStart = clipboard.clips.reduce((min, clip) => Math.min(min, clip.startTime), Infinity)
  const pasteAt = atTime ?? selectCurrentTime(state)
  const duplicates = clipboard.clips
    .filter(clip => !clip.assetId || Boolean(selectAssetById(state, clip.assetId)))
    .map(clip => ({
      ...clip,
      id: makeId('clip'),
      startTime: pasteAt + (clip.startTime - earliestStart),
      linkedClipIds: undefined,
    }))
  if (duplicates.length === 0) return state

  let next = replaceActiveTimeline(state, timeline => ({
    ...timeline,
    clips: [...timeline.clips, ...duplicates],
  }))
  next = setSelectedClipIds(next, new Set(duplicates.map(clip => clip.id)))
  return next
}

export function undo(state: EditorState): EditorState {
  const previous = state.history.undoStack[state.history.undoStack.length - 1]
  if (!previous) return state
  const currentSnapshot = getUndoSnapshot(state)
  const next = applyUndoSnapshot(state, previous)
  if (equalUndoSnapshot(previous, currentSnapshot)) {
    return {
      ...next,
      history: {
        undoStack: state.history.undoStack.slice(0, -1),
        redoStack: state.history.redoStack,
      },
    }
  }
  return {
    ...next,
    history: {
      undoStack: state.history.undoStack.slice(0, -1),
      redoStack: [...state.history.redoStack, currentSnapshot],
    },
  }
}

export function redo(state: EditorState): EditorState {
  const next = state.history.redoStack[state.history.redoStack.length - 1]
  if (!next) return state
  const currentSnapshot = getUndoSnapshot(state)
  const restored = applyUndoSnapshot(state, next)
  if (equalUndoSnapshot(next, currentSnapshot)) {
    return {
      ...restored,
      history: {
        undoStack: state.history.undoStack,
        redoStack: state.history.redoStack.slice(0, -1),
      },
    }
  }
  return {
    ...restored,
    history: {
      undoStack: [...state.history.undoStack, currentSnapshot],
      redoStack: state.history.redoStack.slice(0, -1),
    },
  }
}

export function startClipRegeneration(state: EditorState, assetId: string, clipId?: string): EditorState {
  let next = updateSession(state, session => ({
    ...session,
    regeneration: {
      ...session.regeneration,
      regeneratingAssetId: assetId,
      regeneratingClipId: clipId ?? null,
      preError: null,
    },
  }))
  if (clipId) {
    next = updateClip(next, clipId, { isRegenerating: true })
  }
  return next
}

export function cancelClipRegeneration(state: EditorState): EditorState {
  const clipId = state.session.regeneration.regeneratingClipId
  let next = updateSession(state, session => ({
    ...session,
    regeneration: {
      ...session.regeneration,
      regeneratingAssetId: null,
      regeneratingClipId: null,
    },
  }))
  if (clipId) next = updateClip(next, clipId, { isRegenerating: false })
  return next
}

export function setClipRegenerating(state: EditorState, clipId: string, value: boolean): EditorState {
  return updateClip(state, clipId, { isRegenerating: value })
}

export function applyGeneratedTake(state: EditorState, assetId: string, take: AssetTake, clipId?: string): EditorState {
  let next = addAssetTake(state, assetId, take)
  const asset = selectAssetById(next, assetId)
  const takeIndex = asset?.takes ? asset.takes.length - 1 : undefined
  if (clipId && takeIndex !== undefined) {
    next = updateClip(next, clipId, {
      takeIndex,
      isRegenerating: false,
    })
  }
  return updateSession(next, session => ({
    ...session,
    regeneration: {
      regeneratingAssetId: null,
      regeneratingClipId: null,
      preError: null,
    },
  }))
}

export function clearRegenerationError(state: EditorState): EditorState {
  return updateSession(state, session => ({
    ...session,
    regeneration: {
      ...session.regeneration,
      preError: null,
    },
  }))
}

export function setRegenerationPreError(state: EditorState, message: string | null): EditorState {
  return updateSession(state, session => ({
    ...session,
    regeneration: {
      ...session.regeneration,
      preError: message,
    },
  }))
}

export function failClipRegeneration(state: EditorState, message: string): EditorState {
  return setRegenerationPreError(cancelClipRegeneration(state), message)
}

export function initializeFromProject(project: Project, layout?: EditorLayout): EditorState {
  return createInitialEditorState(getEditorModel(project), layout)
}

export function deriveProjectPatch(state: EditorState): EditorModel {
  return state.editorModel
}

export function commitToProject(state: EditorState, baseProject: Project): Project {
  return updatedProject(baseProject, state.editorModel)
}

export function setDirty(state: EditorState, dirty: boolean): EditorState {
  return {
    ...state,
    projectSync: {
      ...state.projectSync,
      dirty,
    },
  }
}

export function markProjectDirty(state: EditorState): EditorState {
  return setDirty(state, true)
}

export function clearProjectDirty(state: EditorState): EditorState {
  return setDirty(state, false)
}

export { markProjectDirty as markDirty }
export { clearProjectDirty as clearDirty }

export function applyPendingClipTakeUpdate(
  editorModel: EditorModel,
  pendingUpdate: PendingClipTakeUpdate | null,
): EditorModel {
  if (!pendingUpdate) return editorModel
  return {
    ...editorModel,
    timelines: editorModel.timelines.map(timeline => ({
      ...timeline,
      clips: timeline.clips.map(clip => {
        if (clip.assetId !== pendingUpdate.assetId) return clip
        if (!pendingUpdate.clipIds.includes(clip.id)) return clip
        return { ...clip, takeIndex: pendingUpdate.newTakeIndex }
      }),
    })),
  }
}
