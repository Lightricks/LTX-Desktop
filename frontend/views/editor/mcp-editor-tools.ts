/**
 * Renderer side of the RiX MCP server (electron/mcp). The main process forwards
 * editor tool calls here over IPC; they run against the live editor store so
 * agent edits share the undo stack and autosave path with manual edits.
 *
 * apply_edits is atomic: all ops are folded into ONE setStateWithHistory call, so
 * a whole agent pass is a single Ctrl+Z — and a failing op leaves the timeline
 * untouched.
 */
import type { Asset, TextOverlayStyle, TimelineClip, TransitionType } from '../../types/project-model'
import { DEFAULT_TEXT_STYLE, transitionTypeValues } from '../../types/project-model'
import { addGenericAssetToProject } from '../../lib/asset-copy'
import { pathToFileUrl } from '../../lib/file-url'
import type { EditorState } from './editor-state'
import type { EditorStoreApi } from './editor-store'
import * as actions from './editor-actions'
import { selectActiveTimeline, selectAssets, selectClips, selectSubtitles, selectTracks } from './editor-selectors'
import { resolveOverlaps } from './video-editor-utils'
import { buildAssetFromPath } from './dropped-media-asset'
import { buildExportPayload } from './export-payload'

type McpEditorEntry = { store: EditorStoreApi; projectId: string; projectName: string }

type HistoryMark = { store: EditorStoreApi; model: EditorState['editorModel'] }

// Module state lives on globalThis so a dev hot-reload of this file doesn't drop the
// mounted editor's registration (VideoEditor won't re-register: its deps didn't change).
const registry = ((globalThis as Record<string, unknown>).__rixMcpEditor ??= {
  activeEditor: null,
  mcpUndoable: [],
  mcpRedoable: [],
}) as {
  activeEditor: McpEditorEntry | null
  mcpUndoable: HistoryMark[]
  mcpRedoable: HistoryMark[]
  handler?: (tool: string, args: Record<string, unknown>) => Promise<unknown>
}

/** For the always-mounted IPC listener: resolves the CURRENT module's handler at call
 *  time, so a hot-reloaded tools module is picked up without resubscribing. */
export function dispatchMcpEditorRequest(tool: string, args: Record<string, unknown>): Promise<unknown> {
  return (registry.handler ?? handleMcpEditorRequest)(tool, args)
}

/** Called by the mounted VideoEditor; returns an unregister function. */
export function registerMcpEditor(entry: McpEditorEntry): () => void {
  registry.activeEditor = entry
  return () => { if (registry.activeEditor === entry) registry.activeEditor = null }
}

function requireEditor(): McpEditorEntry {
  if (!registry.activeEditor) {
    throw new Error('No RiX project is open in the editor. Open a project and visit the Video Editor tab once, then retry.')
  }
  return registry.activeEditor
}

/** Mutating tools must name the project they mean to change: the user can switch
 *  projects mid-run, and an edit/undo landing in the wrong project destroys work. */
function requireProject(args: Record<string, unknown>): McpEditorEntry {
  const entry = requireEditor()
  const expected = args.project_id
  if (typeof expected !== 'string' || !expected) {
    throw new Error('"project_id" is required for edits — take it from rix_status.')
  }
  if (expected !== entry.projectId) {
    throw new Error(
      `Refusing: the open project is now "${entry.projectName}" (${entry.projectId}), not ${expected}. `
      + 'The user switched projects — stop and confirm with them before editing.',
    )
  }
  return entry
}

// MCP undo/redo may only walk back the MCP's OWN batches, and only while nothing
// else has touched the document since — never the user's edits. Each mark is the
// editorModel object produced by an MCP step (reference equality = untouched).
const { mcpUndoable, mcpRedoable } = registry

function stepHistory(args: Record<string, unknown>, direction: 'undo' | 'redo') {
  const { store } = requireProject(args)
  const marks = direction === 'undo' ? mcpUndoable : mcpRedoable
  const top = marks[marks.length - 1]
  if (!top || top.store !== store || top.model !== store.getState().state.editorModel) {
    throw new Error(
      `Refusing to ${direction}: the last change here was not an MCP ${direction === 'undo' ? 'batch' : 'undo'} `
      + '(the user may have edited since, or there is nothing to revert). Make a corrective apply_edits batch instead, '
      + `or ask the user to ${direction} in the app.`,
    )
  }
  marks.pop()
  store.getState().setStateWithoutHistory(prev => (direction === 'undo' ? actions.undo(prev) : actions.redo(prev)))
  const after = { store, model: store.getState().state.editorModel };
  (direction === 'undo' ? mcpRedoable : mcpUndoable).push(after)
  return { ok: true, timeline: getTimeline() }
}

const round = (n: number) => Math.round(n * 1000) / 1000
const basename = (p: string) => p.split(/[\\/]/).pop() || p
const AUDIO_EXT = /\.(mp3|wav|ogg|aac|flac|m4a)$/i
const VIDEO_EXT = /\.(mp4|webm|mkv|mov|avi)$/i

function timelineDuration(clips: TimelineClip[]): number {
  return clips.reduce((max, c) => Math.max(max, c.startTime + c.duration), 0)
}

function assetLabel(asset: Asset | null | undefined): string {
  if (!asset) return ''
  return asset.prompt.startsWith('Imported: ') ? asset.prompt.slice(10) : basename(asset.path)
}

// ---------------------------------------------------------------- read tools

function status() {
  const { store, projectId, projectName } = requireEditor()
  const state = store.getState().state
  const active = selectActiveTimeline(state)
  return {
    project: { id: projectId, name: projectName },
    activeTimelineId: active?.id ?? null,
    timelines: state.editorModel.timelines.map(t => ({
      id: t.id,
      name: t.name,
      active: t.id === active?.id,
      clipCount: t.clips.length,
      duration: round(timelineDuration(t.clips)),
    })),
    projectAssetCount: state.editorModel.assets.length,
  }
}

function listProjectAssets(args: { query?: string; type?: string; limit?: number }) {
  const { store } = requireEditor()
  const state = store.getState().state
  const bins = state.editorModel.bins
  const q = args.query?.toLowerCase().trim()
  const matches = selectAssets(state)
    .filter(a => a.type !== 'adjustment')
    .filter(a => !args.type || a.type === args.type)
    .filter(a => !q || a.prompt.toLowerCase().includes(q) || a.path.toLowerCase().includes(q)
      || (a.binId && bins[a.binId]?.toLowerCase().includes(q)))
    .sort((a, b) => b.createdAt - a.createdAt)
  const limit = Math.max(1, Math.min(args.limit ?? 50, 500))
  return {
    total: matches.length,
    assets: matches.slice(0, limit).map(a => ({
      id: a.id,
      type: a.type,
      path: a.path,
      duration: a.duration != null ? round(a.duration) : undefined,
      width: a.width,
      height: a.height,
      prompt: a.prompt.length > 240 ? `${a.prompt.slice(0, 240)}…` : a.prompt,
      folder: a.binId ? bins[a.binId] : undefined,
      favorite: a.favorite || undefined,
      createdAt: new Date(a.createdAt).toISOString(),
    })),
  }
}

function describeClip(c: TimelineClip) {
  return {
    id: c.id,
    track: c.trackIndex,
    type: c.type,
    start: round(c.startTime),
    end: round(c.startTime + c.duration),
    duration: round(c.duration),
    ...(c.type === 'text'
      ? { text: c.textStyle?.text, fadeIn: c.textFadeIn ?? 0.5, fadeOut: c.textFadeOut ?? 0.5 }
      : {
        in: round(c.trimStart),
        asset: c.asset ? { id: c.asset.id, name: assetLabel(c.asset), path: c.asset.path, duration: c.asset.duration } : undefined,
      }),
    ...(c.speed !== 1 ? { speed: c.speed } : {}),
    ...(c.reversed ? { reversed: true } : {}),
    ...(c.muted ? { muted: true } : {}),
    ...(c.volume !== 1 ? { volume: c.volume } : {}),
    ...(c.audioFadeIn ? { audioFadeIn: c.audioFadeIn } : {}),
    ...(c.audioFadeOut ? { audioFadeOut: c.audioFadeOut } : {}),
    ...(c.transitionIn?.type && c.transitionIn.type !== 'none' ? { transitionIn: c.transitionIn } : {}),
    ...(c.transitionOut?.type && c.transitionOut.type !== 'none' ? { transitionOut: c.transitionOut } : {}),
    ...(c.opacity !== 100 ? { opacity: c.opacity } : {}),
    ...(c.linkedClipIds?.length ? { linkedClipIds: c.linkedClipIds } : {}),
  }
}

function getTimeline() {
  const { store } = requireEditor()
  const state = store.getState().state
  const timeline = selectActiveTimeline(state)
  if (!timeline) throw new Error('No active timeline')
  const clips = [...timeline.clips].sort((a, b) => a.trackIndex - b.trackIndex || a.startTime - b.startTime)
  return {
    id: timeline.id,
    name: timeline.name,
    duration: round(timelineDuration(clips)),
    tracks: timeline.tracks.map((t, index) => ({
      index,
      name: t.name,
      kind: t.type === 'subtitle' ? 'subtitle' : t.kind ?? 'video',
      ...(t.muted ? { muted: true } : {}),
      ...(t.locked ? { locked: true } : {}),
      ...(t.enabled === false ? { enabled: false } : {}),
    })),
    clips: clips.map(describeClip),
    subtitles: (timeline.subtitles || []).map(s => ({ id: s.id, text: s.text, start: s.startTime, end: s.endTime })),
  }
}

/** Export inputs for the active timeline, plus a frame size matching its first visual clip. */
function exportPayload() {
  const { store, projectName } = requireEditor()
  const state = store.getState().state
  const assets = selectAssets(state)
  const clips = selectClips(state)
  const payload = buildExportPayload(assets, clips, selectTracks(state), selectSubtitles(state))
  if (payload.clips.length === 0) throw new Error('The active timeline has no video/image/audio clips to render')
  const firstVisual = [...clips].sort((a, b) => a.startTime - b.startTime)
    .find(c => (c.type === 'video' || c.type === 'image') && c.asset?.width && c.asset?.height)
  const aspect = firstVisual ? firstVisual.asset!.width! / firstVisual.asset!.height! : 16 / 9
  return {
    payload,
    aspect,
    duration: timelineDuration(clips),
    projectName,
    timelineName: selectActiveTimeline(state)?.name ?? 'Timeline',
  }
}

// ---------------------------------------------------------------- apply_edits

type AssetRef = string // asset id, "$ref" from this batch, or an absolute file path

type EditOp =
  | { op: 'import'; path: string; ref?: string }
  | { op: 'add_clip'; asset: AssetRef; track?: number; start?: number; in?: number; duration?: number; speed?: number; with_audio?: boolean; audio_track?: number; ref?: string }
  | { op: 'duck'; clip: string; ranges: Array<[number, number]>; level?: number; attack?: number; release?: number }
  | { op: 'add_text'; text: string; start: number; duration?: number; track?: number; style?: Partial<TextOverlayStyle>; fade_in?: number; fade_out?: number; ref?: string }
  | { op: 'update_clip'; clip: string; set: Record<string, unknown> }
  | { op: 'transition'; clip: string; in?: { type: TransitionType; duration?: number }; out?: { type: TransitionType; duration?: number } }
  | { op: 'delete'; clips: string[] }
  | { op: 'clear' }
  | { op: 'new_timeline'; name?: string }
  | { op: 'add_track'; kind: 'video' | 'audio' }

const looksLikePath = (s: string) => /[\\/]/.test(s) && /\.[a-z0-9]{2,4}$/i.test(s)

// Library files already imported this session, so re-running a cut doesn't
// duplicate the same source into the project over and over.
const importedBySource = new Map<string, string>()

function mediaDuration(path: string, audio: boolean): Promise<number> {
  return new Promise(resolve => {
    const el = document.createElement(audio ? 'audio' : 'video')
    el.preload = 'metadata'
    el.src = pathToFileUrl(path)
    el.onloadedmetadata = () => resolve(Number.isFinite(el.duration) ? el.duration : 5)
    el.onerror = () => resolve(5)
  })
}

async function importPath(srcPath: string, entry: McpEditorEntry, known: Asset[]): Promise<Asset> {
  const priorId = importedBySource.get(srcPath)
  const prior = (priorId && known.find(a => a.id === priorId)) || known.find(a => a.path === srcPath)
  if (prior) return prior
  let asset: Asset
  if (AUDIO_EXT.test(srcPath)) {
    const copied = await addGenericAssetToProject(srcPath, entry.projectId)
    const path = copied?.path ?? srcPath
    asset = {
      id: `asset-${Date.now()}-${Math.random().toString(36).slice(2, 11)}`,
      type: 'audio',
      path,
      prompt: `Imported: ${basename(srcPath)}`,
      resolution: 'imported',
      duration: await mediaDuration(path, true),
      createdAt: Date.now(),
    }
  } else {
    asset = await buildAssetFromPath(srcPath, VIDEO_EXT.test(srcPath), entry.projectId, basename(srcPath))
  }
  importedBySource.set(srcPath, asset.id)
  return asset
}

function num(v: unknown, name: string): number | undefined {
  if (v === undefined || v === null) return undefined
  if (typeof v !== 'number' || !Number.isFinite(v)) throw new Error(`"${name}" must be a number`)
  return v
}

function transition(t: { type: TransitionType; duration?: number } | undefined, name: string) {
  if (!t) return undefined
  if (!transitionTypeValues.includes(t.type)) {
    throw new Error(`${name}.type must be one of: ${transitionTypeValues.join(', ')}`)
  }
  return { type: t.type, duration: t.duration ?? 0.5 }
}

function defaultTrack(state: EditorState, kind: 'video' | 'audio'): number {
  const idx = selectTracks(state).findIndex(t => t.kind === kind && t.type !== 'subtitle' && !t.locked)
  if (idx < 0) throw new Error(`No unlocked ${kind} track — add one with {"op":"add_track","kind":"${kind}"}`)
  return idx
}

async function applyEdits(args: { ops?: EditOp[] } & Record<string, unknown>) {
  const entry = requireProject(args)
  const ops = args.ops
  if (!Array.isArray(ops) || ops.length === 0) throw new Error('"ops" must be a non-empty array')

  // Phase 1 (async): import every file the batch references, before touching state.
  const refs = new Map<string, string>() // "$name" -> asset or clip id
  const importedIds = new Map<number, string>() // op index -> asset id
  const imported: Asset[] = []
  const known = () => [...imported, ...selectAssets(entry.store.getState().state)]
  for (const [i, op] of ops.entries()) {
    try {
      if (op.op === 'import') {
        const asset = await importPath(op.path, entry, known())
        if (!imported.includes(asset) && !selectAssets(entry.store.getState().state).includes(asset)) imported.push(asset)
        importedIds.set(i, asset.id)
        if (op.ref) refs.set(`$${op.ref.replace(/^\$/, '')}`, asset.id)
      } else if (op.op === 'add_clip' && looksLikePath(op.asset)) {
        const asset = await importPath(op.asset, entry, known())
        if (!imported.includes(asset) && !selectAssets(entry.store.getState().state).includes(asset)) imported.push(asset)
        refs.set(op.asset, asset.id)
      }
    } catch (e) {
      throw new Error(`op ${i} (${op.op}): import failed — ${e instanceof Error ? e.message : String(e)}`)
    }
  }

  // Phase 2 (sync): fold all ops into one state transition = one undo step.
  const results: Array<Record<string, unknown>> = []
  const outcome: { failure: Error | null } = { failure: null }
  entry.store.getState().setStateWithHistory(prev => {
    try {
      let s = imported.length > 0 ? actions.addAssetsToEditor(prev, imported) : prev
      const resolveId = (id: string) => refs.get(id) ?? refs.get(`$${id.replace(/^\$/, '')}`) ?? id
      const findClip = (id: string) => {
        const clip = selectClips(s).find(c => c.id === resolveId(id))
        if (!clip) throw new Error(`clip "${id}" not found on the active timeline`)
        return clip
      }

      for (const [i, op] of ops.entries()) {
        try {
          switch (op.op) {
            case 'import': {
              results.push({ op: 'import', assetId: importedIds.get(i), path: op.path })
              break
            }
            case 'add_clip': {
              const assetId = resolveId(op.asset)
              const asset = selectAssets(s).find(a => a.id === assetId)
              if (!asset) throw new Error(`asset "${op.asset}" not found (use an asset id, a $ref, or an absolute file path)`)
              const trackIndex = op.track ?? defaultTrack(s, asset.type === 'audio' ? 'audio' : 'video')
              const clipsBefore = selectClips(s)
              const start = num(op.start, 'start')
                ?? clipsBefore.filter(c => c.trackIndex === trackIndex).reduce((m, c) => Math.max(m, c.startTime + c.duration), 0)
              const insertion = actions.buildDroppedAssetInsertion(asset, trackIndex, start, selectTracks(s))
              if (insertion.clips.length === 0) {
                throw new Error(`cannot place a ${asset.type} on track ${trackIndex} (wrong kind, locked, or not patched)`)
              }
              const speed = num(op.speed, 'speed') ?? 1
              const trimIn = asset.type === 'image' ? 0 : Math.max(0, num(op.in, 'in') ?? 0)
              const srcDur = asset.duration ?? insertion.duration
              const maxDur = asset.type === 'image' ? Infinity : Math.max(0.1, (srcDur - trimIn) / speed)
              const duration = Math.min(num(op.duration, 'duration') ?? maxDur, maxDur)
              if (!Number.isFinite(duration) || duration <= 0) throw new Error('duration must be > 0')
              let placed = insertion.clips.map(c => ({
                ...c,
                speed,
                trimStart: trimIn,
                duration,
                trimEnd: asset.type === 'image' ? 0 : Math.max(0, srcDur - trimIn - duration * speed),
              }))
              if (op.with_audio === false && asset.type === 'video') {
                const audioIds = new Set(placed.filter(c => c.type === 'audio').map(c => c.id))
                placed = placed.filter(c => !audioIds.has(c.id)).map(c => ({ ...c, linkedClipIds: undefined }))
              }
              let tracksAfter = insertion.tracks
              if (op.audio_track !== undefined && asset.type === 'video') {
                // Route the linked dialogue to a chosen audio track (e.g. A2) so it can't
                // overwrite the music bed on A1.
                if (tracksAfter[op.audio_track]?.kind !== 'audio' || tracksAfter[op.audio_track]?.locked) {
                  throw new Error(`audio_track ${op.audio_track} is not an unlocked audio track — add one with {"op":"add_track","kind":"audio"}`)
                }
                // Drop an audio track the insertion auto-created if we're not using it.
                const autoIdx = tracksAfter.length > selectTracks(s).length ? tracksAfter.length - 1 : -1
                placed = placed.map(c => (c.type === 'audio' ? { ...c, trackIndex: op.audio_track! } : c))
                if (autoIdx >= 0 && autoIdx !== op.audio_track) tracksAfter = tracksAfter.slice(0, autoIdx)
              }
              const placedIds = new Set(placed.map(c => c.id))
              s = actions.setTimelineTracks(s, tracksAfter)
              s = actions.setTimelineClips(s, clips => resolveOverlaps([...clips, ...placed], placedIds))
              const main = placed.find(c => c.type !== 'audio') ?? placed[0]
              if (op.ref) {
                const name = op.ref.replace(/^\$/, '')
                refs.set(`$${name}`, main.id)
                // "$name.audio" addresses the linked dialogue clip (fades, volume) on its own.
                const linkedAudio = placed.find(c => c !== main && c.type === 'audio')
                if (linkedAudio) refs.set(`$${name}.audio`, linkedAudio.id)
              }
              results.push({ op: 'add_clip', clipId: main.id, linked: placed.filter(c => c !== main).map(c => c.id), start: round(start), end: round(start + duration) })
              break
            }
            case 'add_text': {
              if (!op.text) throw new Error('"text" is required')
              const start = num(op.start, 'start') ?? 0
              const duration = num(op.duration, 'duration') ?? 3
              // Titles must never share a track with picture: placing onto a track overwrites
              // whatever it covers. Default to a video track holding only text that's free
              // over [start, end), else add a new video track for titles.
              let track = op.track
              if (track === undefined) {
                const clips = selectClips(s)
                track = selectTracks(s).findIndex((t, idx) => t.kind === 'video' && t.type !== 'subtitle' && !t.locked
                  && clips.every(c => c.trackIndex !== idx
                    || (c.type === 'text' && (c.startTime + c.duration <= start || c.startTime >= start + duration))))
                if (track < 0) {
                  s = actions.addTrack(s, 'video')
                  track = selectTracks(s).length - 1
                }
              }
              const added: TimelineClip = {
                ...actions.createTextClip({ ...(op.style ?? {}), text: op.text }, start, track),
                duration,
                ...(op.fade_in !== undefined ? { textFadeIn: op.fade_in } : {}),
                ...(op.fade_out !== undefined ? { textFadeOut: op.fade_out } : {}),
              }
              s = actions.setTimelineClips(s, clips => resolveOverlaps([...clips, added], new Set([added.id])))
              if (op.ref) refs.set(`$${op.ref.replace(/^\$/, '')}`, added.id)
              results.push({ op: 'add_text', clipId: added.id, track: added.trackIndex, start, end: round(start + duration) })
              break
            }
            case 'update_clip': {
              const clip = findClip(op.clip)
              const set = op.set ?? {}
              const patch: Partial<TimelineClip> = {}
              const pick = (key: string) => num(set[key], key)
              if (pick('start') !== undefined) patch.startTime = Math.max(0, pick('start')!)
              if (pick('duration') !== undefined) patch.duration = Math.max(0.1, pick('duration')!)
              if (pick('in') !== undefined) patch.trimStart = Math.max(0, pick('in')!)
              if (pick('speed') !== undefined) patch.speed = Math.max(0.01, pick('speed')!)
              if (pick('volume') !== undefined) patch.volume = Math.max(0, Math.min(2, pick('volume')!))
              if (pick('opacity') !== undefined) patch.opacity = Math.max(0, Math.min(100, pick('opacity')!))
              if (pick('audio_fade_in') !== undefined) patch.audioFadeIn = Math.max(0, pick('audio_fade_in')!)
              if (pick('audio_fade_out') !== undefined) patch.audioFadeOut = Math.max(0, pick('audio_fade_out')!)
              if (pick('text_fade_in') !== undefined) patch.textFadeIn = Math.max(0, pick('text_fade_in')!)
              if (pick('text_fade_out') !== undefined) patch.textFadeOut = Math.max(0, pick('text_fade_out')!)
              if (Array.isArray(set.volume_keyframes)) {
                patch.volumeKeyframes = (set.volume_keyframes as Array<{ t: unknown; value: unknown }>).map((k, idx) => {
                  const t = num(k?.t, `volume_keyframes[${idx}].t`)
                  const value = num(k?.value, `volume_keyframes[${idx}].value`)
                  if (t === undefined || value === undefined) throw new Error(`volume_keyframes[${idx}] needs numeric t and value`)
                  return { t: Math.max(0, t), value: Math.max(0, Math.min(2, value)) }
                }).sort((a, b) => a.t - b.t)
              }
              if (typeof set.muted === 'boolean') patch.muted = set.muted
              if (typeof set.reversed === 'boolean') patch.reversed = set.reversed
              if (set.color && typeof set.color === 'object') patch.colorCorrection = { ...clip.colorCorrection, ...(set.color as object) }
              if (clip.type === 'text' && (typeof set.text === 'string' || (set.style && typeof set.style === 'object'))) {
                patch.textStyle = {
                  ...(clip.textStyle ?? DEFAULT_TEXT_STYLE),
                  ...((set.style as Partial<TextOverlayStyle>) ?? {}),
                  ...(typeof set.text === 'string' ? { text: set.text } : {}),
                }
              }
              const unknown = Object.keys(set).filter(k => ![
                'start', 'duration', 'in', 'speed', 'volume', 'opacity', 'audio_fade_in', 'audio_fade_out',
                'text_fade_in', 'text_fade_out', 'muted', 'reversed', 'color', 'text', 'style', 'volume_keyframes',
              ].includes(k))
              if (unknown.length) throw new Error(`unknown field(s) in "set": ${unknown.join(', ')}`)
              const next = { ...clip, ...patch }
              if (next.asset && next.type !== 'image' && next.type !== 'text' && next.asset.duration) {
                next.trimEnd = Math.max(0, next.asset.duration - next.trimStart - next.duration * next.speed)
              }
              s = actions.updateClip(s, clip.id, next)
              // Keep a video's linked audio in step on timing edits.
              const timing: Partial<TimelineClip> = {}
              for (const k of ['startTime', 'duration', 'trimStart', 'trimEnd', 'speed', 'reversed'] as const) {
                if (k in patch || k === 'trimEnd') (timing as Record<string, unknown>)[k] = next[k]
              }
              // (A fades/volume-only edit on one half of a linked pair stays on that half.)
              if (Object.keys(timing).some(k => k !== 'trimEnd')) {
                for (const linkedId of clip.linkedClipIds ?? []) s = actions.updateClip(s, linkedId, timing)
              }
              results.push({ op: 'update_clip', clipId: clip.id })
              break
            }
            case 'duck': {
              // Dip a music bed under dialogue: ranges are TIMELINE seconds of speech.
              // Builds a volume envelope (replacing any existing one): base → level over
              // `attack` before each range, back up over `release` after it. Ranges whose
              // ramps would touch are merged so the music doesn't pump between lines.
              const clip = findClip(op.clip)
              if (clip.type !== 'audio' && clip.type !== 'video') throw new Error('duck target must be an audio or video clip')
              const base = clip.volume ?? 1
              const level = Math.max(0, Math.min(1, num(op.level, 'level') ?? 0.25)) * base
              const attack = Math.max(0.02, num(op.attack, 'attack') ?? 0.2)
              const release = Math.max(0.02, num(op.release, 'release') ?? 0.4)
              if (!Array.isArray(op.ranges) || op.ranges.length === 0) throw new Error('"ranges" must be a non-empty array of [start, end] timeline seconds')
              const ranges = op.ranges
                .map(([a, b], idx) => {
                  const start = num(a, `ranges[${idx}][0]`)!, end = num(b, `ranges[${idx}][1]`)!
                  if (!(end > start)) throw new Error(`ranges[${idx}] must have end > start`)
                  return [start - clip.startTime, end - clip.startTime] as [number, number]
                })
                .sort((x, y) => x[0] - y[0])
              const merged: Array<[number, number]> = []
              for (const r of ranges) {
                const last = merged[merged.length - 1]
                if (last && r[0] - attack <= last[1] + release) last[1] = Math.max(last[1], r[1])
                else merged.push([...r])
              }
              const clampT = (t: number) => Math.max(0, Math.min(clip.duration, t))
              const keys: Array<{ t: number; value: number }> = [{ t: 0, value: base }]
              for (const [a, b] of merged) {
                keys.push({ t: clampT(a - attack), value: base }, { t: clampT(a), value: level },
                  { t: clampT(b), value: level }, { t: clampT(b + release), value: base })
              }
              keys.push({ t: clip.duration, value: base })
              // Collapse duplicate times (ramps clamped at the clip edges), keeping the last value.
              const volumeKeyframes = keys
                .sort((x, y) => x.t - y.t)
                .filter((k, idx, arr) => idx === arr.length - 1 || arr[idx + 1].t - k.t > 1e-3)
                .map(k => ({ t: round(k.t), value: round(k.value) }))
              s = actions.updateClip(s, clip.id, { volumeKeyframes })
              results.push({ op: 'duck', clipId: clip.id, dips: merged.map(([a, b]) => [round(a + clip.startTime), round(b + clip.startTime)]), level: round(level) })
              break
            }
            case 'transition': {
              const clip = findClip(op.clip)
              const tIn = transition(op.in, 'in')
              const tOut = transition(op.out, 'out')
              s = actions.updateClip(s, clip.id, { ...(tIn ? { transitionIn: tIn } : {}), ...(tOut ? { transitionOut: tOut } : {}) })
              results.push({ op: 'transition', clipId: clip.id })
              break
            }
            case 'delete': {
              const ids = (op.clips ?? []).map(id => findClip(id).id)
              const withLinked = new Set(ids)
              for (const id of ids) for (const l of selectClips(s).find(c => c.id === id)?.linkedClipIds ?? []) withLinked.add(l)
              s = actions.deleteClips(s, [...withLinked])
              results.push({ op: 'delete', deleted: [...withLinked] })
              break
            }
            case 'clear': {
              const all = selectClips(s).map(c => c.id)
              s = actions.deleteClips(s, all)
              results.push({ op: 'clear', deleted: all.length })
              break
            }
            case 'new_timeline': {
              s = actions.createTimeline(s, op.name)
              results.push({ op: 'new_timeline', timelineId: selectActiveTimeline(s)?.id })
              break
            }
            case 'add_track': {
              if (op.kind !== 'video' && op.kind !== 'audio') throw new Error('kind must be "video" or "audio"')
              s = actions.addTrack(s, op.kind)
              results.push({ op: 'add_track', track: selectTracks(s).length - 1, kind: op.kind })
              break
            }
            default:
              throw new Error(`unknown op "${(op as { op?: string }).op}"`)
          }
        } catch (e) {
          throw new Error(`op ${i} (${(op as { op?: string }).op}): ${e instanceof Error ? e.message : String(e)}`)
        }
      }
      return s
    } catch (e) {
      outcome.failure = e instanceof Error ? e : new Error(String(e))
      return prev // atomic: nothing applied
    }
  })
  if (outcome.failure) throw new Error(`${outcome.failure.message} — no edits were applied.`)
  mcpUndoable.push({ store: entry.store, model: entry.store.getState().state.editorModel })
  mcpRedoable.length = 0
  return { applied: ops.length, results, timeline: getTimeline() }
}

// ---------------------------------------------------------------- dispatcher

export async function handleMcpEditorRequest(tool: string, args: Record<string, unknown>): Promise<unknown> {
  switch (tool) {
    case 'status': return status()
    case 'list_project_assets': return listProjectAssets(args)
    case 'get_timeline': return getTimeline()
    case 'export_payload': return exportPayload()
    case 'apply_edits': return applyEdits(args as { ops?: EditOp[] } & Record<string, unknown>)
    case 'undo': return stepHistory(args, 'undo')
    case 'redo': return stepHistory(args, 'redo')
    default: throw new Error(`Unknown editor tool "${tool}"`)
  }
}
registry.handler = handleMcpEditorRequest
