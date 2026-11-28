import type { components } from '../generated/backend-openapi'

import {
  isConfirmableBudgetTier,
  worseRisk,
  type BudgetMode,
  type SelectionRisk,
  type SelectionTier,
} from './budget-risk.ts'

export type VideoGenerationModelSpecsResponse = components['schemas']['GenerateVideoModelsSpecsResponse']
export type VideoGenerationModelSpecItem = components['schemas']['LTXVideoGenerationModelSpecItem']
export type DownloadedLocalVideoGenerationModelSpecItem =
  components['schemas']['DownloadedLocalVideoGenerationModelSpecItem']
export type OfferingId = DownloadedLocalVideoGenerationModelSpecItem['model']
export type VideoGenerationResolutionSpec = components['schemas']['LTXVideoGenerationResolutionSpec']
export type VideoGenerationAspectSpec = components['schemas']['LTXVideoGenerationAspectSpec']
export type VideoGenerationOfferingCapabilities = NonNullable<
  VideoGenerationModelSpecItem['spec']['capabilities']
>
export type VideoGenerationPipeline = components['schemas']['GenerateVideoRequest']['model']
export type VideoGenerationResolution = components['schemas']['GenerateVideoRequest']['resolution']
export type VideoGenerationDuration = Exclude<
  components['schemas']['GenerateVideoRequest']['duration'],
  null
>
export type VideoGenerationFps = components['schemas']['GenerateVideoRequest']['fps']
export type VideoGenerationAspectRatio = components['schemas']['GenerateVideoRequest']['aspectRatio']
export type DroppedFastCell = components['schemas']['DroppedFastCell']
export type DroppedFastAspectCell = components['schemas']['DroppedFastAspectCell']
export type WarnedFastCell = components['schemas']['WarnedFastCell']
export type ChunkedFastCell = components['schemas']['ChunkedFastCell']
export type WarnedFastAspectCell = components['schemas']['WarnedFastAspectCell']
export type FastCellAsk = components['schemas']['FastCellAsk']
export type MemoryCapacity = components['schemas']['MemoryCapacity']
export type JobMemoryAsk = components['schemas']['JobMemoryAsk']
export type { BudgetMode, SelectionRisk, SelectionTier }

/** Canonical resolution order for merged kept+disabled pickers. */
export const VIDEO_GENERATION_RESOLUTION_ORDER: readonly VideoGenerationResolution[] = [
  '270p',
  '360p',
  '540p',
  '720p',
  '1080p',
  '1440p',
  '2160p',
]

/** A withheld option with its backend-composed refusal to show as a tooltip. */
export type DisabledVideoGenerationOption<T> = {
  value: T
  message: string
  readout?: DroppedFastCell['readout']
}

/** A selectable option carrying a budget-risk tooltip. */
export type WarnedVideoGenerationOption<T> = {
  value: T
  message: string
  tier: SelectionTier
  readout?: SelectionRisk['readout']
}

/** How a job runs when the backend splits it into overlapping windows (more than one pass). */
export type ChunkedRun = {
  windows: number
  windowFrames: number
}

/** A selectable option whose job runs in more than one overlapping window. */
export type ChunkedVideoGenerationOption<T> = {
  value: T
  windows: number
  windowFrames: number
}

/** Shared local-video catalog. Features pick a subset of keys; models restrict values. */
export type VideoGenerationFormSettings = {
  model: VideoGenerationPipeline
  aspectRatio: VideoGenerationAspectRatio
  resolution: VideoGenerationResolution
  duration: VideoGenerationDuration
  fps: VideoGenerationFps
}

export interface VideoGenerationSettingsShape {
  model: string
  duration: number | null
  videoResolution: string
  fps: number
  aspectRatio?: string
  audio?: boolean
}

export const VIDEO_GENERATION_ASPECT_RATIOS = [
  '21:9',
  '16:9',
  '3:2',
  '4:3',
  '1:1',
  '4:5',
  '9:16',
] as const satisfies readonly VideoGenerationAspectRatio[]

/** Distilled models were trained on this pair. Other shapes still run, with a quality caveat. */
export const TRAINED_VIDEO_ASPECT_RATIOS = ['16:9', '9:16'] as const satisfies readonly VideoGenerationAspectRatio[]

export const UNTRAINED_ASPECT_RATIO_WARNING =
  'The model was trained on 16:9 and 9:16, other ratios may produce lower quality results.'

export function isTrainedVideoAspectRatio(value: string): boolean {
  return (TRAINED_VIDEO_ASPECT_RATIOS as readonly string[]).includes(value)
}

/** Trained pair first, then the rest in the cell's order. */
export function orderAspectRatiosForPicker<T extends string>(
  ratios: readonly T[],
): T[] {
  const trained = TRAINED_VIDEO_ASPECT_RATIOS.filter((ratio) =>
    (ratios as readonly string[]).includes(ratio),
  ) as T[]
  const rest = ratios.filter((ratio) => !isTrainedVideoAspectRatio(ratio))
  return [...trained, ...rest]
}

export function untrainedAspectRatioWarning(
  value: string,
): string | undefined {
  if (value === 'auto' || isTrainedVideoAspectRatio(value)) return undefined
  return UNTRAINED_ASPECT_RATIO_WARNING
}

export function isVideoGenerationAspectRatio(
  value: unknown,
): value is VideoGenerationAspectRatio {
  return (VIDEO_GENERATION_ASPECT_RATIOS as readonly unknown[]).includes(value)
}

export interface ResolvedVideoGenerationOptions {
  modelOptions: VideoGenerationModelSpecItem[]
  resolutionOptions: VideoGenerationResolution[]
  fpsOptions: VideoGenerationFps[]
  durationOptions: VideoGenerationDuration[]
  /** Withheld options for disabled-with-tooltip pickers. Display-only: selections and clamps stay kept-only. */
  disabledResolutionOptions: DisabledVideoGenerationOption<VideoGenerationResolution>[]
  disabledFpsOptions: DisabledVideoGenerationOption<VideoGenerationFps>[]
  disabledDurationOptions: DisabledVideoGenerationOption<VideoGenerationDuration>[]
  /** Shapes the current cell cannot run (cell-scoped, unlike the lists above). Empty while the cell is unknown. */
  disabledAspectRatioOptions: DisabledVideoGenerationOption<VideoGenerationAspectRatio>[]
  /** Selectable cells with a server-composed budget warning (tight, or a Brave bet). */
  warnedResolutionOptions: WarnedVideoGenerationOption<VideoGenerationResolution>[]
  warnedFpsOptions: WarnedVideoGenerationOption<VideoGenerationFps>[]
  warnedDurationOptions: WarnedVideoGenerationOption<VideoGenerationDuration>[]
  warnedAspectRatioOptions: WarnedVideoGenerationOption<VideoGenerationAspectRatio>[]
  /**
   * Selectable cells the backend runs in overlapping windows, per picker. Resolution and fps
   * options are judged at the selected duration, so both are empty while the duration is
   * automatic. Brave and Cautious alike: an offered cell with no entry runs in one pass.
   * A resolution that does not offer the selected fps and duration has no entry either, and
   * is not a single pass: choosing it changes the duration, so its windows are unknown.
   */
  chunkedResolutionOptions: ChunkedVideoGenerationOption<VideoGenerationResolution>[]
  /** Resolutions that offer the selected fps and duration and run them in one pass. */
  singlePassResolutionOptions: VideoGenerationResolution[]
  chunkedFpsOptions: ChunkedVideoGenerationOption<VideoGenerationFps>[]
  chunkedDurationOptions: ChunkedVideoGenerationOption<VideoGenerationDuration>[]
  aspectRatioOptions: VideoGenerationAspectRatio[]
  selectedModel: VideoGenerationPipeline | null
  selectedResolution: VideoGenerationResolution | null
  selectedFps: VideoGenerationFps | null
  selectedDuration: VideoGenerationDuration | null
  selectedAspectRatio: VideoGenerationAspectRatio | null
  autoDurationAvailable: boolean
  hasCompatibleOptions: boolean
}

type DurationSelectionMode = 'preserve' | 'smallest_valid'

/** GenSpace picker floor. The API envelope includes 2–5s so gap fill can request shorts. */
export const GENSPACE_MIN_SELECTABLE_DURATION_S = 2

interface ResolveVideoGenerationOptionsParams<T extends VideoGenerationSettingsShape> {
  settings: T
  modelSpecs: VideoGenerationModelSpecItem[]
  hasAudio?: boolean
  minimumDuration?: number
  durationSelection?: DurationSelectionMode
}

function getResolutionMap(
  item: VideoGenerationModelSpecItem,
  options: { hasAudio: boolean },
): Record<string, VideoGenerationResolutionSpec> {
  const { hasAudio } = options
  if (!hasAudio) {
    return item.spec.supported_resolutions_durations
  }
  // A model with no a2v spec doesn't support audio-conditioned generation at all —
  // must not fall back to the plain (non-a2v) matrix, or it looks compatible when it isn't.
  return item.spec.a2v_supported_resolutions_durations ?? {}
}

function getResolutionEntries(
  item: VideoGenerationModelSpecItem,
  options: { hasAudio: boolean },
): Array<[VideoGenerationResolution, VideoGenerationResolutionSpec]> {
  return Object.entries(getResolutionMap(item, options)).map(([resolution, spec]) => [
    resolution as VideoGenerationResolution,
    spec,
  ])
}

function getDurationsForFps(
  resolutionSpec: VideoGenerationResolutionSpec,
  fps: VideoGenerationFps,
): VideoGenerationDuration[] {
  return (resolutionSpec.fps_to_durations[String(fps)] ?? []) as VideoGenerationDuration[]
}

function filterDurationsByMinimum(
  durations: VideoGenerationDuration[],
  minimumDuration: number | undefined,
): VideoGenerationDuration[] {
  if (minimumDuration === undefined) return durations
  return durations.filter((duration) => duration >= minimumDuration)
}

function getCompatibleFps(
  resolutionSpec: VideoGenerationResolutionSpec,
  options: { minimumDuration: number | undefined },
): VideoGenerationFps[] {
  const { minimumDuration } = options
  return Object.keys(resolutionSpec.fps_to_durations).map((fps) => Number(fps) as VideoGenerationFps).filter((fps) => (
    filterDurationsByMinimum(getDurationsForFps(resolutionSpec, fps), minimumDuration).length > 0
  ))
}

function getCompatibleResolutionEntries(
  item: VideoGenerationModelSpecItem,
  options: { hasAudio: boolean; minimumDuration: number | undefined },
): Array<[VideoGenerationResolution, VideoGenerationResolutionSpec]> {
  return getResolutionEntries(item, { hasAudio: options.hasAudio }).filter(([, resolutionSpec]) => (
    getCompatibleFps(resolutionSpec, { minimumDuration: options.minimumDuration }).length > 0
  ))
}

function getCompatibleModelOptions(
  modelSpecs: VideoGenerationModelSpecItem[],
  options: { hasAudio: boolean; minimumDuration: number | undefined },
): VideoGenerationModelSpecItem[] {
  const { hasAudio, minimumDuration } = options
  // Always filter by resolution compatibility — hasAudio alone (independent of any
  // minimumDuration constraint) can exclude a model, e.g. a fast-tier pipeline with no
  // a2v spec. Skipping this whenever minimumDuration is unset used to let incompatible
  // (audio-unsupported) models stay selectable and get stuck with no valid resolution.
  return modelSpecs.filter((item) => (
    getCompatibleResolutionEntries(item, { hasAudio, minimumDuration }).length > 0
  ))
}

function emptyResolvedOptions(
  modelOptions: VideoGenerationModelSpecItem[],
  extras: Partial<ResolvedVideoGenerationOptions> = {},
): ResolvedVideoGenerationOptions {
  return {
    modelOptions,
    resolutionOptions: [],
    fpsOptions: [],
    durationOptions: [],
    disabledResolutionOptions: [],
    disabledFpsOptions: [],
    disabledDurationOptions: [],
    disabledAspectRatioOptions: [],
    warnedResolutionOptions: [],
    warnedFpsOptions: [],
    warnedDurationOptions: [],
    warnedAspectRatioOptions: [],
    chunkedResolutionOptions: [],
    singlePassResolutionOptions: [],
    chunkedFpsOptions: [],
    chunkedDurationOptions: [],
    aspectRatioOptions: [],
    selectedModel: null,
    selectedResolution: null,
    selectedFps: null,
    selectedDuration: null,
    selectedAspectRatio: null,
    autoDurationAvailable: false,
    hasCompatibleOptions: false,
    ...extras,
  }
}

/** Dropped cells for one item in the active matrix (plain or A2V). */
function droppedCellsForItem(
  item: VideoGenerationModelSpecItem | null | undefined,
  hasAudio: boolean,
): DroppedFastCell[] {
  if (!item) return []
  return (item.spec.dropped_cells ?? []).filter((cell) => cell.audio === hasAudio)
}

const RESOLUTION_ORDER_INDEX = new Map<VideoGenerationResolution, number>(
  VIDEO_GENERATION_RESOLUTION_ORDER.map((resolution, index) => [resolution, index]),
)

/**
 * Refusal of the cheapest dropped cell: smallest resolution, then duration,
 * then fps. One order for every field tooltip (resolution, fps, union):
 * resolution dominates tokens, duration varies more than fps. Field
 * tooltips have no exact cell — the smallest names the binding constraint.
 */
function cheapestDroppedCell(cells: readonly DroppedFastCell[]): DroppedFastCell | undefined {
  return [...cells].sort(
    (a, b) =>
      (RESOLUTION_ORDER_INDEX.get(a.resolution) ?? 99) -
        (RESOLUTION_ORDER_INDEX.get(b.resolution) ?? 99) ||
      a.duration - b.duration ||
      a.fps - b.fps,
  )[0]
}

export function cheapestDroppedCellMessage(cells: readonly DroppedFastCell[]): string {
  return cheapestDroppedCell(cells)?.message ?? ''
}

/**
 * Cells keyed by `key`, skipping kept values. Grouping primitive behind
 * `disabledOptionsFromDroppedCells`, which owns the per-group message and
 * the sort.
 */
function groupDroppedCells<T extends string | number, C extends DroppedFastCell>(
  cells: readonly C[],
  key: (cell: C) => T,
  kept: readonly T[],
): Map<T, C[]> {
  const keptSet = new Set<T>(kept)
  const groups = new Map<T, C[]>()
  for (const cell of cells) {
    const value = key(cell)
    if (keptSet.has(value)) continue
    const group: C[] = groups.get(value) ?? []
    group.push(cell)
    groups.set(value, group)
  }
  return groups
}

/**
 * Withheld options from dropped cells: grouped by `key`, skipping kept
 * values, each messaged from its cheapest cell via
 * `cheapestDroppedCellMessage`, sorted by `compare`. Single home for every
 * disabled-with-tooltip list — one cheapest-cell order for every picker
 * scope, so the grouping and the message share it.
 */
export function disabledOptionsFromDroppedCells<T extends string | number, C extends DroppedFastCell>(
  cells: readonly C[],
  key: (cell: C) => T,
  kept: readonly T[],
  compare: (a: T, b: T) => number,
): DisabledVideoGenerationOption<T>[] {
  const groups = groupDroppedCells(cells, key, kept)
  return [...groups.entries()]
    .map(([value, group]) => {
      const cell = cheapestDroppedCell(group)
      return {
        value,
        message: cell?.message ?? '',
        ...(cell?.readout ? { readout: cell.readout } : {}),
      }
    })
    .sort((a, b) => compare(a.value, b.value))
}

function disabledResolutionsForItem(
  item: VideoGenerationModelSpecItem | null | undefined,
  hasAudio: boolean,
  kept: readonly VideoGenerationResolution[],
): DisabledVideoGenerationOption<VideoGenerationResolution>[] {
  return disabledOptionsFromDroppedCells(
    droppedCellsForItem(item, hasAudio),
    (cell) => cell.resolution,
    kept,
    (a, b) =>
      (RESOLUTION_ORDER_INDEX.get(a) ?? 99) -
      (RESOLUTION_ORDER_INDEX.get(b) ?? 99),
  )
}

function disabledFpsForItem(
  item: VideoGenerationModelSpecItem | null | undefined,
  hasAudio: boolean,
  resolution: VideoGenerationResolution,
  keptFps: readonly VideoGenerationFps[],
): DisabledVideoGenerationOption<VideoGenerationFps>[] {
  return disabledOptionsFromDroppedCells(
    droppedCellsForItem(item, hasAudio).filter(
      (cell) => cell.resolution === resolution,
    ),
    (cell) => cell.fps,
    keptFps,
    (a, b) => a - b,
  )
}

function disabledDurationsForItem(
  item: VideoGenerationModelSpecItem | null | undefined,
  hasAudio: boolean,
  resolution: VideoGenerationResolution,
  fps: VideoGenerationFps,
  minimumDuration: number | undefined,
): DisabledVideoGenerationOption<VideoGenerationDuration>[] {
  return droppedCellsForItem(item, hasAudio)
    .filter(
      (cell) =>
        cell.resolution === resolution &&
        cell.fps === fps &&
        (minimumDuration === undefined || cell.duration >= minimumDuration),
    )
    .map((cell) => ({
      value: cell.duration,
      message: cell.message,
      ...(cell.readout ? { readout: cell.readout } : {}),
    }))
    .sort((a, b) => a.value - b.value)
}

const ASPECT_CANVAS_WEIGHT: Record<VideoGenerationAspectRatio, number> = {
  '1:1': 1,
  '4:5': 5 / 4,
  '4:3': 4 / 3,
  '3:2': 3 / 2,
  '16:9': 16 / 9,
  '9:16': 16 / 9,
  '21:9': 21 / 9,
}

export function specForAspect(
  spec: VideoGenerationModelSpecItem['spec'],
  aspectRatio: string | null | undefined,
): VideoGenerationModelSpecItem['spec'] {
  if (!aspectRatio || !isVideoGenerationAspectRatio(aspectRatio)) return spec
  const aspectSpec = spec.aspect_specs?.[aspectRatio]
  if (!aspectSpec) return spec
  return {
    ...spec,
    supported_resolutions_durations: aspectSpec.supported_resolutions_durations,
    a2v_supported_resolutions_durations: aspectSpec.a2v_supported_resolutions_durations,
    dropped_cells: aspectSpec.dropped_cells ?? [],
    warned_cells: aspectSpec.warned_cells ?? [],
    chunked_cells: aspectSpec.chunked_cells ?? [],
    cell_asks: aspectSpec.cell_asks ?? [],
  }
}

export function itemForAspect(
  item: VideoGenerationModelSpecItem,
  aspectRatio: string | null | undefined,
): VideoGenerationModelSpecItem {
  const nextSpec = specForAspect(item.spec, aspectRatio)
  return nextSpec === item.spec ? item : { ...item, spec: nextSpec }
}

/**
 * Shapes the picked cell cannot run. Cell-scoped (unlike the union lists
 * above): a 21:9 refusal at 20s must not follow the picker to 5s. Empty while
 * the cell is unknown (no duration yet, auto duration) — unknown stays
 * enabled and create-time enforcement names the refusal instead.
 */
function disabledAspectsForItem(
  item: VideoGenerationModelSpecItem | null | undefined,
  hasAudio: boolean,
  resolution: VideoGenerationResolution,
  fps: VideoGenerationFps,
  duration: VideoGenerationDuration | null,
  aspectRatios: readonly VideoGenerationAspectRatio[],
): DisabledVideoGenerationOption<VideoGenerationAspectRatio>[] {
  if (!item || duration == null) return []
  const aspectSpecs = item.spec.aspect_specs
  if (aspectSpecs) {
    const out: DisabledVideoGenerationOption<VideoGenerationAspectRatio>[] = []
    for (const ratio of aspectRatios) {
      const aspectSpec = aspectSpecs[ratio]
      if (!aspectSpec) continue
      const cell = (aspectSpec.dropped_cells ?? []).find(
        (candidate) =>
          candidate.audio === hasAudio &&
          candidate.resolution === resolution &&
          candidate.fps === fps &&
          candidate.duration === duration,
      )
      if (cell) {
        out.push({
          value: ratio,
          message: cell.message,
          ...(cell.readout ? { readout: cell.readout } : {}),
        })
      }
    }
    return out
  }
  return (item.spec.dropped_aspect_cells ?? [])
    .filter(
      (cell) =>
        cell.audio === hasAudio &&
        cell.resolution === resolution &&
        cell.fps === fps &&
        cell.duration === duration,
    )
    .map((cell) => ({
      value: cell.aspectRatio,
      message: cell.message,
      ...(cell.readout ? { readout: cell.readout } : {}),
    }))
}

function warnedAspectsForItem(
  item: VideoGenerationModelSpecItem,
  grid: FastGridWarnings,
  hasAudio: boolean,
  resolution: VideoGenerationResolution,
  fps: VideoGenerationFps,
  duration: VideoGenerationDuration | null,
  selectedAspectRatio: VideoGenerationAspectRatio | null,
  aspectRatios: readonly VideoGenerationAspectRatio[],
): WarnedVideoGenerationOption<VideoGenerationAspectRatio>[] {
  if (duration == null) return []
  const aspectSpecs = item.spec.aspect_specs
  if (aspectSpecs && selectedAspectRatio != null) {
    const baseWeight = ASPECT_CANVAS_WEIGHT[selectedAspectRatio]
    const rows = new Map<VideoGenerationAspectRatio, SelectionRisk>()
    for (const ratio of aspectRatios) {
      if (ASPECT_CANVAS_WEIGHT[ratio] <= baseWeight) continue
      const aspectSpec = aspectSpecs[ratio]
      if (!aspectSpec) continue
      for (const cell of aspectSpec.warned_cells ?? []) {
        if (
          cell.audio === hasAudio &&
          cell.resolution === resolution &&
          cell.fps === fps &&
          cell.duration === duration
        ) {
          const prev = rows.get(ratio) ?? null
          const next = worseRisk(prev, {
            message: cell.message,
            tier: cell.tier,
            readout: cell.readout,
          })
          if (next) rows.set(ratio, next)
        }
      }
    }
    return warnedFromRows(rows)
  }
  return warnedFromRows(grid.aspectWarnings.get(`${resolution}|${fps}|${duration}`))
}

function chooseOption<T>(current: string | number | null, options: T[]): T | null {
  return options.find((option) => option === current) ?? options[0] ?? null
}

export type OfferingVideoGenerationModelSpecItem = {
  model: OfferingId
  spec: VideoGenerationModelSpecItem["spec"]
}

export function getVideoGenerationModelSpecs(
  specs: VideoGenerationModelSpecsResponse | null | undefined,
  options: { useApiSpecs: boolean },
): VideoGenerationModelSpecItem[] {
  const { useApiSpecs } = options
  if (!specs) return []
  return useApiSpecs ? specs.api_models : specs.local_models
}

/** Downloaded offerings keyed by offering id, not pipeline. */
export function getOfferingVideoGenerationModelSpecs(
  specs: VideoGenerationModelSpecsResponse | null | undefined,
): OfferingVideoGenerationModelSpecItem[] {
  return (specs?.downloaded_local_models ?? []).map((item) => ({
    model: item.model,
    spec: item.spec,
  }))
}

/**
 * Adapter for `resolveVideoGenerationOptions` / clamp, which still key models as
 * `pipeline`. Localized here so Home option fields can keep using `item.model`.
 */
export function offeringSpecsAsPipelineItems(
  items: OfferingVideoGenerationModelSpecItem[],
): VideoGenerationModelSpecItem[] {
  return items.map((item) => ({
    pipeline: item.model as VideoGenerationPipeline,
    spec: item.spec,
  }))
}

/**
 * Prefer the offering that matches Settings' active checkpoint; else newest
 * downloaded. `active_offering` is the stable join — both local offerings share
 * pipeline `fast`, so display_name is not a discriminator.
 */
export function defaultOfferingId(
  specs: VideoGenerationModelSpecsResponse | null | undefined,
): OfferingId | null {
  const downloaded = specs?.downloaded_local_models ?? []
  const active = specs?.active_offering
  if (active != null) {
    const matching = downloaded.find((item) => item.model === active)
    if (matching) return matching.model
  }
  return downloaded[0]?.model ?? null
}

/**
 * Restore a stored offering id when the specs response still downloads it,
 * else fall back to Settings' active downloaded offering. The specs response
 * is the single source of offered ids.
 */
export function restoreOfferingId(
  model: unknown,
  specs: VideoGenerationModelSpecsResponse | null | undefined,
): OfferingId | null {
  const downloaded = specs?.downloaded_local_models ?? []
  const match =
    typeof model === "string"
      ? downloaded.find((item) => item.model === model)
      : undefined
  if (match) return match.model
  return defaultOfferingId(specs)
}

export function withDefaultOffering<T extends { model: OfferingId }>(
  specs: VideoGenerationModelSpecsResponse | null | undefined,
  defaults: T,
): T {
  return {
    ...defaults,
    model: defaultOfferingId(specs) ?? defaults.model,
  }
}

export function getLocalOfferingCapabilities(
  specs: VideoGenerationModelSpecsResponse | null | undefined,
): VideoGenerationOfferingCapabilities | null {
  return specs?.local_models[0]?.spec.capabilities ?? null
}

export function getApiOfferingCapabilities(
  specs: VideoGenerationModelSpecsResponse | null | undefined,
  pipeline: string | null | undefined,
): VideoGenerationOfferingCapabilities | null {
  if (!specs || !pipeline) return null
  return specs.api_models.find((item) => item.pipeline === pipeline)?.spec.capabilities ?? null
}

type RiskRows<K extends string | number> = Map<string, Map<K, SelectionRisk>>

/** The server's warned cells for one matrix, indexed for the pickers. */
type FastGridWarnings = {
  /** Grouped by `resolution|fps`. */
  durationWarnings: RiskRows<VideoGenerationDuration>
  /** Grouped by `resolution|fps|duration`. */
  aspectWarnings: RiskRows<VideoGenerationAspectRatio>
  /** Kept cells that run in more than one window, keyed `resolution|fps|duration`. */
  chunkedCells: Map<string, ChunkedRun>
}

function rememberRisk<K extends string | number>(
  rows: RiskRows<K>,
  group: string,
  id: K,
  next: SelectionRisk,
): void {
  const current = rows.get(group)
  const groupRows = current ?? new Map<K, SelectionRisk>()
  if (!current) rows.set(group, groupRows)
  const worse = worseRisk(groupRows.get(id) ?? null, next)
  if (worse) groupRows.set(id, worse)
}

function warnedOption<T extends string | number>(
  value: T,
  risk: SelectionRisk,
): WarnedVideoGenerationOption<T> {
  return {
    value,
    message: risk.message,
    tier: risk.tier,
    ...(risk.readout ? { readout: risk.readout } : {}),
  }
}

function warnedFromRows<T extends string | number>(
  rows: ReadonlyMap<T, SelectionRisk> | undefined,
): WarnedVideoGenerationOption<T>[] {
  if (!rows) return []
  return [...rows.entries()].map(([value, risk]) => warnedOption(value, risk))
}

function fastGridWarnings(
  spec: VideoGenerationModelSpecItem['spec'],
  hasAudio: boolean,
): FastGridWarnings {
  const durationWarnings: RiskRows<VideoGenerationDuration> = new Map()
  const aspectWarnings: RiskRows<VideoGenerationAspectRatio> = new Map()
  const chunkedCells = new Map<string, ChunkedRun>()
  for (const cell of spec.chunked_cells ?? []) {
    if (cell.audio !== hasAudio) continue
    chunkedCells.set(`${cell.resolution}|${cell.fps}|${cell.duration}`, {
      windows: cell.windows,
      windowFrames: cell.window_frames,
    })
  }
  for (const cell of spec.warned_cells ?? []) {
    if (cell.audio !== hasAudio) continue
    rememberRisk(durationWarnings, `${cell.resolution}|${cell.fps}`, cell.duration, {
      message: cell.message,
      tier: cell.tier,
      readout: cell.readout,
    })
  }
  for (const cell of spec.warned_aspect_cells ?? []) {
    if (cell.audio !== hasAudio) continue
    rememberRisk(
      aspectWarnings,
      `${cell.resolution}|${cell.fps}|${cell.duration}`,
      cell.aspectRatio,
      { message: cell.message, tier: cell.tier, readout: cell.readout },
    )
  }
  return { durationWarnings, aspectWarnings, chunkedCells }
}

export function resolveVideoGenerationOptions<T extends VideoGenerationSettingsShape>({
  settings,
  modelSpecs,
  hasAudio = false,
  minimumDuration,
  durationSelection = 'preserve',
}: ResolveVideoGenerationOptionsParams<T>): ResolvedVideoGenerationOptions {
  const aspectSpecs = modelSpecs.map((item) => itemForAspect(item, settings.aspectRatio))
  const modelOptions = getCompatibleModelOptions(aspectSpecs, { hasAudio, minimumDuration })
  const selectedModelItem = modelOptions.find((item) => item.pipeline === settings.model) ?? modelOptions[0] ?? null
  // Dropped cells render from the selected row; with no compatible row the
  // first row still explains the empty grid (resolutions only — fps and
  // durations need a selection for context).
  const displayItem = selectedModelItem ?? aspectSpecs[0] ?? null
  if (!selectedModelItem) {
    return emptyResolvedOptions(modelOptions, {
      disabledResolutionOptions: disabledResolutionsForItem(displayItem, hasAudio, []),
    })
  }

  const resolutionEntries = getCompatibleResolutionEntries(selectedModelItem, { hasAudio, minimumDuration })
  const resolutionOptions = resolutionEntries.map(([resolution]) => resolution)
  const disabledResolutionOptions = disabledResolutionsForItem(selectedModelItem, hasAudio, resolutionOptions)
  const selectedResolution = chooseOption(settings.videoResolution, resolutionOptions)
  if (!selectedResolution) {
    return emptyResolvedOptions(modelOptions, {
      selectedModel: selectedModelItem.pipeline,
      resolutionOptions,
      disabledResolutionOptions,
    })
  }

  const selectedResolutionSpec = resolutionEntries.find(([resolution]) => resolution === selectedResolution)?.[1] ?? null
  if (!selectedResolutionSpec) {
    return emptyResolvedOptions(modelOptions, {
      selectedModel: selectedModelItem.pipeline,
      resolutionOptions,
      disabledResolutionOptions,
      selectedResolution,
    })
  }

  const aspectRatioOptions = (selectedResolutionSpec.aspect_ratios ?? []) as VideoGenerationAspectRatio[]
  const selectedAspectRatio = chooseOption(settings.aspectRatio ?? null, aspectRatioOptions)
  const fpsOptions = getCompatibleFps(selectedResolutionSpec, { minimumDuration })
  const disabledFpsOptions = disabledFpsForItem(selectedModelItem, hasAudio, selectedResolution, fpsOptions)
  const selectedFps = chooseOption(settings.fps, fpsOptions)
  if (!selectedFps) {
    return emptyResolvedOptions(modelOptions, {
      selectedModel: selectedModelItem.pipeline,
      resolutionOptions,
      disabledResolutionOptions,
      selectedResolution,
      fpsOptions,
      disabledFpsOptions,
      aspectRatioOptions,
      selectedAspectRatio,
    })
  }

  const durationOptions = filterDurationsByMinimum(
    getDurationsForFps(selectedResolutionSpec, selectedFps),
    minimumDuration,
  )
  const disabledDurationOptions = disabledDurationsForItem(
    selectedModelItem,
    hasAudio,
    selectedResolution,
    selectedFps,
    minimumDuration,
  )
  const autoDurationAvailable = !hasAudio && Boolean(selectedModelItem.spec.capabilities?.auto_duration)
  const selectedDuration = durationSelection === 'smallest_valid'
    ? durationOptions[0] ?? null
    : autoDurationAvailable && settings.duration === null
      ? null
      : chooseOption(settings.duration, durationOptions)
  const disabledAspectRatioOptions = disabledAspectsForItem(
    selectedModelItem,
    hasAudio,
    selectedResolution,
    selectedFps,
    selectedDuration,
    aspectRatioOptions,
  )
  const grid = fastGridWarnings(selectedModelItem.spec, hasAudio)
  const warnedDurationOptions = warnedFromRows(
    grid.durationWarnings.get(`${selectedResolution}|${selectedFps}`),
  )
  const warnedFpsOptions = fpsOptions.flatMap((fps) => {
    const risk =
      selectedDuration == null
        ? undefined
        : grid.durationWarnings.get(`${selectedResolution}|${fps}`)?.get(selectedDuration)
    return risk == null ? [] : [warnedOption(fps, risk)]
  })
  const warnedResolutionOptions = resolutionOptions.flatMap((resolution) => {
    const risk =
      selectedDuration == null
        ? undefined
        : grid.durationWarnings.get(`${resolution}|${selectedFps}`)?.get(selectedDuration)
    return risk == null ? [] : [warnedOption(resolution, risk)]
  })
  const warnedAspectRatioOptions = warnedAspectsForItem(
    selectedModelItem,
    grid,
    hasAudio,
    selectedResolution,
    selectedFps,
    selectedDuration,
    selectedAspectRatio,
    aspectRatioOptions,
  )

  const chunkedAt = (resolution: string, fps: number, duration: number): ChunkedRun | undefined =>
    grid.chunkedCells.get(`${resolution}|${fps}|${duration}`)
  const chunkedDurationOptions = durationOptions.flatMap((duration) => {
    const run = chunkedAt(selectedResolution, selectedFps, duration)
    return run == null ? [] : [{ value: duration, ...run }]
  })
  const chunkedFpsOptions = fpsOptions.flatMap((fps) => {
    const run = selectedDuration == null ? undefined : chunkedAt(selectedResolution, fps, selectedDuration)
    return run == null ? [] : [{ value: fps, ...run }]
  })
  const chunkedResolutionOptions = resolutionOptions.flatMap((resolution) => {
    const run = selectedDuration == null ? undefined : chunkedAt(resolution, selectedFps, selectedDuration)
    return run == null ? [] : [{ value: resolution, ...run }]
  })
  const singlePassResolutionOptions = resolutionEntries.flatMap(([resolution, spec]) => {
    const offered =
      selectedDuration != null && getDurationsForFps(spec, selectedFps).includes(selectedDuration)
    return offered && chunkedAt(resolution, selectedFps, selectedDuration) == null ? [resolution] : []
  })

  return {
    modelOptions,
    resolutionOptions,
    fpsOptions,
    durationOptions,
    disabledResolutionOptions,
    disabledFpsOptions,
    disabledDurationOptions,
    disabledAspectRatioOptions,
    warnedResolutionOptions,
    warnedFpsOptions,
    warnedDurationOptions,
    warnedAspectRatioOptions,
    chunkedResolutionOptions,
    singlePassResolutionOptions,
    chunkedFpsOptions,
    chunkedDurationOptions,
    selectedModel: selectedModelItem.pipeline,
    selectedResolution,
    selectedFps,
    selectedDuration,
    aspectRatioOptions,
    selectedAspectRatio,
    autoDurationAvailable,
    hasCompatibleOptions: selectedDuration !== null || autoDurationAvailable,
  }
}

function offeringItem(
  specs: VideoGenerationModelSpecsResponse | null | undefined,
  model: string,
): VideoGenerationModelSpecItem | null {
  const downloaded = specs?.downloaded_local_models ?? []
  const match = downloaded.find((item) => item.model === model) ?? downloaded[0]
  if (!match) {
    const local = specs?.local_models[0]
    return local ?? null
  }
  return { pipeline: match.model as VideoGenerationPipeline, spec: match.spec }
}

/** Worst budget warning for the current fast cell, or null when it fits cleanly. */
export function fastSelectionRisk(
  specs: VideoGenerationModelSpecsResponse | null | undefined,
  selection: {
    model: string
    resolution: string
    fps: number
    duration: number | null
    aspectRatio: string | null
    audio?: boolean
  },
): SelectionRisk | null {
  const item = offeringItem(specs, selection.model)
  if (!item || selection.duration == null) return null
  if (!(VIDEO_GENERATION_RESOLUTION_ORDER as readonly string[]).includes(selection.resolution)) {
    return null
  }
  const resolution = selection.resolution as VideoGenerationResolution
  const audio = selection.audio ?? false
  const aspect = selection.aspectRatio
  if (
    aspect != null &&
    aspect !== 'auto' &&
    isVideoGenerationAspectRatio(aspect) &&
    item.spec.aspect_specs?.[aspect] != null
  ) {
    const aspectGrid = fastGridWarnings(specForAspect(item.spec, aspect), audio)
    return (
      aspectGrid.durationWarnings
        .get(`${resolution}|${selection.fps}`)
        ?.get(selection.duration as VideoGenerationDuration) ?? null
    )
  }
  const grid = fastGridWarnings(item.spec, audio)
  const durationRisk =
    grid.durationWarnings
      .get(`${resolution}|${selection.fps}`)
      ?.get(selection.duration as VideoGenerationDuration) ?? null
  if (aspect == null || aspect === 'auto' || !isVideoGenerationAspectRatio(aspect)) {
    return durationRisk
  }
  const aspectRisk =
    grid.aspectWarnings
      .get(`${resolution}|${selection.fps}|${selection.duration}`)
      ?.get(aspect) ?? null
  return worseRisk(durationRisk, aspectRisk)
}

export type FastCellSelection = {
  model: string
  resolution: string
  fps: number
  duration: number | null
  aspectRatio: string | null
  audio?: boolean
}

function selectionSpec(
  specs: VideoGenerationModelSpecsResponse | null | undefined,
  selection: FastCellSelection,
): VideoGenerationModelSpecItem['spec'] | null {
  const item = offeringItem(specs, selection.model)
  if (!item) return null
  return selection.aspectRatio === 'auto' ? item.spec : specForAspect(item.spec, selection.aspectRatio)
}

/** What the cell asks of the compared memory, kept or withheld. Null while the cell is unknown. */
export function fastCellAsk(
  specs: VideoGenerationModelSpecsResponse | null | undefined,
  selection: FastCellSelection,
): FastCellAsk | null {
  if (selection.duration == null) return null
  const spec = selectionSpec(specs, selection)
  const audio = selection.audio ?? false
  return (
    spec?.cell_asks?.find(
      (cell) =>
        cell.resolution === selection.resolution &&
        cell.fps === selection.fps &&
        cell.duration === selection.duration &&
        cell.audio === audio,
    ) ?? null
  )
}

/** Durations the pickers offer for the selection's resolution and fps (kept, bets included). */
export function offeredFastDurations(
  specs: VideoGenerationModelSpecsResponse | null | undefined,
  selection: FastCellSelection,
): readonly number[] {
  const spec = selectionSpec(specs, selection)
  const grid = selection.audio ? spec?.a2v_supported_resolutions_durations : spec?.supported_resolutions_durations
  return grid?.[selection.resolution]?.fps_to_durations[String(selection.fps)] ?? []
}

/** The cell is offered in the pickers under the echoed mode (kept, bets included). */
export function isFastCellOffered(
  specs: VideoGenerationModelSpecsResponse | null | undefined,
  selection: FastCellSelection,
): boolean {
  if (selection.duration == null) return false
  return offeredFastDurations(specs, selection).includes(selection.duration)
}

type ResolutionMap = VideoGenerationModelSpecItem['spec']['supported_resolutions_durations']
type FastSpec = VideoGenerationModelSpecItem['spec']

function betDropReason(tier: SelectionTier): DroppedFastCell['reason'] {
  return tier === 'unmeasured' ? 'unmeasured_cell' : 'stream_over'
}

function stripAspectSpecBets(aspectSpec: VideoGenerationAspectSpec): VideoGenerationAspectSpec {
  const bets = (aspectSpec.warned_cells ?? []).filter((cell) => isConfirmableBudgetTier(cell.tier))
  if (bets.length === 0) return aspectSpec
  const betKeys = new Set(
    bets.map((cell) => `${cell.audio}|${cell.resolution}|${cell.fps}|${cell.duration}`),
  )
  const chunkedByKey = new Map(
    (aspectSpec.chunked_cells ?? []).map((cell) => [
      `${cell.audio}|${cell.resolution}|${cell.fps}|${cell.duration}`,
      cell,
    ]),
  )
  const strip = (map: ResolutionMap, audio: boolean): ResolutionMap => {
    const out: ResolutionMap = {}
    for (const [resolution, resolutionSpec] of Object.entries(map)) {
      const fpsToDurations: Record<string, number[]> = {}
      for (const [fps, durations] of Object.entries(resolutionSpec.fps_to_durations)) {
        const left = durations.filter(
          (duration) => !betKeys.has(`${audio}|${resolution}|${fps}|${duration}`),
        )
        if (left.length > 0) fpsToDurations[fps] = left
      }
      if (Object.keys(fpsToDurations).length > 0) {
        out[resolution] = {
          ...resolutionSpec,
          fps_to_durations: fpsToDurations as VideoGenerationResolutionSpec['fps_to_durations'],
        }
      }
    }
    return out
  }
  return {
    ...aspectSpec,
    supported_resolutions_durations: strip(aspectSpec.supported_resolutions_durations, false),
    a2v_supported_resolutions_durations:
      aspectSpec.a2v_supported_resolutions_durations == null
        ? aspectSpec.a2v_supported_resolutions_durations
        : strip(aspectSpec.a2v_supported_resolutions_durations, true),
    warned_cells: (aspectSpec.warned_cells ?? []).filter(
      (cell) => !isConfirmableBudgetTier(cell.tier),
    ),
    chunked_cells: (aspectSpec.chunked_cells ?? []).filter(
      (cell) => !betKeys.has(`${cell.audio}|${cell.resolution}|${cell.fps}|${cell.duration}`),
    ),
    dropped_cells: [
      ...(aspectSpec.dropped_cells ?? []),
      ...bets.map(({ resolution, fps, duration, tier, message, readout, audio }) => {
        const chunked = chunkedByKey.get(`${audio}|${resolution}|${fps}|${duration}`)
        return {
          resolution,
          fps,
          duration,
          reason: betDropReason(tier),
          message,
          readout,
          audio,
          ...(chunked ? { windows: chunked.windows, window_frames: chunked.window_frames } : {}),
        }
      }),
    ],
  }
}

function stripBetCells(spec: FastSpec): FastSpec {
  const bets = (spec.warned_cells ?? []).filter((cell) => isConfirmableBudgetTier(cell.tier))
  const aspectBets = (spec.warned_aspect_cells ?? []).filter((cell) =>
    isConfirmableBudgetTier(cell.tier),
  )
  const strippedAspectSpecs = spec.aspect_specs
    ? (Object.fromEntries(
        Object.entries(spec.aspect_specs).map(([ratio, aspectSpec]) => [
          ratio,
          stripAspectSpecBets(aspectSpec),
        ]),
      ) as NonNullable<FastSpec['aspect_specs']>)
    : undefined
  if (bets.length === 0 && aspectBets.length === 0 && !strippedAspectSpecs) return spec
  const baseStripped = stripAspectSpecBets({
    supported_resolutions_durations: spec.supported_resolutions_durations,
    a2v_supported_resolutions_durations: spec.a2v_supported_resolutions_durations,
    dropped_cells: spec.dropped_cells ?? undefined,
    warned_cells: spec.warned_cells ?? undefined,
    chunked_cells: spec.chunked_cells ?? undefined,
  })
  return {
    ...spec,
    supported_resolutions_durations: baseStripped.supported_resolutions_durations,
    a2v_supported_resolutions_durations: baseStripped.a2v_supported_resolutions_durations,
    warned_cells: baseStripped.warned_cells,
    warned_aspect_cells: (spec.warned_aspect_cells ?? []).filter(
      (cell) => !isConfirmableBudgetTier(cell.tier),
    ),
    ...(spec.chunked_cells ? { chunked_cells: baseStripped.chunked_cells } : {}),
    ...(strippedAspectSpecs ? { aspect_specs: strippedAspectSpecs } : {}),
    dropped_cells: baseStripped.dropped_cells,
    dropped_aspect_cells: [
      ...(spec.dropped_aspect_cells ?? []),
      ...aspectBets.map(({ resolution, fps, duration, aspectRatio, tier, message, readout, audio }) => ({
        resolution,
        fps,
        duration,
        aspectRatio,
        reason: betDropReason(tier),
        message,
        readout,
        audio,
      })),
    ],
  }
}

/**
 * The specs with every Brave bet moved from the kept grid to the disabled
 * list, its warning as the refusal. For surfaces with no risk confirm
 * (GenSpace, the editor gap fill), which must not start a bet unasked.
 */
export function withoutBetCells(
  specs: VideoGenerationModelSpecsResponse,
): VideoGenerationModelSpecsResponse {
  return {
    ...specs,
    local_models: specs.local_models.map((item) => ({ ...item, spec: stripBetCells(item.spec) })),
    ...(specs.downloaded_local_models
      ? {
          downloaded_local_models: specs.downloaded_local_models.map((item) => ({
            ...item,
            spec: stripBetCells(item.spec),
          })),
        }
      : {}),
  }
}

export function sanitizeVideoGenerationSettings<T extends VideoGenerationSettingsShape>(
  settings: T,
  modelSpecs: VideoGenerationModelSpecItem[],
  options: {
    hasAudio?: boolean
    minimumDuration?: number
    durationSelection?: DurationSelectionMode
  } = {},
): T | null {
  const resolved = resolveVideoGenerationOptions({
    settings,
    modelSpecs,
    hasAudio: options.hasAudio,
    minimumDuration: options.minimumDuration,
    durationSelection: options.durationSelection,
  })
  if (
    !resolved.hasCompatibleOptions
    || !resolved.selectedModel
    || !resolved.selectedResolution
    || !resolved.selectedFps
    || (resolved.selectedDuration === null && !resolved.autoDurationAvailable)
  ) {
    return null
  }

  return {
    ...settings,
    model: resolved.selectedModel,
    videoResolution: resolved.selectedResolution,
    fps: resolved.selectedFps,
    duration: resolved.selectedDuration,
    aspectRatio: resolved.selectedAspectRatio ?? '16:9',
  }
}

export function areVideoGenerationSettingsEquivalent<T extends VideoGenerationSettingsShape>(
  left: T,
  right: T,
): boolean {
  return (
    left.model === right.model
    && left.duration === right.duration
    && left.videoResolution === right.videoResolution
    && left.fps === right.fps
    && (left.aspectRatio ?? '16:9') === (right.aspectRatio ?? '16:9')
    && (left.audio ?? false) === (right.audio ?? false)
  )
}

/**
 * Fallback labels for persisted `generationParams.model` / picker pipeline ids.
 *
 * Prefer `resolvePipelineDisplayName` (backend spec) at generation time. Local ids like
 * "fast"/"pro" are shared across LTX versions, so the fallback here stays version-agnostic —
 * only API ids (`fast-2.5`, …) encode their version in the id itself.
 */
const PIPELINE_DISPLAY_NAMES: Record<string, string> = {
  fast: 'LTX Fast',
  pro: 'LTX Pro',
  'fast-2.5': 'LTX-2.5 Fast',
  'pro-2.5': 'LTX-2.5 Pro',
  'ltx-2.3-fast': 'LTX 2.3 Fast',
  'ltx-2.5-fast': 'LTX 2.5 Fast',
}

/** Returns a display label for a known video pipeline, or null if unknown/absent. */
export function formatPipelineDisplayName(model: string | undefined | null): string | null {
  if (!model) return null
  return PIPELINE_DISPLAY_NAMES[model] ?? null
}

/**
 * Version-correct label for `pipeline` taken from the backend specs currently in effect.
 * Returns null when the pipeline isn't in `modelSpecs`, so callers can fall back.
 */
export function resolvePipelineDisplayName(
  modelSpecs: VideoGenerationModelSpecItem[],
  pipeline: string | undefined | null,
): string | null {
  if (!pipeline) return null
  return modelSpecs.find((item) => item.pipeline === pipeline)?.spec.display_name ?? null
}
