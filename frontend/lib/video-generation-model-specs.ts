import type { components } from '../generated/backend-openapi'

export type VideoGenerationModelSpecsResponse = components['schemas']['GenerateVideoModelsSpecsResponse']
export type VideoGenerationModelSpecItem = components['schemas']['LTXVideoGenerationModelSpecItem']
export type DownloadedLocalVideoGenerationModelSpecItem =
  components['schemas']['DownloadedLocalVideoGenerationModelSpecItem']
export type OfferingId = DownloadedLocalVideoGenerationModelSpecItem['model']
export type VideoGenerationResolutionSpec = components['schemas']['LTXVideoGenerationResolutionSpec']
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

function chooseOption<T>(current: string | number | null, options: T[]): T | null {
  return options.find((option) => option === current) ?? options[0] ?? null
}

/** Same ids as backend `OFFERING_IDS`. Home create `params.model` values. */
export const OFFERING_IDS = ["ltx-2.5-fast", "ltx-2.3-fast"] as const

export function isOfferingId(value: unknown): value is OfferingId {
  return typeof value === "string" && (OFFERING_IDS as readonly string[]).includes(value)
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

/** Restore a stored offering id, or fall back to Settings' active downloaded offering. */
export function restoreOfferingId(
  model: unknown,
  specs: VideoGenerationModelSpecsResponse | null | undefined,
): OfferingId | null {
  if (isOfferingId(model)) return model
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

export function resolveVideoGenerationOptions<T extends VideoGenerationSettingsShape>({
  settings,
  modelSpecs,
  hasAudio = false,
  minimumDuration,
  durationSelection = 'preserve',
}: ResolveVideoGenerationOptionsParams<T>): ResolvedVideoGenerationOptions {
  const modelOptions = getCompatibleModelOptions(modelSpecs, { hasAudio, minimumDuration })
  const selectedModelItem = modelOptions.find((item) => item.pipeline === settings.model) ?? modelOptions[0] ?? null
  if (!selectedModelItem) {
    return emptyResolvedOptions(modelOptions)
  }

  const resolutionEntries = getCompatibleResolutionEntries(selectedModelItem, { hasAudio, minimumDuration })
  const resolutionOptions = resolutionEntries.map(([resolution]) => resolution)
  const selectedResolution = chooseOption(settings.videoResolution, resolutionOptions)
  if (!selectedResolution) {
    return emptyResolvedOptions(modelOptions, { selectedModel: selectedModelItem.pipeline, resolutionOptions })
  }

  const selectedResolutionSpec = resolutionEntries.find(([resolution]) => resolution === selectedResolution)?.[1] ?? null
  if (!selectedResolutionSpec) {
    return emptyResolvedOptions(modelOptions, {
      selectedModel: selectedModelItem.pipeline,
      resolutionOptions,
      selectedResolution,
    })
  }

  const aspectRatioOptions = (selectedResolutionSpec.aspect_ratios ?? []) as VideoGenerationAspectRatio[]
  const selectedAspectRatio = chooseOption(settings.aspectRatio ?? null, aspectRatioOptions)
  const fpsOptions = getCompatibleFps(selectedResolutionSpec, { minimumDuration })
  const selectedFps = chooseOption(settings.fps, fpsOptions)
  if (!selectedFps) {
    return emptyResolvedOptions(modelOptions, {
      selectedModel: selectedModelItem.pipeline,
      resolutionOptions,
      selectedResolution,
      fpsOptions,
      aspectRatioOptions,
      selectedAspectRatio,
    })
  }

  const durationOptions = filterDurationsByMinimum(
    getDurationsForFps(selectedResolutionSpec, selectedFps),
    minimumDuration,
  )
  const autoDurationAvailable = !hasAudio && Boolean(selectedModelItem.spec.capabilities?.auto_duration)
  const selectedDuration = durationSelection === 'smallest_valid'
    ? durationOptions[0] ?? null
    : autoDurationAvailable && settings.duration === null
      ? null
      : chooseOption(settings.duration, durationOptions)

  return {
    modelOptions,
    resolutionOptions,
    fpsOptions,
    durationOptions,
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
