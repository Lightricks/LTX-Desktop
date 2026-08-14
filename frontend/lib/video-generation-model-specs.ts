import type { components } from '../generated/backend-openapi'

export type VideoGenerationModelSpecsResponse = components['schemas']['GenerateVideoModelsSpecsResponse']
export type VideoGenerationModelSpecItem = components['schemas']['LTXVideoGenerationModelSpecItem']
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

export interface VideoGenerationSettingsShape {
  model: string
  duration: number | null
  videoResolution: string
  fps: number
  aspectRatio?: string
  audio?: boolean
}

export interface ResolvedVideoGenerationOptions {
  modelOptions: VideoGenerationModelSpecItem[]
  resolutionOptions: VideoGenerationResolution[]
  fpsOptions: VideoGenerationFps[]
  durationOptions: VideoGenerationDuration[]
  selectedModel: VideoGenerationPipeline | null
  selectedResolution: VideoGenerationResolution | null
  selectedFps: VideoGenerationFps | null
  selectedDuration: VideoGenerationDuration | null
  autoDurationAvailable: boolean
  hasCompatibleOptions: boolean
}

type DurationSelectionMode = 'preserve' | 'smallest_valid'

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
    selectedModel: null,
    selectedResolution: null,
    selectedFps: null,
    selectedDuration: null,
    autoDurationAvailable: false,
    hasCompatibleOptions: false,
    ...extras,
  }
}

function chooseOption<T>(current: string | number | null, options: T[]): T | null {
  return options.find((option) => option === current) ?? options[0] ?? null
}

export function getVideoGenerationModelSpecs(
  specs: VideoGenerationModelSpecsResponse | null | undefined,
  options: { useApiSpecs: boolean },
): VideoGenerationModelSpecItem[] {
  const { useApiSpecs } = options
  if (!specs) return []
  return useApiSpecs ? specs.api_models : specs.local_models
}

export function getLocalOfferingCapabilities(
  specs: VideoGenerationModelSpecsResponse | null | undefined,
): VideoGenerationOfferingCapabilities | null {
  return specs?.local_models[0]?.spec.capabilities ?? null
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

  const fpsOptions = getCompatibleFps(selectedResolutionSpec, { minimumDuration })
  const selectedFps = chooseOption(settings.fps, fpsOptions)
  if (!selectedFps) {
    return emptyResolvedOptions(modelOptions, {
      selectedModel: selectedModelItem.pipeline,
      resolutionOptions,
      selectedResolution,
      fpsOptions,
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
    aspectRatio: (settings.aspectRatio === '9:16' ? '9:16' : '16:9') as VideoGenerationAspectRatio,
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
