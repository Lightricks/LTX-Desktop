export type FirstRunDownloadFailurePolicy = 'blocking' | 'nonblocking'

export type FirstRunDownloadStep<CpId extends string = string> = {
  type: 'download'
  cpIds: CpId[]
  failurePolicy: FirstRunDownloadFailurePolicy
}

export type FirstRunModelPackRole = 'base' | 'upscaler' | 'vae' | 'support'
export type FirstRunLoraKind = 'lora' | 'ic-lora'

export type FirstRunLoraItem = {
  id: string
  name: string
  sizeBytes: number
  downloaded: boolean
  kind: FirstRunLoraKind
}

type CatalogLoraLike = {
  id: string
  name: string
  download: { variants: readonly { size_bytes: number }[] }
}

const MODEL_PACK_ROLES = new Set<string>(['base', 'upscaler', 'vae', 'support'])

export function isFirstRunModelPackItem(item: { role: string }): boolean {
  return MODEL_PACK_ROLES.has(item.role)
}

export function isFirstRunTextEncoder(item: { role: string }): boolean {
  return item.role === 'text_encoder'
}

export function isFirstRunImageModel(item: { role: string }): boolean {
  return item.role === 'image'
}

// LoRA ids are only unique within their kind (a catalog LoRA and an IC-LoRA can
// share an id), so selection state and render keys use the qualified key.
export function firstRunLoraKey(item: Pick<FirstRunLoraItem, 'kind' | 'id'>): string {
  return `${item.kind}:${item.id}`
}

export function firstRunLoraFromCatalog(
  kind: FirstRunLoraKind,
  catalog: CatalogLoraLike,
  downloaded: boolean,
): FirstRunLoraItem {
  return {
    id: catalog.id,
    name: catalog.name,
    sizeBytes: catalog.download.variants[0]?.size_bytes ?? 0,
    downloaded,
    kind,
  }
}

export function defaultSelectedFirstRunLoraKeys(items: readonly FirstRunLoraItem[]): string[] {
  return items.filter((item) => !item.downloaded).map(firstRunLoraKey)
}

export function createDownloadProgressPollGuard() {
  let inFlight = false
  const handledTerminalSessions = new Set<string>()

  return {
    begin(sessionId: string): boolean {
      if (inFlight || handledTerminalSessions.has(sessionId)) {
        return false
      }
      inFlight = true
      return true
    },
    markTerminal(sessionId: string): void {
      handledTerminalSessions.add(sessionId)
    },
    end(): void {
      inFlight = false
    },
  }
}

type LtxRecommendationLike<CpId extends string = string> = {
  status: string
  cps_to_download?: readonly CpId[]
  optional_cp_ids?: readonly CpId[]
  recommended_quality_cp_ids?: readonly CpId[]
}

type ImgGenRecommendationLike<CpId extends string = string> = {
  cp_to_download: CpId | null
}

export function uniqueCpIds<CpId extends string>(cpIds: readonly CpId[]): CpId[] {
  return [...new Set(cpIds)]
}

/**
 * Checkpoint ids to show in first-run Setup: the full bundle (required +
 * optional + recommended quality + image), not the missing-only download plan.
 *
 * Pass an `include_installed` recommendation: the same model Install will
 * download, with files already on disk kept in the set. `describeCheckpoints`
 * then marks those `downloaded`.
 */
export function firstRunCatalogCpIds<CpId extends string>(input: {
  ltxRecommendation: LtxRecommendationLike<CpId>
  imgGenCpToDownload: CpId | null
}): CpId[] {
  return uniqueCpIds([
    ...(input.ltxRecommendation.cps_to_download ?? []),
    ...(input.ltxRecommendation.optional_cp_ids ?? []),
    ...(input.ltxRecommendation.recommended_quality_cp_ids ?? []),
    ...(input.imgGenCpToDownload ? [input.imgGenCpToDownload] : []),
  ])
}

export function recommendedQualityCpIds<CpId extends string>(
  recommendation: LtxRecommendationLike<CpId>,
): CpId[] {
  if (recommendation.status !== 'download' && recommendation.status !== 'upgrade') {
    return []
  }
  return uniqueCpIds(recommendation.recommended_quality_cp_ids ?? [])
}

export function defaultSelectedRecommendedQualityCpIds<CpId extends string>(
  recommendation: LtxRecommendationLike<CpId>,
): CpId[] {
  return recommendedQualityCpIds(recommendation)
}

/**
 * Whether the boot `missing-models` gate can clear.
 *
 * The LTX pack is required: `status === 'download'` means the current video
 * bundle is still absent, including any required companion (the text encoder
 * when it is in that set because there is no API key). `ok` / `upgrade` are
 * both enough to enter — an upgrade is offered in-app.
 *
 * `imgGen` is the live recommendation. Skipping Z Image Turbo in Setup does
 * not change it: the backend still returns `cp_to_download: 'z-image-turbo'`
 * when the file is missing. That must not block entry. Quality extras and
 * LoRAs are opt-in and never flip LTX status to `download`.
 */
export function isRequiredLtxBundleReady<CpId extends string>(
  ltx: LtxRecommendationLike<CpId>,
  imgGen: ImgGenRecommendationLike<CpId>,
): boolean {
  void imgGen
  return ltx.status !== 'download'
}

export function buildFirstRunDownloadSteps<CpId extends string>(input: {
  ltxRecommendation: LtxRecommendationLike<CpId>
  imgGenCpToDownload: CpId | null
  selectedRecommendedQualityCpIds: readonly CpId[]
  optedInOptionalCpIds: readonly CpId[]
  excludedCpIds?: readonly CpId[]
}): FirstRunDownloadStep<CpId>[] {
  const recommendedSet = new Set(recommendedQualityCpIds(input.ltxRecommendation))
  const recommended = uniqueCpIds(
    input.selectedRecommendedQualityCpIds.filter((cpId) => recommendedSet.has(cpId)),
  )
  const recommendedLookup = new Set(recommended)
  const excluded = new Set(input.excludedCpIds ?? [])

  const blocking: CpId[] = []
  if (input.ltxRecommendation.status === 'download') {
    blocking.push(...(input.ltxRecommendation.cps_to_download ?? []))
  }
  if (input.imgGenCpToDownload) {
    blocking.push(input.imgGenCpToDownload)
  }
  blocking.push(...input.optedInOptionalCpIds)

  const steps: FirstRunDownloadStep<CpId>[] = []
  const blockingUnique = uniqueCpIds(
    blocking.filter((cpId) => !recommendedLookup.has(cpId) && !excluded.has(cpId)),
  )
  if (blockingUnique.length > 0) {
    steps.push({ type: 'download', cpIds: blockingUnique, failurePolicy: 'blocking' })
  }
  if (recommended.length > 0) {
    steps.push({ type: 'download', cpIds: recommended, failurePolicy: 'nonblocking' })
  }
  return steps
}

export function resolveDownloadFailure<FailedCpId extends string, QueueItem>(
  failedStep: FirstRunDownloadStep<FailedCpId>,
  remainingQueue: readonly QueueItem[],
): { action: 'error' } | { action: 'continue'; next: QueueItem } | { action: 'complete' } {
  if (failedStep.failurePolicy === 'blocking') {
    return { action: 'error' }
  }
  const next = remainingQueue[0]
  if (next !== undefined) {
    return { action: 'continue', next }
  }
  return { action: 'complete' }
}

// User-facing (and logged) messages for location-step recommendation failures.
// One builder per fetch phase so the UI text can't drift from the log text.
export function recommendationsErrorMessage(detail: string): string {
  return `Failed to fetch model recommendations: ${detail}`
}

export function describeCheckpointsErrorMessage(detail: string): string {
  return `Failed to describe checkpoints: ${detail}`
}

export type LoraDownloadProgressLike = {
  status: string
  progress: number
  expected_bytes: number
  speed_bytes_per_sec: number
  downloaded_bytes: number
  error?: string | null
}

export type FirstRunDownloadProgressView = {
  status: 'downloading' | 'complete' | 'error'
  all_files: string[]
  completed_files: string[]
  current_downloading_file: string
  current_file_progress: number
  expected_total_bytes: number
  speed_bytes_per_sec: number
  total_downloaded_bytes: number
  total_progress: number
  error: string | null
}

// The installing step renders one model-progress shape for both model bundles
// and LoRAs, so LoRA progress is mapped onto that shape at the poll boundary.
export function mapLoraProgressToDownloadProgress(
  progress: LoraDownloadProgressLike,
  item: Pick<FirstRunLoraItem, 'name' | 'sizeBytes'>,
): FirstRunDownloadProgressView {
  const complete = progress.status === 'complete'
  return {
    status: progress.status === 'error' ? 'error' : complete ? 'complete' : 'downloading',
    all_files: [item.name],
    completed_files: complete ? [item.name] : [],
    current_downloading_file: item.name,
    current_file_progress: progress.progress,
    expected_total_bytes: progress.expected_bytes || item.sizeBytes,
    speed_bytes_per_sec: progress.speed_bytes_per_sec,
    total_downloaded_bytes: progress.downloaded_bytes,
    total_progress: progress.progress,
    error: progress.error ?? null,
  }
}
