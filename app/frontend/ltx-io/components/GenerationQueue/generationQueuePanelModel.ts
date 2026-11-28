import { CUSTOM_LORA_HOME_FEATURE } from "../../../lib/custom-lora.ts";
import type { ExploreAsset } from "../../../lib/explore-contract.ts";
import {
  formatRemainingTime,
  formatShortRemainingTime,
} from "../../../lib/duration-format.ts";
import { MAX_IN_FLIGHT_PROGRESS_PERCENT } from "../../../lib/generation-display-progress.ts";
import { customLoraRunPath } from "../../../paths.ts";
import type {
  ExploreQueueEntry,
  ReorderGenerationQueueRequest,
} from "../../hooks/generationQueryKeys.ts";
import { generationPresentation } from "../../lib/generationPresentation.ts";
import { ApiResultError } from "../../lib/unwrapApiResult.ts";
import { customLoraIdOfGeneration } from "../../screens/Feature/definitions/customLora.ts";

/** Input assets in spec order, then any remaining `input_assets` from the API. */
export function orderedQueueInputAssets(
  entry: Pick<ExploreQueueEntry, "generation" | "input_assets">,
): ExploreAsset[] {
  const orderedIds = generationPresentation(entry.generation).inputAssetIds;
  const byId = new Map(entry.input_assets.map((asset) => [asset.id, asset]));
  const seen = new Set<string>();
  const ordered: ExploreAsset[] = [];
  for (const id of orderedIds) {
    const asset = byId.get(id);
    if (asset == null || seen.has(id)) continue;
    seen.add(id);
    ordered.push(asset);
  }
  for (const asset of entry.input_assets) {
    if (seen.has(asset.id)) continue;
    seen.add(asset.id);
    ordered.push(asset);
  }
  return ordered;
}

export type QueueEntryLike = {
  generation: {
    id: string;
    status: string;
  };
};

export function getQueuePanelState<T extends QueueEntryLike>(
  snapshot: {
    active: readonly T[];
    queued: readonly T[];
  },
  options?: { isReordering?: boolean; isCancelPending?: boolean },
) {
  const cancelRequestPending = options?.isCancelPending === true;
  return {
    activeIsFixed: snapshot.active.length > 0,
    queuedCancelDisabled: cancelRequestPending,
    sortableIds: snapshot.queued.map((entry) => entry.generation.id),
    sortingDisabled: options?.isReordering === true,
  };
}

/** A running row cannot be cancelled twice, and not while another cancel request is pending. */
export function isActiveCancelDisabled(
  entry: QueueEntryLike,
  cancelRequestPending: boolean,
): boolean {
  return entry.generation.status === "cancelling" || cancelRequestPending;
}

/** The lane label for a row that runs outside the local GPU. Null for a local row. */
export function queueLaneLabel(lane: string | null | undefined): string | null {
  if (lane == null || lane === "local") return null;
  return lane;
}

/** The row badges plus the lane label. The label is not repeated when a badge already shows it. */
export function queueRowBadges(
  badges: readonly string[],
  lane: string | null | undefined,
): string[] {
  const laneLabel = queueLaneLabel(lane);
  if (laneLabel == null || badges.includes(laneLabel)) return [...badges];
  return [...badges, laneLabel];
}

/**
 * What a provider job reports while it runs. Null for a local job, which shows
 * its own progress.
 */
export function activeQueueTitleStatus(
  apiStatus: "pending" | "processing" | null | undefined,
): string | null {
  if (apiStatus === "pending") return "Waiting";
  if (apiStatus === "processing") return "Generating";
  return null;
}

export function queuedEntryForActiveDrag<T extends QueueEntryLike>(
  queued: readonly T[],
  activeId: string | null,
): { entry: T; index: number } | null {
  if (activeId == null) return null;
  const index = queued.findIndex((entry) => entry.generation.id === activeId);
  if (index < 0) return null;
  return { entry: queued[index], index };
}

export function reorderQueuedEntries<T extends QueueEntryLike>(
  queued: readonly T[],
  activeId: string,
  overId: string,
): T[] {
  const oldIndex = queued.findIndex((entry) => entry.generation.id === activeId);
  const newIndex = queued.findIndex((entry) => entry.generation.id === overId);
  if (oldIndex < 0 || newIndex < 0) return [...queued];

  const next = [...queued];
  const [moved] = next.splice(oldIndex, 1);
  next.splice(newIndex, 0, moved);
  return next;
}

export function requestForQueueOrder<T extends QueueEntryLike>(
  queued: readonly T[],
  generationId: string,
): ReorderGenerationQueueRequest | null {
  const index = queued.findIndex((entry) => entry.generation.id === generationId);
  if (index < 0) return null;
  return {
    generation_id: generationId,
    before_generation_id: queued[index + 1]?.generation.id ?? null,
  };
}

export function isQueueReorderConflict(error: unknown): boolean {
  return (
    error instanceof ApiResultError &&
    error.code === "INVALID_GENERATION_STATUS"
  );
}

export type QueueProgressSnapshot = {
  progress: number;
  currentStep: number | null;
  totalSteps: number | null;
  remainingSeconds?: number | null;
};

export function activeQueueProgressPercent(progress: QueueProgressSnapshot): number {
  const raw =
    progress.currentStep != null &&
    progress.totalSteps != null &&
    progress.totalSteps > 1
      ? (progress.currentStep / progress.totalSteps) * 100
      : progress.progress;
  return Math.min(
    MAX_IN_FLIGHT_PROGRESS_PERCENT,
    Math.max(0, Math.round(raw)),
  );
}

type QueueEntryWithProgress = {
  generation: { id: string };
  progress?: QueueProgressSnapshot | null;
};

/**
 * Percent for a generation that has completed at least one denoise step.
 * Phase placeholders (loading 5%, text encoding 10%) and the step-0 reset
 * are omitted so the result tile stays on dots until the bar can rise.
 */
function reportedDenoisePercent(progress: QueueProgressSnapshot): number | undefined {
  if (
    progress.currentStep == null ||
    progress.totalSteps == null ||
    progress.totalSteps <= 1 ||
    progress.currentStep <= 0
  ) {
    return undefined;
  }
  return activeQueueProgressPercent(progress);
}

/** Percent for each queue row that has real denoise progress. */
export function progressPercentByGenerationId(
  snapshot:
    | {
        active: readonly QueueEntryWithProgress[];
        queued: readonly QueueEntryWithProgress[];
      }
    | null
    | undefined,
): Map<string, number> {
  const map = new Map<string, number>();
  if (snapshot == null) return map;
  for (const entry of [...snapshot.active, ...snapshot.queued]) {
    if (entry.progress == null) continue;
    const percent = reportedDenoisePercent(entry.progress);
    if (percent == null) continue;
    map.set(entry.generation.id, percent);
  }
  return map;
}

/** Live remaining seconds for each running queue row that reported an ETA. */
export function remainingSecondsByGenerationId(
  snapshot:
    | {
        active: readonly QueueEntryWithProgress[];
        queued: readonly QueueEntryWithProgress[];
      }
    | null
    | undefined,
): Map<string, number> {
  const map = new Map<string, number>();
  if (snapshot == null) return map;
  for (const entry of [...snapshot.active, ...snapshot.queued]) {
    const rem = entry.progress?.remainingSeconds;
    if (rem == null || rem < 0) continue;
    map.set(entry.generation.id, rem);
  }
  return map;
}

/**
 * Label for the progress control. It follows the result placeholder: null until a denoise
 * step has run, so the queue row and the placeholder never show different percents.
 */
export function formatActiveQueueProgress(progress: QueueProgressSnapshot): string | null {
  const percent = reportedDenoisePercent(progress);
  const remaining = formatShortRemainingTime(progress.remainingSeconds);
  if (percent == null) return remaining;
  return remaining == null ? `${percent}%` : `${percent}% · ${remaining}`;
}

export function resultLocationForFeature(
  feature: string,
  generationId: string,
  features: readonly { id: string; path: string }[],
  spec?: unknown,
): { pathname: string; search: string } | null {
  const match = features.find((item) => item.id === feature);
  if (match == null) return null;
  let pathname = match.path;
  if (feature === CUSTOM_LORA_HOME_FEATURE.id) {
    const customLoraId = customLoraIdOfGeneration(spec);
    if (customLoraId == null) return null;
    pathname = customLoraRunPath(customLoraId);
  }
  return {
    pathname,
    search: `?result=${encodeURIComponent(generationId)}`,
  };
}

export function formatActiveQueueProgressDetail(progress: QueueProgressSnapshot): string {
  const percent = reportedDenoisePercent(progress);
  const remaining = formatRemainingTime(progress.remainingSeconds);
  if (percent == null) {
    return remaining ?? "Waiting for progress…";
  }
  const stepText = `Step ${progress.currentStep} of ${progress.totalSteps} (${percent}%)`;
  return remaining == null ? stepText : `${stepText} — ${remaining}`;
}

/**
 * The text a screen reader gets for a running row. A provider job reports a phase with no step
 * count, so its provider status is the detail. Without either, the job has not reported yet.
 */
export function activeQueueRowDetail(
  progress: QueueProgressSnapshot | null | undefined,
  apiStatus: Parameters<typeof activeQueueTitleStatus>[0],
): string {
  if (progress != null && formatActiveQueueProgress(progress) != null) {
    return formatActiveQueueProgressDetail(progress);
  }
  const statusLabel = activeQueueTitleStatus(apiStatus);
  return statusLabel != null ? `${statusLabel}…` : "Waiting for progress…";
}
