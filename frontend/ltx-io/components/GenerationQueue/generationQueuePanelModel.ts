import type { ExploreAsset } from "../../../lib/explore-contract.ts";
import { MAX_IN_FLIGHT_PROGRESS_PERCENT } from "../../../lib/generation-display-progress.ts";
import type {
  ExploreQueueEntry,
  ReorderGenerationQueueRequest,
} from "../../hooks/generationQueryKeys.ts";
import { generationPresentation } from "../../lib/generationPresentation.ts";
import { ApiResultError } from "../../lib/unwrapApiResult.ts";

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
    active: T | null;
    queued: readonly T[];
  },
  options?: { isReordering?: boolean; isCancelPending?: boolean },
) {
  const cancelRequestPending = options?.isCancelPending === true;
  return {
    activeIsFixed: snapshot.active != null,
    activeCancelDisabled:
      snapshot.active?.generation.status === "cancelling" ||
      cancelRequestPending,
    queuedCancelDisabled: cancelRequestPending,
    sortableIds: snapshot.queued.map((entry) => entry.generation.id),
    sortingDisabled: options?.isReordering === true,
  };
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
        active: QueueEntryWithProgress | null;
        queued: readonly QueueEntryWithProgress[];
      }
    | null
    | undefined,
): Map<string, number> {
  const map = new Map<string, number>();
  if (snapshot == null) return map;
  const entries = [snapshot.active, ...snapshot.queued].filter(
    (entry): entry is QueueEntryWithProgress => entry != null,
  );
  for (const entry of entries) {
    if (entry.progress == null) continue;
    const percent = reportedDenoisePercent(entry.progress);
    if (percent == null) continue;
    map.set(entry.generation.id, percent);
  }
  return map;
}

/** Short label for the progress control (percentage from steps when available). */
export function formatActiveQueueProgress(progress: {
  progress: number;
  currentStep: number | null;
  totalSteps: number | null;
}): string {
  return `${activeQueueProgressPercent(progress)}%`;
}

export function resultLocationForFeature(
  feature: string,
  generationId: string,
  features: readonly { id: string; path: string }[],
): { pathname: string; search: string } | null {
  const match = features.find((item) => item.id === feature);
  if (match == null) return null;
  return {
    pathname: match.path,
    search: `?result=${encodeURIComponent(generationId)}`,
  };
}

export function formatActiveQueueProgressDetail(progress: {
  progress: number;
  currentStep: number | null;
  totalSteps: number | null;
}): string {
  const percent = activeQueueProgressPercent(progress);
  if (
    progress.currentStep != null &&
    progress.totalSteps != null &&
    progress.totalSteps > 1
  ) {
    return `Step ${progress.currentStep} of ${progress.totalSteps} (${percent}%)`;
  }
  return `${percent}% complete`;
}
