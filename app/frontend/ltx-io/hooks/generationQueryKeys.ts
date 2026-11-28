import type { QueryClient } from "@tanstack/react-query";

import type { components } from "../../generated/backend-openapi.ts";
import type { BoundApiClient } from "../../lib/api-client.ts";
import type { BudgetMode } from "../../lib/budget-risk.ts";
import type { ExploreAsset } from "../../lib/explore-contract.ts";
import type { Generation } from "../lib/resultsFeedModel.ts";
import { unwrapApiResult } from "../lib/unwrapApiResult.ts";
import type {
  ExploreApi,
  ExploreModelsVersion,
} from "../runtime/ExploreRuntime.tsx";

export {
  assetQueryKeys,
  isImageAsset,
} from "./assetQueryKeys.ts";

export const VIDEO_GENERATION_MODEL_SPECS_QUERY_OPTIONS = {
  refetchOnMount: true,
  refetchOnWindowFocus: true,
} as const;

export const generationQueryKeys = {
  all: ["generations"] as const,
  detail: (id: string) => [...generationQueryKeys.all, "detail", id] as const,
  list: (feature: string) =>
    [...generationQueryKeys.all, "list", feature] as const,
  recentFeatures: (limit: number) =>
    [...generationQueryKeys.all, "recent-features", limit] as const,
  dashboards: ["generations", "dashboard"] as const,
  dashboard: (range: components["schemas"]["DashboardSelection"]["range"], tz: string) =>
    [...generationQueryKeys.dashboards, range, tz] as const,
  queue: ["generation-queue"] as const,
  allSpecs: ["video-generation-model-specs"] as const,
  specs: <T extends ExploreModelsVersion>(
    modelsVersion: T,
    budgetMode: BudgetMode | "server" = "server",
    neverSplitIntoParts: boolean | "server" = "server",
  ) =>
    [...generationQueryKeys.allSpecs, modelsVersion, budgetMode, neverSplitIntoParts] as const,
};

export type GenerationQueueApi = Pick<
  BoundApiClient,
  "getGenerationQueue" | "reorderGenerationQueue"
>;

export type ReorderGenerationQueueRequest =
  components["schemas"]["ReorderGenerationQueueRequest"];

type DesktopQueueSnapshot = components["schemas"]["QueueSnapshot"];

export type ExploreQueueEntry = Omit<
  DesktopQueueSnapshot["queued"][number],
  "generation" | "input_assets"
> & {
  generation: Generation;
  input_assets: ExploreAsset[];
};

export type ExploreQueueSnapshot = {
  /** One entry per running job. Jobs on different lanes run at the same time. */
  active: ExploreQueueEntry[];
  queued: ExploreQueueEntry[];
  done: ExploreQueueEntry[];
  failed: ExploreQueueEntry[];
  unseen_ids: string[];
};

/** An entry of an older snapshot has no `cancellable`. */
type LegacyActiveEntry = Omit<ExploreQueueEntry, "cancellable"> & {
  cancellable?: boolean;
};

/**
 * Older snapshots (and cached ones) hold a single entry or null. Both read as a list.
 * A missing `cancellable` reads as true, like the backend default: only an explicit
 * false hides Cancel.
 */
function activeEntries(
  active: LegacyActiveEntry | readonly LegacyActiveEntry[] | null | undefined,
): ExploreQueueEntry[] {
  if (active == null) return [];
  const entries: readonly LegacyActiveEntry[] = Array.isArray(active)
    ? active
    : [active as LegacyActiveEntry];
  return entries.map((entry) => ({
    ...entry,
    cancellable: entry.cancellable ?? true,
  }));
}

export function normalizeQueueSnapshot(snapshot: {
  active?: LegacyActiveEntry | readonly LegacyActiveEntry[] | null;
  queued: ExploreQueueSnapshot["queued"];
  done?: ExploreQueueSnapshot["done"] | null;
  failed?: ExploreQueueSnapshot["failed"] | null;
  unseen_ids?: string[] | null;
}): ExploreQueueSnapshot {
  return {
    active: activeEntries(snapshot.active),
    queued: snapshot.queued,
    done: snapshot.done ?? [],
    failed: snapshot.failed ?? [],
    unseen_ids: snapshot.unseen_ids ?? [],
  };
}

export async function fetchAsset(
  api: ExploreApi,
  assetId: string,
): Promise<ExploreAsset> {
  return unwrapApiResult(await api.getAsset(assetId));
}

export async function fetchGenerations(
  api: ExploreApi,
  feature: string,
): Promise<Generation[]> {
  return unwrapApiResult(await api.listGenerations({ feature }));
}

export async function fetchVideoGenerationModelSpecs(api: ExploreApi) {
  return unwrapApiResult(await api.getGenerateVideoModelSpecs());
}

export async function fetchGenerationQueue(
  api: GenerationQueueApi,
): Promise<ExploreQueueSnapshot> {
  return normalizeQueueSnapshot(unwrapApiResult(await api.getGenerationQueue()));
}

export async function fetchQueueSnapshotAfterReorder(
  api: GenerationQueueApi,
  request: ReorderGenerationQueueRequest,
): Promise<ExploreQueueSnapshot> {
  return normalizeQueueSnapshot(
    unwrapApiResult(await api.reorderGenerationQueue(request)),
  );
}

export function replaceGenerationQueueCache(
  queryClient: QueryClient,
  snapshot: ExploreQueueSnapshot,
): void {
  queryClient.setQueryData(generationQueryKeys.queue, snapshot);
}

export function invalidateAffectedGenerationQueries(
  queryClient: QueryClient,
): void {
  void queryClient.invalidateQueries({ queryKey: generationQueryKeys.all });
  void queryClient.invalidateQueries({ queryKey: generationQueryKeys.queue });
}
