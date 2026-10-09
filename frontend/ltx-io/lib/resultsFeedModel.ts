import type { ExploreGeneration } from "@/lib/explore-contract";

export type Generation = ExploreGeneration;

export const MAX_VISIBLE_FINISHED_RESULTS = 10;

const IN_FLIGHT_STATUSES = new Set<Generation["status"]>([
  "queued",
  "running",
  "cancelling",
]);
const FINISHED_STATUSES = new Set<Generation["status"]>([
  "succeeded",
  "failed",
  "cancelled",
]);

export function isInFlightGeneration(generation: Generation): boolean {
  return IN_FLIGHT_STATUSES.has(generation.status);
}

/** True when a later snapshot contains a succeeded id the earlier one did not. */
export function hasNewSucceededGeneration(
  previous: readonly Generation[] | undefined,
  next: readonly Generation[] | undefined,
): boolean {
  if (previous == null || next == null) return false;
  const previouslySucceeded = new Set(
    previous
      .filter((generation) => generation.status === "succeeded")
      .map((generation) => generation.id),
  );
  return next.some(
    (generation) =>
      generation.status === "succeeded" &&
      !previouslySucceeded.has(generation.id),
  );
}

/** Newest-first: keep every in-flight row; cap finished frames. */
export function getVisibleGenerations(
  generations: Generation[],
  maxFinished: number = MAX_VISIBLE_FINISHED_RESULTS,
): Generation[] {
  const visible: Generation[] = [];
  let finishedCount = 0;
  for (const generation of generations) {
    if (isInFlightGeneration(generation)) {
      visible.push(generation);
      continue;
    }
    if (FINISHED_STATUSES.has(generation.status) && finishedCount < maxFinished) {
      visible.push(generation);
      finishedCount += 1;
    }
  }
  return visible;
}

/**
 * Read a generation's `spec.params` as a plain object, or null when absent.
 * Accepts `unknown` so it also guards specs hydrated from untyped sources
 * (e.g. a definition's `fromGeneration(spec: unknown)`).
 */
export function readGenerationParams(spec: unknown): Record<string, unknown> | null {
  if (!spec || typeof spec !== "object") return null;
  const params = (spec as { params?: unknown }).params;
  if (!params || typeof params !== "object" || Array.isArray(params)) {
    return null;
  }
  return params as Record<string, unknown>;
}
