import type { ExploreApi } from "../runtime/ExploreRuntime";

/**
 * Cancels a queued generation when no component is left to show the result, such as after a
 * modal closes. A failed cancel has no screen to report to, so it is ignored.
 */
export function cancelGenerationQuietly(
  api: Pick<ExploreApi, "cancelQueuedGeneration">,
  generationId: string,
): void {
  void Promise.resolve(api.cancelQueuedGeneration(generationId)).catch(() => undefined);
}
