import { isInFlightGeneration, type Generation } from "../lib/resultsFeedModel.ts";

export const DESKTOP_QUEUE_POLL_INTERVAL_MS = 1000;
export const REMOTE_QUEUE_POLL_INTERVAL_MS = 2000;
/** In-flight generation lists poll slowly; creating or cancelling refreshes them immediately. */
export const GENERATION_LIST_POLL_INTERVAL_MS = 10_000;
export const IDLE_QUEUE_POLL_INTERVAL_MS = 5000;
export const CLOSED_QUEUE_POLL_INTERVAL_MS = 10_000;
export const GENERATION_ERROR_BACKOFF_START_MS = 2000;
export const GENERATION_ERROR_BACKOFF_MAX_MS = 30_000;

export type ExploreGenerationPollingPolicy = {
  /** Live queue poll while it has work. Generation lists use GENERATION_LIST_POLL_INTERVAL_MS. */
  queueActiveIntervalMs: number;
  errorBackoffMs: (failureCount: number) => number;
};

export type GenerationPollQueryState = {
  data: readonly Generation[] | undefined;
  error: unknown;
  fetchFailureCount: number;
};

export const GENERATION_POLLING_QUERY_OPTIONS = {
  refetchOnWindowFocus: true,
  refetchOnReconnect: true,
  refetchIntervalInBackground: false,
} as const;

export function generationErrorBackoffMs(failureCount: number): number {
  const attempt = Math.max(1, failureCount);
  return Math.min(
    GENERATION_ERROR_BACKOFF_MAX_MS,
    GENERATION_ERROR_BACKOFF_START_MS * 2 ** (attempt - 1),
  );
}

export const DESKTOP_GENERATION_POLLING_POLICY: ExploreGenerationPollingPolicy = {
  queueActiveIntervalMs: DESKTOP_QUEUE_POLL_INTERVAL_MS,
  errorBackoffMs: generationErrorBackoffMs,
};

export const REMOTE_GENERATION_POLLING_POLICY: ExploreGenerationPollingPolicy = {
  queueActiveIntervalMs: REMOTE_QUEUE_POLL_INTERVAL_MS,
  errorBackoffMs: generationErrorBackoffMs,
};

export function generationRefetchInterval(
  policy: ExploreGenerationPollingPolicy,
  state: GenerationPollQueryState,
  keepPolling: boolean,
): number | false {
  if (state.error) {
    return policy.errorBackoffMs(state.fetchFailureCount);
  }
  if (keepPolling) {
    return GENERATION_LIST_POLL_INTERVAL_MS;
  }
  const rows = state.data;
  if (rows == null) {
    return GENERATION_LIST_POLL_INTERVAL_MS;
  }
  return rows.some(isInFlightGeneration) ? GENERATION_LIST_POLL_INTERVAL_MS : false;
}

export type QueuePollQueryState = {
  data?: { active: unknown; queued: readonly unknown[] } | undefined;
  error: unknown;
  fetchFailureCount: number;
};

/**
 * `isLive` is true for the observer behind a visible queue panel: it polls fast,
 * and slows to the idle rate once the queue is empty (local enqueues invalidate
 * the query right away). Background observers (badge, dashboard, results feed)
 * pass false and poll slowly. Each observer has its own timer on the shared
 * query, so only the panel's observer runs fast. Errors back off either way,
 * and never faster than the mode's normal rate.
 */
export function queueRefetchInterval(
  policy: ExploreGenerationPollingPolicy,
  state: QueuePollQueryState,
  isLive: boolean,
): number {
  if (!isLive) {
    return state.error
      ? Math.max(
          CLOSED_QUEUE_POLL_INTERVAL_MS,
          policy.errorBackoffMs(state.fetchFailureCount),
        )
      : CLOSED_QUEUE_POLL_INTERVAL_MS;
  }
  if (state.error) {
    return policy.errorBackoffMs(state.fetchFailureCount);
  }
  const { data } = state;
  if (data && data.active == null && data.queued.length === 0) {
    return IDLE_QUEUE_POLL_INTERVAL_MS;
  }
  return policy.queueActiveIntervalMs;
}
