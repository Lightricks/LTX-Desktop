import type { QueryClient } from "@tanstack/react-query";

import { unwrapApiResult } from "../ltx-io/lib/unwrapApiResult.ts";
import type { BoundApiClient } from "./api-client.ts";

/**
 * Client side of the shared generation seed (GET/POST /api/generation-seed).
 *
 * The desktop and the phone share one seed on the server, so every surface reads it through
 * one react-query entry and writes through one debounced writer per QueryClient. Kept free of
 * React so the Settings screen, the Generate forms, and the tests all use the same logic.
 */

export type GenerationSeedState = { seed: number; locked: boolean };

/** The two endpoints this needs, so the Settings screen can pass the plain `ApiClient`. */
export type GenerationSeedApi = Pick<
  BoundApiClient,
  "getGenerationSeed" | "updateGenerationSeed"
>;

export const GENERATION_SEED_QUERY_KEY = ["settings", "generation-seed"] as const;
/** Once loaded, each client re-reads the seed now and then to pick up the other's edit. */
export const GENERATION_SEED_POLL_MS = 30_000;
/** Until the first load succeeds the seed controls and Generate are disabled, so retry fast. */
export const GENERATION_SEED_RETRY_MS = 3000;
/** Typing in the seed box coalesces into one write, which persists settings.json once. */
export const GENERATION_SEED_SAVE_DEBOUNCE_MS = 300;

/** Latest cached seed, or undefined before the first load. Safe to call from event handlers. */
export function readCachedGenerationSeed(
  queryClient: QueryClient,
): GenerationSeedState | undefined {
  return queryClient.getQueryData<GenerationSeedState>(GENERATION_SEED_QUERY_KEY);
}

class GenerationSeedWriter {
  private pending: Partial<GenerationSeedState> | null = null;
  private timer: ReturnType<typeof setTimeout> | null = null;
  private inFlight = 0;
  private chain: Promise<unknown> = Promise.resolve();

  private readonly queryClient: QueryClient;

  constructor(queryClient: QueryClient) {
    this.queryClient = queryClient;
  }

  /** True while an edit is queued or on the wire, i.e. the server copy is older than the cache. */
  hasUnsavedEdits(): boolean {
    return this.pending !== null || this.inFlight > 0;
  }

  save(api: GenerationSeedApi, patch: Partial<GenerationSeedState>): void {
    const current = readCachedGenerationSeed(this.queryClient);
    if (!current) return;
    // A poll already on the wire would resolve with the pre-edit value and overwrite this one.
    void this.queryClient.cancelQueries({ queryKey: GENERATION_SEED_QUERY_KEY });
    this.queryClient.setQueryData<GenerationSeedState>(GENERATION_SEED_QUERY_KEY, {
      ...current,
      ...patch,
    });
    this.pending = { ...this.pending, ...patch };
    if (this.timer !== null) clearTimeout(this.timer);
    this.timer = setTimeout(() => this.flush(api), GENERATION_SEED_SAVE_DEBOUNCE_MS);
  }

  private flush(api: GenerationSeedApi): void {
    const patch = this.pending;
    this.timer = null;
    this.pending = null;
    if (patch === null) return;
    this.inFlight += 1;
    // Writes go out one at a time so a slow earlier one can't land after a later one.
    this.chain = this.chain
      .then(async () => unwrapApiResult(await api.updateGenerationSeed(patch)))
      .then(
        () => {
          this.inFlight -= 1;
        },
        () => {
          this.inFlight -= 1;
          // Drop the optimistic value and let the server's copy win.
          void this.queryClient.invalidateQueries({ queryKey: GENERATION_SEED_QUERY_KEY });
        },
      );
  }
}

const writers = new WeakMap<QueryClient, GenerationSeedWriter>();

function writerFor(queryClient: QueryClient): GenerationSeedWriter {
  let writer = writers.get(queryClient);
  if (!writer) {
    writer = new GenerationSeedWriter(queryClient);
    writers.set(queryClient, writer);
  }
  return writer;
}

export function saveGenerationSeed(
  queryClient: QueryClient,
  api: GenerationSeedApi,
  patch: Partial<GenerationSeedState>,
): void {
  writerFor(queryClient).save(api, patch);
}

/** Query function: the server's copy, unless a local edit has not reached the server yet. */
export async function fetchGenerationSeed(
  queryClient: QueryClient,
  api: GenerationSeedApi,
): Promise<GenerationSeedState> {
  const cached = readCachedGenerationSeed(queryClient);
  if (cached && writerFor(queryClient).hasUnsavedEdits()) return cached;
  const { seed, locked } = unwrapApiResult(await api.getGenerationSeed());
  return { seed, locked };
}
