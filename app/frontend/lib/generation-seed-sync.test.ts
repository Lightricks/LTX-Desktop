import assert from "node:assert/strict";
import { afterEach, beforeEach, describe, it, mock } from "node:test";
import { QueryClient } from "@tanstack/react-query";

import {
  GENERATION_SEED_QUERY_KEY,
  GENERATION_SEED_SAVE_DEBOUNCE_MS,
  fetchGenerationSeed,
  readCachedGenerationSeed,
  saveGenerationSeed,
  type GenerationSeedApi,
  type GenerationSeedState,
} from "./generation-seed-sync.ts";

type Server = { state: GenerationSeedState; writes: Partial<GenerationSeedState>[] };

function fakeApi(server: Server, options: { failWrites?: boolean } = {}) {
  const gets: Array<() => void> = [];
  const api = {
    getGenerationSeed: () =>
      new Promise((resolve) => {
        const snapshot = { ...server.state };
        gets.push(() => resolve({ ok: true, data: snapshot }));
      }),
    updateGenerationSeed: async (patch: Partial<GenerationSeedState>) => {
      server.writes.push(patch);
      if (options.failWrites) {
        return { ok: false, error: { message: "boom" } };
      }
      server.state = { ...server.state, ...patch };
      return { ok: true, data: server.state };
    },
  } as unknown as GenerationSeedApi;
  return { api, gets };
}

async function settle() {
  for (let i = 0; i < 5; i += 1) await Promise.resolve();
}

describe("generation-seed-sync", () => {
  let queryClient: QueryClient;

  beforeEach(() => {
    mock.timers.enable({ apis: ["setTimeout"] });
    queryClient = new QueryClient();
    queryClient.setQueryData(GENERATION_SEED_QUERY_KEY, { seed: 1, locked: false });
  });

  afterEach(() => {
    mock.timers.reset();
    queryClient.clear();
  });

  it("a poll that was already on the wire cannot overwrite a newer local edit", async () => {
    const server: Server = { state: { seed: 1, locked: false }, writes: [] };
    const { api, gets } = fakeApi(server);

    const poll = queryClient
      .fetchQuery({
        queryKey: GENERATION_SEED_QUERY_KEY,
        queryFn: () => fetchGenerationSeed(queryClient, api),
        staleTime: 0,
      })
      .catch(() => undefined);
    await settle();
    saveGenerationSeed(queryClient, api, { seed: 99 });
    gets.forEach((resolve) => resolve());
    await poll;

    assert.deepEqual(readCachedGenerationSeed(queryClient), { seed: 99, locked: false });
  });

  it("a poll that starts before the edit reaches the server keeps the local value", async () => {
    const server: Server = { state: { seed: 1, locked: false }, writes: [] };
    const { api, gets } = fakeApi(server);
    saveGenerationSeed(queryClient, api, { seed: 99 });

    const polled = await fetchGenerationSeed(queryClient, api);

    assert.deepEqual(polled, { seed: 99, locked: false });
    assert.equal(gets.length, 0);
  });

  it("typing a seed digit by digit is one write, with the last value and the lock change", async () => {
    const server: Server = { state: { seed: 1, locked: false }, writes: [] };
    const { api } = fakeApi(server);

    saveGenerationSeed(queryClient, api, { seed: 4 });
    saveGenerationSeed(queryClient, api, { seed: 42 });
    saveGenerationSeed(queryClient, api, { locked: true });
    saveGenerationSeed(queryClient, api, { seed: 421 });
    assert.equal(server.writes.length, 0);
    mock.timers.tick(GENERATION_SEED_SAVE_DEBOUNCE_MS);
    await settle();

    assert.deepEqual(server.writes, [{ seed: 421, locked: true }]);
  });

  it("a rejected write marks the cached seed stale so the server's copy comes back", async () => {
    const server: Server = { state: { seed: 1, locked: false }, writes: [] };
    const { api } = fakeApi(server, { failWrites: true });

    saveGenerationSeed(queryClient, api, { seed: 99 });
    mock.timers.tick(GENERATION_SEED_SAVE_DEBOUNCE_MS);
    await settle();

    assert.equal(
      queryClient.getQueryState(GENERATION_SEED_QUERY_KEY)?.isInvalidated,
      true,
    );
  });

  it("ignores an edit made before the seed has loaded", () => {
    const empty = new QueryClient();
    const server: Server = { state: { seed: 1, locked: false }, writes: [] };
    const { api } = fakeApi(server);

    saveGenerationSeed(empty, api, { seed: 99 });
    mock.timers.tick(GENERATION_SEED_SAVE_DEBOUNCE_MS);

    assert.equal(readCachedGenerationSeed(empty), undefined);
    assert.equal(server.writes.length, 0);
  });
});
