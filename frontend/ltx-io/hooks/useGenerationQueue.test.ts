import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";

import { createApiClient } from "../../lib/api-client.ts";
import {
  DESKTOP_GENERATION_POLLING_POLICY,
  CLOSED_QUEUE_POLL_INTERVAL_MS,
  GENERATION_POLLING_QUERY_OPTIONS,
  IDLE_QUEUE_POLL_INTERVAL_MS,
  queueRefetchInterval,
  REMOTE_GENERATION_POLLING_POLICY,
} from "../runtime/generationPollingPolicy.ts";
import {
  fetchGenerationQueue,
  fetchQueueSnapshotAfterReorder,
  generationQueryKeys,
  invalidateAffectedGenerationQueries,
  replaceGenerationQueueCache,
  type ExploreQueueSnapshot,
  type ReorderGenerationQueueRequest,
} from "./generationQueryKeys.ts";
import { generationQueueQueryOptions } from "./useGenerationQueue.ts";
import { useReorderGenerationQueue } from "./useReorderGenerationQueue.ts";

const generation = {
  id: "gen-1",
  created_at: 1,
  queued_at: 1,
  attempt_count: 0,
  contract_version: 1,
  feature: "text-to-video",
  status: "queued",
  spec: {},
  outputs: [],
};

const emptySnapshot = {
  active: null,
  queued: [],
  done: [],
  failed: [],
  unseen_ids: [],
};

const optimisticSnapshot = {
  active: null,
  queued: [
    { generation: { ...generation, id: "gen-a" }, input_assets: [], progress: null },
    { generation: { ...generation, id: "gen-b" }, input_assets: [], progress: null },
  ],
  done: [],
  failed: [],
  unseen_ids: [],
};

const serverSnapshot = {
  active: null,
  queued: [
    { generation: { ...generation, id: "gen-b" }, input_assets: [], progress: null },
    { generation: { ...generation, id: "gen-a" }, input_assets: [], progress: null },
  ],
  done: [],
  failed: [],
  unseen_ids: [],
};

describe("generation queue query key", () => {
  it("uses a global generation-queue key", () => {
    assert.deepEqual(generationQueryKeys.queue, ["generation-queue"]);
  });
});

describe("open queue panel polling", () => {
  it("polls an open panel fast with work in the queue and slowly when it is empty", () => {
    const options = generationQueueQueryOptions({
      api: createApiClient(async () => new Response(JSON.stringify(emptySnapshot), { status: 200 })),
      generationPolling: DESKTOP_GENERATION_POLLING_POLICY,
      isLive: true,
    });
    const emptyQuery = {
      state: { data: emptySnapshot, error: null, fetchFailureCount: 0 },
    };
    const busyQuery = {
      state: { data: optimisticSnapshot, error: null, fetchFailureCount: 0 },
    };

    assert.equal(options.enabled, true);
    assert.equal(options.queryKey, generationQueryKeys.queue);
    assert.equal(options.refetchInterval(emptyQuery), IDLE_QUEUE_POLL_INTERVAL_MS);
    assert.equal(
      options.refetchInterval(busyQuery),
      DESKTOP_GENERATION_POLLING_POLICY.queueActiveIntervalMs,
    );
    assert.equal(
      queueRefetchInterval(DESKTOP_GENERATION_POLLING_POLICY, busyQuery.state, true),
      DESKTOP_GENERATION_POLLING_POLICY.queueActiveIntervalMs,
    );
    assert.equal(options.refetchOnWindowFocus, GENERATION_POLLING_QUERY_OPTIONS.refetchOnWindowFocus);
    assert.equal(options.refetchOnReconnect, GENERATION_POLLING_QUERY_OPTIONS.refetchOnReconnect);
    assert.equal(
      options.refetchIntervalInBackground,
      GENERATION_POLLING_QUERY_OPTIONS.refetchIntervalInBackground,
    );
  });

  it("keeps the queue query enabled and polls slowly while the panel is closed", () => {
    const options = generationQueueQueryOptions({
      api: createApiClient(async () => new Response(JSON.stringify(emptySnapshot), { status: 200 })),
      generationPolling: DESKTOP_GENERATION_POLLING_POLICY,
      isLive: false,
    });
    const emptyQuery = {
      state: { data: emptySnapshot, error: null, fetchFailureCount: 0 },
    };

    assert.equal(options.enabled, true);
    assert.equal(options.refetchInterval(emptyQuery), CLOSED_QUEUE_POLL_INTERVAL_MS);
    assert.equal(
      queueRefetchInterval(DESKTOP_GENERATION_POLLING_POLICY, emptyQuery.state, false),
      CLOSED_QUEUE_POLL_INTERVAL_MS,
    );
    assert.equal(options.refetchOnWindowFocus, GENERATION_POLLING_QUERY_OPTIONS.refetchOnWindowFocus);
    assert.equal(options.refetchOnReconnect, GENERATION_POLLING_QUERY_OPTIONS.refetchOnReconnect);
  });

  it("uses the Desktop and Remote polling intervals supplied by ExploreRuntime", () => {
    const desktop = generationQueueQueryOptions({
      api: createApiClient(async () => new Response(JSON.stringify(emptySnapshot), { status: 200 })),
      generationPolling: DESKTOP_GENERATION_POLLING_POLICY,
      isLive: true,
    });
    const remote = generationQueueQueryOptions({
      api: createApiClient(async () => new Response(JSON.stringify(emptySnapshot), { status: 200 })),
      generationPolling: REMOTE_GENERATION_POLLING_POLICY,
      isLive: true,
    });
    const query = {
      state: { data: optimisticSnapshot, error: null, fetchFailureCount: 0 },
    };

    assert.equal(desktop.refetchInterval(query), 1000);
    assert.equal(remote.refetchInterval(query), 2000);
  });
});

describe("generation queue client", () => {
  it("loads the snapshot through the injected API client", async () => {
    const calls: string[] = [];
    const api = createApiClient(async (path) => {
      calls.push(`GET ${path}`);
      return new Response(JSON.stringify(emptySnapshot), { status: 200 });
    });

    const snapshot = await fetchGenerationQueue(api);

    assert.deepEqual(snapshot, emptySnapshot);
    assert.deepEqual(calls, ["GET /api/generation-queue"]);
  });

  it("replaces optimistic order with the returned queue snapshot", async () => {
    const calls: string[] = [];
    const posts: unknown[] = [];
    const staleSnapshot = { active: null, queued: [] };
    const api = createApiClient(async (path, init) => {
      calls.push(`${init?.method ?? "GET"} ${path}`);
      if (path === "/api/generation-queue/reorder") {
        posts.push(JSON.parse(String(init?.body)));
        return new Response(JSON.stringify(serverSnapshot), { status: 200 });
      }
      return new Response(JSON.stringify(staleSnapshot), { status: 200 });
    });
    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
    });
    queryClient.setQueryData(generationQueryKeys.queue, optimisticSnapshot);

    const snapshot = await fetchQueueSnapshotAfterReorder(api, {
      generation_id: "gen-b",
      before_generation_id: "gen-a",
    });
    replaceGenerationQueueCache(queryClient, snapshot);

    assert.deepEqual(posts, [
      { generation_id: "gen-b", before_generation_id: "gen-a" },
    ]);
    assert.deepEqual(calls, ["POST /api/generation-queue/reorder"]);
    assert.deepEqual(
      queryClient.getQueryData(generationQueryKeys.queue),
      serverSnapshot,
    );
    assert.notDeepEqual(
      queryClient.getQueryData(generationQueryKeys.queue),
      optimisticSnapshot,
    );
    assert.notDeepEqual(
      queryClient.getQueryData(generationQueryKeys.queue),
      staleSnapshot,
    );
  });

  it("invalidates affected generation history after create, cancel, or reorder", () => {
    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
    });
    queryClient.setQueryData(generationQueryKeys.list("text-to-video"), [generation]);
    queryClient.setQueryData(generationQueryKeys.list("image-to-video"), [generation]);
    queryClient.setQueryData(generationQueryKeys.queue, optimisticSnapshot);

    invalidateAffectedGenerationQueries(queryClient);

    assert.equal(
      queryClient.getQueryState(generationQueryKeys.list("text-to-video"))?.isInvalidated,
      true,
    );
    assert.equal(
      queryClient.getQueryState(generationQueryKeys.list("image-to-video"))?.isInvalidated,
      true,
    );
    assert.equal(
      queryClient.getQueryState(generationQueryKeys.queue)?.isInvalidated,
      true,
    );
  });

  it("invalidates generation history after a successful reorder mutation", async () => {
    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
    });
    queryClient.setQueryData(generationQueryKeys.list("text-to-video"), [generation]);
    queryClient.setQueryData(generationQueryKeys.queue, optimisticSnapshot);
    const calls: string[] = [];
    const api = createApiClient(async (path, init) => {
      calls.push(`${init?.method ?? "GET"} ${path}`);
      return new Response(JSON.stringify(serverSnapshot), { status: 200 });
    });
    let mutateAsync:
      | ((request: ReorderGenerationQueueRequest) => Promise<ExploreQueueSnapshot>)
      | undefined;

    function MutationCapture() {
      mutateAsync = useReorderGenerationQueue({ api }).mutateAsync;
      return null;
    }

    renderToStaticMarkup(
      createElement(
        QueryClientProvider,
        { client: queryClient },
        createElement(MutationCapture),
      ),
    );
    assert.ok(mutateAsync);
    await mutateAsync({
      generation_id: "gen-b",
      before_generation_id: "gen-a",
    });

    assert.deepEqual(calls, ["POST /api/generation-queue/reorder"]);
    assert.deepEqual(
      queryClient.getQueryData(generationQueryKeys.queue),
      serverSnapshot,
    );
    assert.equal(
      queryClient.getQueryState(generationQueryKeys.list("text-to-video"))?.isInvalidated,
      true,
    );
  });
});
