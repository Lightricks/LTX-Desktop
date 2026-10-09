import assert from "node:assert/strict";
import { describe, it } from "node:test";

import type { Generation } from "../lib/resultsFeedModel.ts";
import {
  DESKTOP_QUEUE_POLL_INTERVAL_MS,
  DESKTOP_GENERATION_POLLING_POLICY,
  GENERATION_ERROR_BACKOFF_MAX_MS,
  GENERATION_ERROR_BACKOFF_START_MS,
  CLOSED_QUEUE_POLL_INTERVAL_MS,
  GENERATION_LIST_POLL_INTERVAL_MS,
  GENERATION_POLLING_QUERY_OPTIONS,
  IDLE_QUEUE_POLL_INTERVAL_MS,
  generationErrorBackoffMs,
  generationRefetchInterval,
  queueRefetchInterval,
  REMOTE_QUEUE_POLL_INTERVAL_MS,
  REMOTE_GENERATION_POLLING_POLICY,
} from "./generationPollingPolicy.ts";

function row(status: Generation["status"]): Generation {
  return { id: status, status } as Generation;
}

describe("host generation poll intervals", () => {
  it("polls the generation list every 10s on Desktop and Remote while a generation is in flight", () => {
    const running = { data: [row("running")], error: null, fetchFailureCount: 0 };

    assert.equal(
      generationRefetchInterval(DESKTOP_GENERATION_POLLING_POLICY, running, false),
      GENERATION_LIST_POLL_INTERVAL_MS,
    );
    assert.equal(
      generationRefetchInterval(REMOTE_GENERATION_POLLING_POLICY, running, false),
      GENERATION_LIST_POLL_INTERVAL_MS,
    );
  });

  it("uses the list interval while keepPolling even if the list is idle", () => {
    const idle = { data: [row("succeeded")], error: null, fetchFailureCount: 0 };

    assert.equal(
      generationRefetchInterval(DESKTOP_GENERATION_POLLING_POLICY, idle, true),
      GENERATION_LIST_POLL_INTERVAL_MS,
    );
    assert.equal(
      generationRefetchInterval(REMOTE_GENERATION_POLLING_POLICY, idle, true),
      GENERATION_LIST_POLL_INTERVAL_MS,
    );
  });
});

describe("generation error backoff", () => {
  it("starts at 2s and doubles until the 30s cap", () => {
    assert.equal(generationErrorBackoffMs(1), GENERATION_ERROR_BACKOFF_START_MS);
    assert.equal(generationErrorBackoffMs(2), 4000);
    assert.equal(generationErrorBackoffMs(3), 8000);
    assert.equal(generationErrorBackoffMs(4), 16_000);
    assert.equal(generationErrorBackoffMs(5), GENERATION_ERROR_BACKOFF_MAX_MS);
    assert.equal(generationErrorBackoffMs(12), GENERATION_ERROR_BACKOFF_MAX_MS);
  });

  it("backs off on query errors instead of using the active interval", () => {
    const failed = {
      data: [row("running")],
      error: new Error("offline"),
      fetchFailureCount: 3,
    };

    assert.equal(
      generationRefetchInterval(DESKTOP_GENERATION_POLLING_POLICY, failed, true),
      8000,
    );
    assert.equal(
      generationRefetchInterval(REMOTE_GENERATION_POLLING_POLICY, failed, false),
      8000,
    );
  });
});

describe("generation idle stop", () => {
  it("stops when there is no in-flight row and create is not keeping the poll alive", () => {
    const idle = {
      data: [row("succeeded"), row("failed")],
      error: null,
      fetchFailureCount: 0,
    };
    const empty = { data: [] as Generation[], error: null, fetchFailureCount: 0 };

    assert.equal(
      generationRefetchInterval(DESKTOP_GENERATION_POLLING_POLICY, idle, false),
      false,
    );
    assert.equal(
      generationRefetchInterval(REMOTE_GENERATION_POLLING_POLICY, idle, false),
      false,
    );
    assert.equal(
      generationRefetchInterval(DESKTOP_GENERATION_POLLING_POLICY, empty, false),
      false,
    );
  });

  it("keeps polling before the first successful snapshot", () => {
    const pending = { data: undefined, error: null, fetchFailureCount: 0 };

    assert.equal(
      generationRefetchInterval(DESKTOP_GENERATION_POLLING_POLICY, pending, false),
      GENERATION_LIST_POLL_INTERVAL_MS,
    );
  });
});

describe("generation polling focus and background options", () => {
  it("refetches on focus and reconnect, and does not poll while hidden", () => {
    assert.equal(GENERATION_POLLING_QUERY_OPTIONS.refetchOnWindowFocus, true);
    assert.equal(GENERATION_POLLING_QUERY_OPTIONS.refetchOnReconnect, true);
    assert.equal(
      GENERATION_POLLING_QUERY_OPTIONS.refetchIntervalInBackground,
      false,
    );
  });
});

describe("open queue panel polling", () => {
  const empty = { error: null, fetchFailureCount: 0 };
  const pending = { error: null, fetchFailureCount: 0 };

  it("polls the queue while the panel is open, including an empty queue", () => {
    assert.equal(
      queueRefetchInterval(DESKTOP_GENERATION_POLLING_POLICY, empty, true),
      DESKTOP_QUEUE_POLL_INTERVAL_MS,
    );
    assert.equal(
      queueRefetchInterval(REMOTE_GENERATION_POLLING_POLICY, pending, true),
      REMOTE_QUEUE_POLL_INTERVAL_MS,
    );
  });

  it("slows down once a loaded queue has nothing active or queued", () => {
    const loadedEmpty = { data: { active: null, queued: [] }, error: null, fetchFailureCount: 0 };
    const running = { data: { active: {}, queued: [] }, error: null, fetchFailureCount: 0 };
    const waiting = { data: { active: null, queued: [{}] }, error: null, fetchFailureCount: 0 };

    assert.equal(
      queueRefetchInterval(DESKTOP_GENERATION_POLLING_POLICY, loadedEmpty, true),
      IDLE_QUEUE_POLL_INTERVAL_MS,
    );
    assert.equal(
      queueRefetchInterval(REMOTE_GENERATION_POLLING_POLICY, loadedEmpty, true),
      IDLE_QUEUE_POLL_INTERVAL_MS,
    );
    assert.equal(
      queueRefetchInterval(DESKTOP_GENERATION_POLLING_POLICY, running, true),
      DESKTOP_QUEUE_POLL_INTERVAL_MS,
    );
    assert.equal(
      queueRefetchInterval(DESKTOP_GENERATION_POLLING_POLICY, waiting, true),
      DESKTOP_QUEUE_POLL_INTERVAL_MS,
    );
  });

  it("polls a closed queue panel slowly", () => {
    assert.equal(
      queueRefetchInterval(DESKTOP_GENERATION_POLLING_POLICY, empty, false),
      CLOSED_QUEUE_POLL_INTERVAL_MS,
    );
    assert.equal(
      queueRefetchInterval(REMOTE_GENERATION_POLLING_POLICY, empty, false),
      CLOSED_QUEUE_POLL_INTERVAL_MS,
    );
  });

  it("uses the Desktop and Remote polling intervals supplied by ExploreRuntime", () => {
    assert.equal(
      queueRefetchInterval(DESKTOP_GENERATION_POLLING_POLICY, empty, true),
      DESKTOP_GENERATION_POLLING_POLICY.queueActiveIntervalMs,
    );
    assert.equal(
      queueRefetchInterval(REMOTE_GENERATION_POLLING_POLICY, empty, true),
      REMOTE_GENERATION_POLLING_POLICY.queueActiveIntervalMs,
    );
    assert.equal(DESKTOP_GENERATION_POLLING_POLICY.queueActiveIntervalMs, 1000);
    assert.equal(REMOTE_GENERATION_POLLING_POLICY.queueActiveIntervalMs, 2000);
  });

  it("backs off background watchers too, but never faster than their normal rate", () => {
    const early = { error: new Error("offline"), fetchFailureCount: 1 };
    const late = { error: new Error("offline"), fetchFailureCount: 5 };

    assert.equal(
      queueRefetchInterval(DESKTOP_GENERATION_POLLING_POLICY, early, false),
      CLOSED_QUEUE_POLL_INTERVAL_MS,
    );
    assert.equal(
      queueRefetchInterval(DESKTOP_GENERATION_POLLING_POLICY, late, false),
      GENERATION_ERROR_BACKOFF_MAX_MS,
    );
  });

  it("backs off on query errors instead of using the open-panel interval", () => {
    const failed = { error: new Error("offline"), fetchFailureCount: 3 };

    assert.equal(
      queueRefetchInterval(DESKTOP_GENERATION_POLLING_POLICY, failed, true),
      8000,
    );
    assert.equal(
      queueRefetchInterval(REMOTE_GENERATION_POLLING_POLICY, failed, false),
      CLOSED_QUEUE_POLL_INTERVAL_MS,
    );
  });
});
