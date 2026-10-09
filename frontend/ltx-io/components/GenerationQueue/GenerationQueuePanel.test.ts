import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { ApiResultError } from "../../lib/unwrapApiResult.ts";
import {
  formatActiveQueueProgress,
  getQueuePanelState,
  isQueueReorderConflict,
  progressPercentByGenerationId,
  orderedQueueInputAssets,
  reorderQueuedEntries,
  requestForQueueOrder,
  resultLocationForFeature,
} from "./generationQueuePanelModel.ts";

const queued = [
  { generation: { id: "queued-a", status: "queued" } },
  { generation: { id: "queued-b", status: "queued" } },
  { generation: { id: "queued-c", status: "queued" } },
];

describe("GenerationQueuePanel queue behavior", () => {
  it("renders the active cancelling item as fixed without disabling queued cancel", () => {
    assert.deepEqual(
      getQueuePanelState({
        active: { generation: { id: "active", status: "cancelling" } },
        queued,
      }),
      {
        activeIsFixed: true,
        activeCancelDisabled: true,
        queuedCancelDisabled: false,
        sortableIds: ["queued-a", "queued-b", "queued-c"],
        sortingDisabled: false,
      },
    );
  });

  it("keeps queued cancel enabled while the active job is running", () => {
    const state = getQueuePanelState({
      active: { generation: { id: "active", status: "running" } },
      queued,
    });
    assert.equal(state.activeCancelDisabled, false);
    assert.equal(state.queuedCancelDisabled, false);
  });

  it("disables active and queued cancel only while a cancel request is pending", () => {
    const pending = getQueuePanelState(
      {
        active: { generation: { id: "active", status: "running" } },
        queued,
      },
      { isCancelPending: true },
    );
    assert.equal(pending.activeCancelDisabled, true);
    assert.equal(pending.queuedCancelDisabled, true);
  });

  it("orders queue input assets from the generation spec", () => {
    assert.deepEqual(
      orderedQueueInputAssets({
        generation: {
          id: "gen",
          feature: "audio-to-video",
          status: "queued",
          spec: {
            inputs: {
              audio: { assetId: "audio" },
              startFrame: { assetId: "frame" },
            },
          },
        },
        input_assets: [
          { id: "audio", media_kind: "audio", name: "a" },
          { id: "frame", media_kind: "image", name: "f" },
        ],
      }).map((asset) => asset.id),
      ["frame", "audio"],
    );
  });

  it("allows only queued rows to be reordered", () => {
    assert.deepEqual(
      reorderQueuedEntries(queued, "queued-c", "queued-a").map(
        (entry) => entry.generation.id,
      ),
      ["queued-c", "queued-a", "queued-b"],
    );
  });

  it("derives the persisted move from final controlled order", () => {
    const finalOrder = reorderQueuedEntries(queued, "queued-a", "queued-c");
    assert.deepEqual(requestForQueueOrder(finalOrder, "queued-a"), {
      generation_id: "queued-a",
      before_generation_id: null,
    });
  });

  it("refreshes the queue after a reorder conflict", () => {
    assert.equal(
      isQueueReorderConflict(
        new ApiResultError("INVALID_GENERATION_STATUS", {
          code: "INVALID_GENERATION_STATUS",
          status: 409,
        }),
      ),
      true,
    );
    assert.equal(
      isQueueReorderConflict(
        new ApiResultError("Unprocessable", {
          code: "VALIDATION_ERROR",
          status: 422,
        }),
      ),
      false,
    );
  });

  it("formats active progress as percent, derived from steps when available", () => {
    assert.equal(
      formatActiveQueueProgress({
        progress: 42,
        currentStep: null,
        totalSteps: null,
      }),
      "42%",
    );
    assert.equal(
      formatActiveQueueProgress({
        progress: 42,
        currentStep: 3,
        totalSteps: 11,
      }),
      "27%",
    );
  });

  it("caps a completed step count at 95 percent", () => {
    assert.equal(
      formatActiveQueueProgress({
        progress: 100,
        currentStep: 11,
        totalSteps: 11,
      }),
      "95%",
    );
  });

  it("maps denoise steps onto generation ids and skips phase placeholders", () => {
    const percents = progressPercentByGenerationId({
      active: {
        generation: { id: "active" },
        progress: { progress: 10, currentStep: 3, totalSteps: 10 },
      },
      queued: [
        { generation: { id: "waiting" }, progress: null },
        {
          generation: { id: "encoding" },
          progress: { progress: 10, currentStep: null, totalSteps: null },
        },
        {
          generation: { id: "step-zero" },
          progress: { progress: 15, currentStep: 0, totalSteps: 8 },
        },
      ],
    });

    assert.equal(percents.get("active"), 30);
    assert.equal(percents.has("encoding"), false);
    assert.equal(percents.has("step-zero"), false);
    assert.equal(percents.has("waiting"), false);
    assert.equal(progressPercentByGenerationId(undefined).size, 0);
  });

  it("disables sorting while a reorder mutation is pending", () => {
    assert.equal(
      getQueuePanelState({ active: null, queued }, { isReordering: true })
        .sortingDisabled,
      true,
    );
  });

  it("opens a finished job on its feature result", () => {
    assert.deepEqual(
      resultLocationForFeature("text-to-video", "job-a", [
        { id: "text-to-video", path: "/text-to-video" },
      ]),
      { pathname: "/text-to-video", search: "?result=job-a" },
    );
    assert.equal(resultLocationForFeature("missing", "job-a", []), null);
  });
});
