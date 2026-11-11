import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { ApiResultError } from "../../lib/unwrapApiResult.ts";
import {
  formatActiveQueueProgress,
  formatActiveQueueProgressDetail,
  activeQueueRowDetail,
  activeQueueTitleStatus,
  getQueuePanelState,
  isActiveCancelDisabled,
  isQueueReorderConflict,
  queueLaneLabel,
  queueRowBadges,
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
        active: [{ generation: { id: "active", status: "cancelling" } }],
        queued,
      }),
      {
        activeIsFixed: true,
        queuedCancelDisabled: false,
        sortableIds: ["queued-a", "queued-b", "queued-c"],
        sortingDisabled: false,
      },
    );
  });

  it("keeps queued cancel enabled while the active job is running", () => {
    const state = getQueuePanelState({
      active: [{ generation: { id: "active", status: "running" } }],
      queued,
    });
    assert.equal(
      isActiveCancelDisabled({ generation: { id: "active", status: "running" } }, false),
      false,
    );
    assert.equal(state.queuedCancelDisabled, false);
  });

  it("disables active and queued cancel only while a cancel request is pending", () => {
    const pending = getQueuePanelState(
      {
        active: [{ generation: { id: "active", status: "running" } }],
        queued,
      },
      { isCancelPending: true },
    );
    assert.equal(
      isActiveCancelDisabled({ generation: { id: "active", status: "running" } }, true),
      true,
    );
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

  it("formats active progress as a percent derived from denoise steps", () => {
    assert.equal(
      formatActiveQueueProgress({
        progress: 42,
        currentStep: 3,
        totalSteps: 11,
      }),
      "27%",
    );
  });

  it("shows no percent before a denoise step, the same as the result placeholder", () => {
    const loading = { progress: 5, currentStep: null, totalSteps: null };
    const encoding = { progress: 10, currentStep: null, totalSteps: null };
    const stepZero = { progress: 15, currentStep: 0, totalSteps: 8 };

    for (const phase of [loading, encoding, stepZero]) {
      assert.equal(formatActiveQueueProgress(phase), null);
      assert.equal(formatActiveQueueProgressDetail(phase), "Waiting for progress…");
    }
  });

  it("gives a provider job its provider status as the row detail, not a stuck waiting text", () => {
    // A provider job reports a phase with no step count, so the percent is not known.
    const phaseOnly = { progress: 15, currentStep: null, totalSteps: null };

    assert.equal(activeQueueRowDetail(phaseOnly, "pending"), "Waiting…");
    assert.equal(activeQueueRowDetail(phaseOnly, "processing"), "Generating…");
    assert.equal(activeQueueRowDetail(null, "processing"), "Generating…");
    assert.equal(activeQueueRowDetail(phaseOnly, null), "Waiting for progress…");
    assert.equal(
      activeQueueRowDetail({ progress: 10, currentStep: 3, totalSteps: 10 }, "processing"),
      "Step 3 of 10 (30%)",
    );
  });

  it("shows the same percent as the placeholder once a step has run", () => {
    const progress = { progress: 10, currentStep: 3, totalSteps: 10 };
    const placeholder = progressPercentByGenerationId({
      active: [{ generation: { id: "a" }, progress }],
      queued: [],
    }).get("a");

    assert.equal(formatActiveQueueProgress(progress), `${placeholder}%`);
    assert.equal(formatActiveQueueProgressDetail(progress), "Step 3 of 10 (30%)");
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
      active: [
        {
          generation: { id: "active" },
          progress: { progress: 10, currentStep: 3, totalSteps: 10 },
        },
      ],
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

  it("keeps every running row fixed and shows progress for each", () => {
    // Bug caught: a second running job (another lane) loses its row or its progress.
    const active = [
      {
        generation: { id: "local-job", status: "running" },
        progress: { progress: 10, currentStep: 5, totalSteps: 10 },
      },
      { generation: { id: "fal-job", status: "running" }, progress: null },
    ];
    const state = getQueuePanelState({ active, queued });
    assert.equal(state.activeIsFixed, true);
    assert.deepEqual(state.sortableIds, ["queued-a", "queued-b", "queued-c"]);
    const percents = progressPercentByGenerationId({ active, queued: [] });
    assert.equal(percents.get("local-job"), 50);
    assert.equal(percents.has("fal-job"), false);
  });

  it("cancels one running row without disabling the other", () => {
    // Bug caught: a cancelling row disables Cancel on every running row.
    assert.equal(
      isActiveCancelDisabled({ generation: { id: "a", status: "cancelling" } }, false),
      true,
    );
    assert.equal(
      isActiveCancelDisabled({ generation: { id: "b", status: "running" } }, false),
      false,
    );
  });

  it("labels a provider lane and leaves the local lane unlabelled", () => {
    // Bug caught: a later row starts first and the user sees no reason.
    assert.equal(queueLaneLabel("local"), null);
    assert.equal(queueLaneLabel(undefined), null);
    assert.equal(queueLaneLabel("fal"), "fal");
  });

  it("shows the lane label once when the provider badge already says it", () => {
    // Bug caught: a fal row showed the "fal" badge twice (and a duplicate React key).
    assert.deepEqual(queueRowBadges(["2K", "fal"], "fal"), ["2K", "fal"]);
    assert.deepEqual(queueRowBadges(["2K"], "fal"), ["2K", "fal"]);
    assert.deepEqual(queueRowBadges(["Local"], "local"), ["Local"]);
  });

  it("names the provider status of a running provider job", () => {
    // Bug caught: a provider job that waits in the provider queue reads as "Generating".
    assert.equal(activeQueueTitleStatus("pending"), "Waiting");
    assert.equal(activeQueueTitleStatus("processing"), "Generating");
    assert.equal(activeQueueTitleStatus(null), null);
  });

  it("disables sorting while a reorder mutation is pending", () => {
    assert.equal(
      getQueuePanelState({ active: [], queued }, { isReordering: true })
        .sortingDisabled,
      true,
    );
  });

  it("opens any queue job on its feature result, or returns null for an unknown feature", () => {
    const features = [
      { id: "text-to-video", path: "/text-to-video" },
      { id: "custom-lora", path: "/custom-lora" },
    ];
    assert.deepEqual(
      resultLocationForFeature("text-to-video", "job-a", features),
      { pathname: "/text-to-video", search: "?result=job-a" },
    );
    assert.deepEqual(
      resultLocationForFeature("custom-lora", "job-b", features, {
        params: { customLoraId: "ink-wash" },
      }),
      { pathname: "/custom-lora/ink-wash", search: "?result=job-b" },
    );
    assert.equal(
      resultLocationForFeature("custom-lora", "job-b", features, {}),
      null,
    );
    assert.equal(resultLocationForFeature("missing", "job-a", []), null);
  });
});
