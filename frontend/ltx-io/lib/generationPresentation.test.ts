import assert from "node:assert/strict";
import { describe, it } from "node:test";

import type { ExploreAsset } from "../../lib/explore-contract.ts";

import { generationPresentation } from "./generationPresentation.ts";

function videoOutput(width: number, height: number): ExploreAsset {
  return {
    id: "output-1",
    media_kind: "video",
    origin: "generated",
    mime_type: "video/mp4",
    name: "output.mp4",
    created_at: 0,
    metadata: {
      mediaType: "video",
      metadata: { durationMs: 4000, width, height, sizeBytes: 1, audioStreamCount: 0 },
    },
  };
}

describe("generationPresentation label of a custom LoRA run", () => {
  const run = (params: Record<string, unknown>, feature = "custom-lora") =>
    generationPresentation({ id: "g", feature, status: "queued", spec: { params } }).label;

  it("is the name the user gave the LoRA, for a desktop run and a remote run", () => {
    assert.equal(run({ loras: [{ ref: "r", scale: 1, customLoraId: "a", displayName: "01" }] }), "01");
    assert.equal(run({ customLoraId: "a", customLoraName: "02", scale: 1 }), "02");
  });

  it("falls back to the feature title when the run has no name", () => {
    assert.equal(run({ loras: [{ ref: "r", scale: 1, customLoraId: "a" }] }), "Custom LoRA");
    assert.equal(run({ customLoraName: "  " }), "Custom LoRA");
  });

  it("does not rename another feature that carries a LoRA name", () => {
    assert.equal(run({ loras: [{ displayName: "Cozy" }] }, "cozy-felt"), "Cozy Felt");
  });
});

describe("generationPresentation", () => {
  it("derives prompt, badges, and image input asset IDs from a generation", () => {
    const presentation = generationPresentation({
      id: "generation-1",
      feature: "image-to-video",
      status: "queued",
      spec: {
        inputs: {
          startFrame: { assetId: "image-start" },
          endFrame: { assetId: "image-end" },
        },
        params: {
          prompt: "A fox runs through snow",
          model: "fast",
          resolution: "720p",
          duration: 5,
          fps: 24,
          aspectRatio: "16:9",
        },
      },
    });

    assert.equal(presentation.prompt, "A fox runs through snow");
    assert.deepEqual(presentation.badges, [
      "LTX Fast",
      "720p",
      "5s",
      "24 fps",
      "16:9",
    ]);
    assert.deepEqual(presentation.inputAssetIds, [
      "image-start",
      "image-end",
    ]);
    assert.deepEqual(presentation.imageInputAssetIds, [
      "image-start",
      "image-end",
    ]);
  });

  it("orders audio-to-video input assets as start frame then audio", () => {
    const presentation = generationPresentation({
      id: "generation-a2v",
      feature: "audio-to-video",
      status: "queued",
      spec: {
        inputs: {
          audio: { assetId: "audio-1" },
          startFrame: { assetId: "frame-1" },
        },
        params: { prompt: "Drums" },
      },
    });

    assert.deepEqual(presentation.inputAssetIds, ["frame-1", "audio-1"]);
  });

  it("rounds duration badges to one decimal without trailing .0", () => {
    const presentation = generationPresentation({
      feature: "retake",
      spec: {
        params: {
          prompt: "the dinosaur hits the ball",
          model: "ltx-2.5-fast",
          duration: 4.724674999999999995,
        },
      },
    });
    assert.deepEqual(presentation.badges, ["LTX 2.5 Fast", "4.7s"]);
  });

  it("keeps whole-second durations compact", () => {
    const presentation = generationPresentation({
      feature: "extend",
      spec: {
        params: {
          prompt: "the scene continues",
          model: "ltx-2.5-fast",
          duration: 4,
        },
      },
    });
    assert.deepEqual(presentation.badges, ["LTX 2.5 Fast", "4s"]);
  });

  it("appends a seed badge when params.seed is a number", () => {
    const presentation = generationPresentation({
      feature: "text-to-video",
      spec: {
        params: {
          prompt: "a fox",
          model: "ltx-2.5-fast",
          duration: 8,
          fps: 24,
          aspectRatio: "16:9",
          seed: 42,
        },
      },
    });
    assert.deepEqual(presentation.badges, [
      "LTX 2.5 Fast",
      "8s",
      "24 fps",
      "16:9",
      "Seed 42",
    ]);
  });

  for (const [controlType, label] of [
    ["pose", "Pose"],
    ["depth", "Depth"],
    ["canny", "Canny edges"],
  ] as const) {
    it(`shows the ${label} control badge after the model and before the resolution`, () => {
      const presentation = generationPresentation({
        feature: "video-guide",
        spec: {
          params: { prompt: "a dancer", model: "ltx-2.5-fast", resolution: "720p", controlType },
        },
      });
      assert.deepEqual(presentation.badges, ["LTX 2.5 Fast", label, "720p"]);
    });
  }

  it("shows no control badge for a run without controlType, such as an old run or another recipe", () => {
    const presentation = generationPresentation({
      feature: "restore",
      spec: { params: { prompt: "a dancer", model: "ltx-2.5-fast", resolution: "720p" } },
    });
    assert.deepEqual(presentation.badges, ["LTX 2.5 Fast", "720p"]);
  });

  it("ignores a controlType that is not a control type instead of printing it", () => {
    const presentation = generationPresentation({
      feature: "video-guide",
      spec: {
        params: { prompt: "a dancer", model: "ltx-2.5-fast", resolution: "720p", controlType: "sketch" },
      },
    });
    assert.deepEqual(presentation.badges, ["LTX 2.5 Fast", "720p"]);
  });

  it("labels retake and extend instead of falling through to Generation", () => {
    assert.equal(
      generationPresentation({
        feature: "retake",
        spec: { params: { prompt: "fix this", duration: 3 } },
      }).label,
      "Retake",
    );
    assert.equal(
      generationPresentation({
        feature: "extend",
        spec: { params: { prompt: "continue", duration: 4 } },
      }).label,
      "Extend",
    );
  });

  it("uses the LoRA recipe title instead of Generation", () => {
    assert.equal(
      generationPresentation({
        feature: "cozy-felt",
        spec: { params: { prompt: "a felt fox" } },
      }).label,
      "Cozy Felt",
    );
  });

  for (const feature of ["retake", "extend"]) {
    it(`shows the size of a succeeded ${feature} output, which its spec never holds`, () => {
      const presentation = generationPresentation({
        feature,
        status: "succeeded",
        spec: { params: { prompt: "p", model: "ltx-2.5-fast", duration: 4 } },
        outputs: [videoOutput(1280, 704)],
      });
      assert.deepEqual(presentation.badges, ["LTX 2.5 Fast", "720p", "4s"]);
    });
  }

  it("shows no size for a retake that has no output yet, not a size the run may change", () => {
    const presentation = generationPresentation({
      feature: "retake",
      status: "running",
      spec: {
        params: {
          prompt: "p",
          model: "ltx-2.5-fast",
          duration: 4,
          resolution: { width: 1280, height: 720 },
        },
      },
      outputs: [],
    });
    assert.deepEqual(presentation.badges, ["LTX 2.5 Fast", "4s"]);
  });

  it("leaves other features alone: a string resolution wins and an output adds no size", () => {
    const withString = generationPresentation({
      feature: "text-to-video",
      status: "succeeded",
      spec: { params: { prompt: "p", resolution: "720p", duration: 4 } },
      outputs: [videoOutput(1920, 1080)],
    });
    const withoutString = generationPresentation({
      feature: "audio-to-video",
      status: "succeeded",
      spec: { params: { prompt: "p", duration: 4 } },
      outputs: [videoOutput(1920, 1080)],
    });
    assert.deepEqual(withString.badges, ["720p", "4s"]);
    assert.deepEqual(withoutString.badges, ["4s"]);
  });

  it("keeps the snapshotted bet line on the queue tooltip and off the badge text", () => {
    const presentation = generationPresentation({
      feature: "text-to-video",
      status: "queued",
      spec: { params: { prompt: "p", resolution: "1080p", duration: 10 } },
      risk: {
        tier: "over",
        message: "Over budget: this needs 14.1 GB but this machine's graphics card has 12.0 GB.",
        readout: { kind: "gpu", need_gb: 14.1, have_gb: 12 },
      },
    });
    assert.equal(presentation.badges.includes("Over budget"), false);
    assert.equal(
      presentation.riskTooltip,
      "Queued despite a warning. Needs 14.1 GB. This GPU has 12 GB.",
    );
    assert.equal(
      generationPresentation({
        feature: "text-to-video",
        status: "queued",
        spec: { params: { prompt: "p" } },
      }).riskTooltip,
      null,
    );
  });
});

describe("generationPresentation input images of a Keyframes run", () => {
  it("lists the stills in time order, not in the order they were added", () => {
    const presentation = generationPresentation({
      id: "g",
      feature: "keyframes",
      status: "queued",
      spec: {
        params: { prompt: "p" },
        inputs: {
          keyframes: [
            { assetId: "late", frameIndex: 96, strength: 1 },
            { assetId: "early", frameIndex: 0, strength: 1 },
          ],
        },
      },
    });

    assert.deepEqual(presentation.inputAssetIds, ["early", "late"]);
    assert.deepEqual(presentation.imageInputAssetIds, ["early", "late"]);
  });
});
