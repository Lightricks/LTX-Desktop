import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { generationPresentation } from "./generationPresentation.ts";

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
});
