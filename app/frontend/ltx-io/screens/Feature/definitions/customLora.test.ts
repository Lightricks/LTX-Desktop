import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  createCustomLoraDefinition,
  generationRanCustomLora,
  recentCustomLoraIds,
  type CustomLoraContext,
  type CustomLoraItem,
} from "./customLora.ts";
import { specFixtureBothOfferings } from "./specFixture.ts";

const definition = createCustomLoraDefinition();
const specs = specFixtureBothOfferings("LTX 2.5 Fast");

const saved = {
  size_bytes: 1,
  downloaded: true,
  can_download: false,
  page_url: null,
} as const;
const lora: CustomLoraItem = {
  ...saved,
  id: "ink-wash",
  name: "Ink Wash",
  source: "file",
  trigger: null,
  description: "An ink wash style.",
  examples: [],
};
const context: CustomLoraContext = { specs, lora };

const START_FRAME = { assetId: "asset-1" };

describe("createCustomLoraDefinition", () => {
  it("drops auto aspect ratio without a Start Frame and keeps it with one", () => {
    const values = { ...definition.defaults, prompt: "p", aspectRatio: "auto" } as const;

    const without = definition.normalize?.(values, context);
    const withFrame = definition.normalize?.({ ...values, startFrame: START_FRAME }, context);

    assert.ok(without);
    assert.notEqual(without.aspectRatio, "auto");
    assert.equal(withFrame?.aspectRatio, "auto");
  });

  it("sends the LoRA id and strength, never a file path, and the Start Frame only when one is set", () => {
    const values = { ...definition.defaults, prompt: " a fox ", strength: 1.5 };

    const textBody = definition.toCreateBody(values, context);
    const imageBody = definition.toCreateBody({ ...values, startFrame: START_FRAME }, context);

    assert.equal(textBody.params.customLoraId, "ink-wash");
    assert.equal(textBody.params.scale, 1.5);
    assert.equal(textBody.params.prompt, "a fox");
    assert.ok(!("ref" in textBody.params));
    assert.ok(!("loras" in textBody.params));
    assert.equal(textBody.inputs, undefined);
    assert.deepEqual(imageBody.inputs, { startFrame: START_FRAME });
  });

  it("restores the strength from a desktop run and from a remote run", () => {
    const common = { prompt: "stored", model: "ltx-2.5-fast", resolution: "720p", duration: 8, fps: 24 };
    const desktop = {
      params: { ...common, loras: [{ ref: "loras/custom/ink-wash/ink-wash.safetensors", scale: 2.25, customLoraId: "ink-wash" }] },
    };
    // The phone gets no loras array. The backend projects the strength.
    const remote = { params: { ...common, customLoraId: "walk-cycle", scale: 0.5 }, inputs: { startFrame: START_FRAME } };

    const fromDesktop = definition.fromGeneration?.(desktop, context);
    const fromRemote = definition.fromGeneration?.(remote, context);

    assert.equal(fromDesktop?.strength, 2.25);
    assert.equal(fromRemote?.strength, 0.5);
    assert.deepEqual(fromRemote?.startFrame, START_FRAME);
  });
});

describe("generationRanCustomLora", () => {
  const desktopRun = { params: { loras: [{ ref: "r", scale: 1, customLoraId: "ink-wash" }] } };
  const remoteRun = { params: { customLoraId: "walk-cycle", scale: 1 } };

  it("keeps only the runs of the open LoRA, for desktop and remote runs", () => {
    assert.equal(generationRanCustomLora(desktopRun, "ink-wash"), true);
    assert.equal(generationRanCustomLora(desktopRun, "walk-cycle"), false);
    assert.equal(generationRanCustomLora(remoteRun, "walk-cycle"), true);
    assert.equal(generationRanCustomLora(remoteRun, "ink-wash"), false);
  });

  it("does not match a run that has no saved LoRA", () => {
    assert.equal(generationRanCustomLora({ params: { prompt: "p" } }, "ink-wash"), false);
    assert.equal(generationRanCustomLora(null, "ink-wash"), false);
  });
});

describe("recentCustomLoraIds", () => {
  it("lists each LoRA once, in the order of its newest run", () => {
    const run = (customLoraId: string) => ({ spec: { params: { customLoraId } } });

    const ids = recentCustomLoraIds([run("b"), run("a"), run("b"), run("c"), run("a")]);

    assert.deepEqual(ids, ["b", "a", "c"]);
  });
});
