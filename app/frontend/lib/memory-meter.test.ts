import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { specFixture } from "../ltx-io/screens/Feature/definitions/specFixture.ts";
import { buildMemoryMeterModel } from "./memory-meter.ts";
import type { FastCellAsk, VideoGenerationModelSpecsResponse } from "./video-generation-model-specs.ts";

type Ask = Pick<FastCellAsk, "resolution" | "duration" | "ask_gib"> & Partial<FastCellAsk>;

function specsWithAsks(
  asks: Ask[],
  memory: { device: "gpu" | "mac"; capacity_gib: number },
): VideoGenerationModelSpecsResponse {
  const specs = structuredClone(specFixture());
  const cellAsks: FastCellAsk[] = asks.map((ask) => ({
    fps: 24,
    audio: false,
    confidence: "solid",
    windows: 1,
    tight: false,
    ...ask,
  }));
  for (const item of [...specs.local_models, ...specs.downloaded_local_models]) {
    item.spec.cell_asks = cellAsks;
  }
  return { ...specs, memory };
}

function selection(resolution: string, duration: number) {
  return { model: "ltx-2.5-fast", resolution, fps: 24, duration, aspectRatio: "16:9" };
}

const FIXES = { fixFields: ["duration", "resolution"] as const };

describe("buildMemoryMeterModel", () => {
  it("shows the parts line, not the tight line, on a job the backend split into parts", () => {
    const specs = specsWithAsks(
      [{ resolution: "1080p", duration: 10, ask_gib: 15.8, windows: 3, tight: true }],
      { device: "gpu", capacity_gib: 16 },
    );
    const meter = buildMemoryMeterModel(specs, selection("1080p", 10));
    assert.match(meter?.line?.text ?? "", /^Made in 3 parts/);
  });

  it("shortens only to a length the picker offers and that fits", () => {
    // 1080p offers 2, 3, 4, 5 and 10 s. 8 s fits but is not offered; 5 s is offered but over.
    const specs = specsWithAsks(
      [
        { resolution: "1080p", duration: 10, ask_gib: 40 },
        { resolution: "1080p", duration: 8, ask_gib: 14 },
        { resolution: "1080p", duration: 5, ask_gib: 20 },
        { resolution: "1080p", duration: 4, ask_gib: 15 },
        { resolution: "1080p", duration: 3, ask_gib: 12 },
      ],
      { device: "gpu", capacity_gib: 16 },
    );
    const meter = buildMemoryMeterModel(specs, selection("1080p", 10), FIXES);
    const shorten = meter?.fixes.find((fix) => fix.dataKey === "duration");
    assert.deepEqual(shorten, { label: "Shorten to 4 s", dataKey: "duration", value: 4 });
  });

  it("offers the sharpest lower resolution that fits, never a higher or unoffered one", () => {
    // 1440p is offered and fits but is higher; 720p fits but is not offered; 540p fits.
    const specs = specsWithAsks(
      [
        { resolution: "1080p", duration: 10, ask_gib: 40 },
        { resolution: "1440p", duration: 10, ask_gib: 10 },
        { resolution: "720p", duration: 10, ask_gib: 12 },
        { resolution: "540p", duration: 10, ask_gib: 12 },
      ],
      { device: "gpu", capacity_gib: 16 },
    );
    for (const item of [...specs.local_models, ...specs.downloaded_local_models]) {
      const grid = item.spec.supported_resolutions_durations;
      grid["1440p"] = structuredClone(grid["1080p"]);
      delete grid["720p"];
    }
    const meter = buildMemoryMeterModel(specs, selection("1080p", 10), FIXES);
    const lower = meter?.fixes.find((fix) => fix.dataKey === "resolution");
    assert.equal(lower?.value, "540p");
  });

  it("puts the rough scale's boundaries at exactly 50% and 85% of capacity", () => {
    const cases: [number, number, string][] = [
      [2, 9.99, "Light"],
      [3, 10, "Medium"],
      [4, 16.99, "Medium"],
      [5, 17, "Heavy"],
    ];
    const specs = specsWithAsks(
      cases.map(([duration, ask_gib]) => ({
        resolution: "1080p",
        duration,
        ask_gib,
        confidence: "rough",
      })),
      { device: "gpu", capacity_gib: 20 },
    );
    for (const [duration, , label] of cases) {
      const meter = buildMemoryMeterModel(specs, selection("1080p", duration));
      assert.equal(meter?.valueText, `${label} for this GPU`, `${duration} s`);
    }
  });

  it("does not tell a Mac user to close other apps when the job is tight", () => {
    const specs = specsWithAsks(
      [{ resolution: "1080p", duration: 10, ask_gib: 24.7, tight: true }],
      { device: "mac", capacity_gib: 25 },
    );
    const meter = buildMemoryMeterModel(specs, selection("1080p", 10));
    assert.ok(meter?.line != null);
    assert.doesNotMatch(meter.line.text, /close other apps/i);
  });
});

describe("buildMemoryMeterModel with a forced shape", () => {
  // A 16 GB card runs 1080p 10 s in 3 parts at 16 GB; one pass asks 22.7. 5 s fits in one pass.
  const onePassOver = { ask_gib: 22.7, confidence: "solid", windows: 1, tight: false } as const;
  const specs = specsWithAsks(
    [
      { resolution: "1080p", duration: 10, ask_gib: 16, windows: 3, alternative: onePassOver },
      { resolution: "1080p", duration: 5, ask_gib: 12.9 },
    ],
    { device: "gpu", capacity_gib: 16 },
  );
  const forced = { ...selection("1080p", 10), passShape: "one_pass" as const };

  it("the meter model reads over for a forced one pass that does not fit", () => {
    assert.equal(buildMemoryMeterModel(specs, selection("1080p", 10))?.over, false);
    const meter = buildMemoryMeterModel(specs, forced);
    assert.equal(meter?.over, true);
    assert.equal(meter?.askGib, 22.7);
    assert.equal(meter?.parts, 1);
  });

  it("fixes for an over forced one pass put Make it in parts first", () => {
    const meter = buildMemoryMeterModel(specs, forced, {
      fixFields: ["passShape", "duration", "resolution"],
    });
    assert.deepEqual(meter?.fixes.slice(0, 2), [
      { label: "Make it in parts", dataKey: "passShape", value: "parts" },
      { label: "Shorten to 5 s", dataKey: "duration", value: 5 },
    ]);
  });

  it("reads over for a forced alternative shape carrying a RAM-shortfall risk", () => {
    const ramShortSpecs = specsWithAsks(
      [
        {
          resolution: "1080p",
          duration: 10,
          ask_gib: 18,
          windows: 1,
          alternative: {
            ask_gib: 14,
            confidence: "solid",
            windows: 2,
            tight: false,
            risk: {
              tier: "over",
              message: "Needs too much memory: this needs 40 GB of system RAM but this machine has 32 GB.",
              readout: { kind: "ram", need_gb: 40, have_gb: 32 },
            },
          },
        },
      ],
      { device: "gpu", capacity_gib: 24 },
    );
    const meter = buildMemoryMeterModel(ramShortSpecs, {
      ...selection("1080p", 10),
      passShape: "parts",
    });
    assert.equal(meter?.over, true);
    assert.equal(
      meter?.line?.text,
      "Needs too much memory: this needs 40 GB of system RAM but this machine has 32 GB.",
    );
  });
});
