import assert from "node:assert/strict";
import { describe, it } from "node:test";

import type { FieldOption, OptionsFieldDef } from "../types.ts";
import {
  DISCRETE_OPTION_SEGMENT_THRESHOLD,
  getOptionControlPresentation,
  orderVideoSettingsOptionFields,
  shouldUseSegmentControl,
  sortSettingFields,
} from "./optionControlPresentation.ts";

function options(count: number): FieldOption[] {
  return Array.from({ length: count }, (_, index) => ({
    value: index,
    label: `Option ${index + 1}`,
  }));
}

type TestSchema = {
  model: { kind: "options"; value: string };
  duration: { kind: "options"; value: number };
  aspectRatio: { kind: "options"; value: string };
  resolution: { kind: "options"; value: string };
  fps: { kind: "options"; value: number };
};

function field(
  overrides: Partial<OptionsFieldDef<TestSchema>> = {},
): OptionsFieldDef<TestSchema> {
  return {
    kind: "options",
    id: "duration",
    label: "Duration",
    dataKey: "duration",
    options: options(2),
    ...overrides,
  };
}

describe("shouldUseSegmentControl", () => {
  it("returns true for 2–3 options", () => {
    assert.equal(shouldUseSegmentControl(options(2)), true);
    assert.equal(shouldUseSegmentControl(options(3)), true);
  });

  it("returns false for 4+ options", () => {
    assert.equal(shouldUseSegmentControl(options(4)), false);
  });

  it("returns false when forcePicker is true", () => {
    assert.equal(
      shouldUseSegmentControl(options(2), { forcePicker: true }),
      false,
    );
  });

  it("uses a threshold of 3", () => {
    assert.equal(DISCRETE_OPTION_SEGMENT_THRESHOLD, 3);
  });
});

describe("getOptionControlPresentation", () => {
  it("returns segment for 2 options", () => {
    assert.equal(getOptionControlPresentation(options(2)), "segment");
  });

  it("returns picker for 4 options", () => {
    assert.equal(getOptionControlPresentation(options(4)), "picker");
  });

  it("returns picker when forcePicker is true with 2 options", () => {
    assert.equal(
      getOptionControlPresentation(options(2), { forcePicker: true }),
      "picker",
    );
  });

  it("returns picker when maxOptionCount is above the segment threshold", () => {
    assert.equal(
      getOptionControlPresentation(options(2), { maxOptionCount: 5 }),
      "picker",
    );
  });

  it("returns display-only for a single option", () => {
    assert.equal(getOptionControlPresentation(options(1)), "display-only");
  });
});

describe("sortSettingFields", () => {
  it("orders pickers before segments", () => {
    const duration = field({
      id: "duration",
      dataKey: "duration",
      forcePicker: true,
      options: options(4),
    });
    const aspectRatio = field({
      id: "aspectRatio",
      dataKey: "aspectRatio",
      label: "Aspect Ratio",
      options: options(2),
    });
    const resolution = field({
      id: "resolution",
      dataKey: "resolution",
      label: "Resolution",
      options: options(2),
      maxOptionCount: 5,
    });

    assert.deepEqual(
      sortSettingFields([aspectRatio, resolution, duration]).map(
        (item) => item.dataKey,
      ),
      ["resolution", "duration", "aspectRatio"],
    );
  });

  it("does not special-case field names when ordering controls", () => {
    const model = field({
      id: "model",
      dataKey: "model",
      label: "Model",
      options: options(1),
    });
    const resolution = field({
      id: "resolution",
      dataKey: "resolution",
      label: "Resolution",
      options: options(3),
      maxOptionCount: 5,
    });
    const aspectRatio = field({
      id: "aspectRatio",
      dataKey: "aspectRatio",
      label: "Aspect Ratio",
      options: options(2),
    });

    assert.deepEqual(
      sortSettingFields([resolution, model, aspectRatio]).map(
        (item) => item.dataKey,
      ),
      ["resolution", "model", "aspectRatio"],
    );
  });
});

describe("orderVideoSettingsOptionFields", () => {
  it("pins model first even when it would otherwise rank as display-only", () => {
    const shuffled = [
      field({ id: "aspectRatio", dataKey: "aspectRatio", options: options(2) }),
      field({
        id: "resolution",
        dataKey: "resolution",
        options: options(2),
        maxOptionCount: 5,
      }),
      field({
        id: "duration",
        dataKey: "duration",
        forcePicker: true,
        options: options(4),
      }),
      field({ id: "fps", dataKey: "fps", options: options(2), maxOptionCount: 4 }),
      field({ id: "model", dataKey: "model", options: options(1) }),
    ];

    assert.deepEqual(
      orderVideoSettingsOptionFields(shuffled).map((item) => item.dataKey),
      ["model", "resolution", "duration", "fps", "aspectRatio"],
    );
  });
});
