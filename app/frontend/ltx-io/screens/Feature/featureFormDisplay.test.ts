import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  applyOptionFieldOrder,
  groupConsecutiveOptionFields,
  optionControlValue,
} from "./featureFormDisplay.ts";
import type { AssetRef, FeatureFormField, OptionsFieldDef } from "./types.ts";

type TestSchema = {
  prompt: { kind: "textarea"; value: string };
  note: { kind: "textarea"; value: string };
  startFrame: { kind: "image-asset"; value: AssetRef | null };
  model: { kind: "options"; value: string };
  duration: { kind: "options"; value: number };
  resolution: { kind: "options"; value: string };
};

function textarea(
  id: "prompt" | "note",
  dataKey: "prompt" | "note" = id,
): FeatureFormField<TestSchema> {
  return {
    kind: "textarea",
    id,
    label: id,
    dataKey,
  };
}

function option(
  id: "model" | "duration" | "resolution",
  dataKey: "model" | "duration" | "resolution" = id,
): OptionsFieldDef<TestSchema> {
  return {
    kind: "options",
    id,
    label: id,
    dataKey,
    options: [],
  };
}

function imageAsset(
  id: "startFrame",
  dataKey: "startFrame" = id,
): FeatureFormField<TestSchema> {
  return {
    kind: "image-asset",
    id,
    label: id,
    dataKey,
  };
}

describe("applyOptionFieldOrder", () => {
  it("reorders option slots and keeps textarea positions", () => {
    const fields: FeatureFormField<TestSchema>[] = [
      textarea("prompt"),
      option("duration"),
      option("model"),
      textarea("note"),
      option("resolution"),
    ];

    const displayFields = applyOptionFieldOrder(fields, (options) => [
      options[2]!,
      options[1]!,
      options[0]!,
    ]);

    assert.deepEqual(
      displayFields.map((field) => field.id),
      ["prompt", "resolution", "model", "note", "duration"],
    );
  });

  it("keeps form.fields when the adapter drops an id", () => {
    const fields: FeatureFormField<TestSchema>[] = [
      textarea("prompt"),
      option("duration"),
      option("model"),
    ];

    const displayFields = applyOptionFieldOrder(fields, (options) => [
      options[0]!,
    ]);

    assert.equal(displayFields, fields);
    assert.deepEqual(
      displayFields.map((field) => field.id),
      ["prompt", "duration", "model"],
    );
  });

  it("keeps form.fields when the adapter returns extra ids", () => {
    const fields: FeatureFormField<TestSchema>[] = [
      textarea("prompt"),
      option("duration"),
      option("model"),
    ];
    const extra = option("resolution");

    const displayFields = applyOptionFieldOrder(fields, (options) => [
      ...options,
      extra,
    ]);

    assert.equal(displayFields, fields);
  });

  it("keeps form.fields when the adapter returns an unknown id", () => {
    const fields: FeatureFormField<TestSchema>[] = [
      textarea("prompt"),
      option("duration"),
      option("model"),
    ];
    const extra = option("resolution");

    const displayFields = applyOptionFieldOrder(fields, (options) => [
      extra,
      options[1]!,
    ]);

    assert.equal(displayFields, fields);
    assert.deepEqual(
      displayFields.map((field) => field.id),
      ["prompt", "duration", "model"],
    );
  });

  it("keeps image-asset slots in place while reordering options", () => {
    const fields: FeatureFormField<TestSchema>[] = [
      imageAsset("startFrame"),
      textarea("prompt"),
      option("duration"),
      option("model"),
    ];

    const displayFields = applyOptionFieldOrder(fields, (options) => [
      options[1]!,
      options[0]!,
    ]);

    assert.deepEqual(
      displayFields.map((field) => field.id),
      ["startFrame", "prompt", "model", "duration"],
    );
  });
});

describe("groupConsecutiveOptionFields", () => {
  it("wraps consecutive options into one group and splits around textareas", () => {
    const groups = groupConsecutiveOptionFields<TestSchema>([
      textarea("prompt"),
      option("model"),
      option("duration"),
      textarea("note"),
      option("resolution"),
    ]);

    assert.deepEqual(
      groups.map((group) =>
        group.kind === "options"
          ? { kind: group.kind, ids: group.fields.map((field) => field.id) }
          : { kind: group.kind, id: group.field.id },
      ),
      [
        { kind: "textarea", id: "prompt" },
        { kind: "options", ids: ["model", "duration"] },
        { kind: "textarea", id: "note" },
        { kind: "options", ids: ["resolution"] },
      ],
    );
  });

  it("keeps image-asset fields as their own groups and still clusters options", () => {
    const groups = groupConsecutiveOptionFields<TestSchema>([
      imageAsset("startFrame"),
      textarea("prompt"),
      option("model"),
      option("duration"),
    ]);

    assert.deepEqual(
      groups.map((group) =>
        group.kind === "options"
          ? { kind: group.kind, ids: group.fields.map((field) => field.id) }
          : { kind: group.kind, id: group.field.id },
      ),
      [
        { kind: "image-asset", id: "startFrame" },
        { kind: "textarea", id: "prompt" },
        { kind: "options", ids: ["model", "duration"] },
      ],
    );
  });
});

describe("optionControlValue", () => {
  const options = [{ value: 24 }, { value: 25 }];

  it("shows the stored value", () => {
    assert.equal(optionControlValue(24, options, 25), 24);
  });

  it("shows the default value while the stored value is null", () => {
    assert.equal(optionControlValue(null, options, 25), 25);
  });

  it("shows the first option when there is no default value", () => {
    assert.equal(optionControlValue(null, options), 24);
  });

  it("throws when there is nothing to show", () => {
    assert.throws(() => optionControlValue(null, []), /no choices/);
  });
});
