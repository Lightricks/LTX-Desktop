import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  applyFeatureFieldChange,
  createFeatureFieldChange,
  type FeatureDefinition,
  type FeatureFieldChange,
  type FeatureFormModel,
  type FeatureValues,
} from "./types.ts";
import type { EmptyStateAdapter } from "./results/ResultsFeed.tsx";

type SyntheticSchema = {
  prompt: { kind: "textarea" };
  tag: { kind: "options"; value: string };
  style: { kind: "options"; value: "plain" | "vivid" };
  duration: { kind: "options"; value: number };
};

type SyntheticValues = FeatureValues<SyntheticSchema>;

type SyntheticBody = {
  text: string;
  tag: string;
  style: SyntheticValues["style"];
};

const defaults: SyntheticValues = {
  prompt: "",
  tag: "draft",
  style: "plain",
  duration: 4,
};

const SyntheticEmptyState: EmptyStateAdapter = () => null;
void SyntheticEmptyState;

const syntheticDefinition: FeatureDefinition<
  SyntheticSchema,
  SyntheticBody,
  undefined
> = {
  id: "text-to-video",
  title: "Synthetic",
  defaults,
  form: (): FeatureFormModel<SyntheticSchema> => ({
    fields: [
      {
        kind: "textarea",
        id: "prompt",
        label: "Prompt",
        dataKey: "prompt",
        placeholder: "Describe something",
      },
      {
        kind: "options",
        id: "tag",
        label: "Tag",
        dataKey: "tag",
        options: [
          { value: "draft", label: "Draft" },
          { value: "any-other-tag", label: "Other" },
        ],
      },
      {
        kind: "options",
        id: "style",
        label: "Style",
        dataKey: "style",
        options: [
          { value: "plain", label: "Plain" },
          { value: "vivid", label: "Vivid" },
        ],
      },
      {
        kind: "options",
        id: "duration",
        label: "Duration",
        dataKey: "duration",
        options: [
          { value: 4, label: "4s" },
          { value: 8, label: "8s" },
        ],
      },
    ],
  }),
  applyChange: (
    values: SyntheticValues,
    change: FeatureFieldChange<SyntheticSchema>,
  ): SyntheticValues => applyFeatureFieldChange(values, change),
  normalize: (values) => values,
  validate: (values) =>
    values.prompt.trim()
      ? []
      : [
          {
            id: "prompt-required",
            fieldId: "prompt",
            message: "Prompt is required.",
          },
        ],
  fromGeneration: () => null,
  toCreateBody: (values) => ({
    text: values.prompt,
    tag: values.tag,
    style: values.style,
  }),
};

describe("FeatureDefinition media-agnostic contract", () => {
  it("builds typed textarea and options fields without video values", () => {
    const form = syntheticDefinition.form(defaults, undefined);

    assert.deepEqual(
      form.fields.map(({ kind, id, dataKey }) => ({ kind, id, dataKey })),
      [
        { kind: "textarea", id: "prompt", dataKey: "prompt" },
        { kind: "options", id: "tag", dataKey: "tag" },
        { kind: "options", id: "style", dataKey: "style" },
        { kind: "options", id: "duration", dataKey: "duration" },
      ],
    );
  });

  it("applies a generic field change without video context", () => {
    const next = syntheticDefinition.applyChange(
      defaults,
      {
        kind: "options",
        fieldId: "style",
        dataKey: "style",
        value: "vivid",
      },
      undefined,
    );

    assert.deepEqual(next, {
      prompt: "",
      tag: "draft",
      style: "vivid",
      duration: 4,
    });
  });

  it("applies a first-class textarea field change", () => {
    const promptChange: FeatureFieldChange<SyntheticSchema> = {
      kind: "textarea",
      fieldId: "prompt",
      dataKey: "prompt",
      value: "A dog running on a beach",
    };

    assert.deepEqual(
      syntheticDefinition.applyChange(defaults, promptChange, undefined),
      {
        prompt: "A dog running on a beach",
        tag: "draft",
        style: "plain",
        duration: 4,
      },
    );
  });

  it("applies a broad-string option change built by the typed helper", () => {
    const tagChange = createFeatureFieldChange<SyntheticSchema, "tag">({
      kind: "options",
      fieldId: "tag",
      dataKey: "tag",
      value: "any-other-tag",
    });

    assert.deepEqual(
      syntheticDefinition.applyChange(defaults, tagChange, undefined),
      {
        prompt: "",
        tag: "any-other-tag",
        style: "plain",
        duration: 4,
      },
    );
  });

  it("preserves the value type for a numeric field change", () => {
    const validChange: FeatureFieldChange<SyntheticSchema> = {
      kind: "options",
      fieldId: "duration",
      dataKey: "duration",
      value: 8,
    };

    assert.deepEqual(
      syntheticDefinition.applyChange(defaults, validChange, undefined),
      { prompt: "", tag: "draft", style: "plain", duration: 8 },
    );
  });

  it("returns validation issues scoped to a form field", () => {
    assert.deepEqual(syntheticDefinition.validate(defaults, undefined), [
      {
        id: "prompt-required",
        fieldId: "prompt",
        message: "Prompt is required.",
      },
    ]);
  });
});
