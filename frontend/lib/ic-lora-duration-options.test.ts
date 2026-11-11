import assert from "node:assert/strict";
import { describe, it } from "node:test";

import type { components } from "../generated/backend-openapi.ts";

import { mergeDurationControlOptions } from "./ic-lora-duration-options.ts";

type CatalogControl = components["schemas"]["IcLoraControl"];
type PreflightCell = components["schemas"]["IcLoraPreflightCell"];

function control(overrides: Partial<CatalogControl> = {}): CatalogControl {
  return {
    id: "duration",
    label: "Duration",
    kind: "int",
    default: 5,
    options: [5, 6],
    ...overrides,
  };
}

function dropped(value: number, message: string) {
  return { value, reason: "stream_over" as const, message };
}

function cell(duration: number, ok: boolean, message?: string): PreflightCell {
  return {
    duration,
    ok,
    message: message ?? null,
    code: ok ? null : "VIDEO_JOB_TOO_LARGE",
  };
}

const labelFor = (option: number | string) => `${option}s`;

describe("mergeDurationControlOptions", () => {
  it("sorts kept and advisory drops numerically without disabling", () => {
    const merged = mergeDurationControlOptions(
      control({
        options: [5, 6],
        dropped_options: [dropped(20, "too large"), dropped(8, "too large")],
      }),
      labelFor,
      null,
    );
    assert.equal(merged.emptyMessage, null);
    assert.deepEqual(
      merged.options.map((option) => option.value),
      ["5", "6", "8", "20"],
    );
    for (const option of merged.options) {
      assert.equal(option.disabled, undefined);
    }
    assert.equal(
      merged.options.find((option) => option.value === "8")?.tooltip,
      "too large",
    );
    assert.equal(
      merged.options.find((option) => option.value === "5")?.tooltip,
      undefined,
    );
  });

  it("lets fresh cells disable failures and clear stale advisories", () => {
    const merged = mergeDurationControlOptions(
      control({
        options: [5],
        dropped_options: [dropped(8, "preview says too large")],
      }),
      labelFor,
      [
        cell(5, true),
        cell(8, false, "still too large on these pixels"),
        cell(20, false, "too large"),
      ],
    );
    assert.equal(merged.emptyMessage, null);
    const byValue = new Map(merged.options.map((option) => [option.value, option]));
    // Passing on real pixels clears the advisory tooltip.
    assert.equal(byValue.get("5")?.disabled, undefined);
    assert.equal(byValue.get("5")?.tooltip, undefined);
    // Failing on real pixels hard-disables with the authoritative refusal.
    assert.equal(byValue.get("8")?.disabled, true);
    assert.equal(byValue.get("8")?.tooltip, "still too large on these pixels");
  });

  it("disables an over-budget preflight duration instead of warning it", () => {
    const merged = mergeDurationControlOptions(
      control({ options: [5], dropped_options: [dropped(8, "preview")] }),
      labelFor,
      [cell(5, true), { ...cell(8, false, "over budget"), tier: "over" }],
    );
    const over = merged.options.find((option) => option.value === "8");
    assert.equal(over?.disabled, true);
    assert.equal(over?.warning, undefined);
  });

  it("does not restore an all-dropped duration list", () => {
    const merged = mergeDurationControlOptions(
      control({
        options: [],
        dropped_options: [{ ...dropped(8, "over budget"), tier: "over" }],
      }),
      labelFor,
      null,
    );
    assert.deepEqual(merged.options, []);
    assert.equal(merged.emptyMessage, "over budget");
  });

  it("names the backend reason when no option survived", () => {
    const merged = mergeDurationControlOptions(
      control({
        options: [],
        dropped_options: [dropped(5, "This machine cannot run local generation.")],
      }),
      labelFor,
      null,
    );
    assert.deepEqual(merged.options, []);
    assert.equal(merged.emptyMessage, "This machine cannot run local generation.");
  });

  it("leaves unevaluated durations as listed", () => {
    const merged = mergeDurationControlOptions(control({ options: [5, 6] }), labelFor, [
      cell(5, false, "too large"),
    ]);
    const byValue = new Map(merged.options.map((option) => [option.value, option]));
    assert.equal(byValue.get("5")?.disabled, true);
    assert.equal(byValue.get("6")?.disabled, undefined);
    assert.equal(byValue.get("6")?.tooltip, undefined);
  });
});
