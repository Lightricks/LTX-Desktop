import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { getIcLoraTrackRecipe } from "../../../../lib/ic-lora-recipes.ts";
import { MOTION_PATH_DEFAULT_EASING, type MotionPath } from "../../../lib/motionPathSampler.ts";
import { createFeatureFieldChange, type AssetRef } from "../types.ts";
import {
  createIcLoraTrackDefinition,
  readTrack,
  type IcLoraTrackSchema,
} from "./icLoraTrack.ts";
import { specFixture } from "./specFixture.ts";

const definition = createIcLoraTrackDefinition(getIcLoraTrackRecipe("motion-track"));
const context = { specs: specFixture() };

const path: MotionPath = {
  points: [
    { id: "a", x: 0.1, y: 0.2, handleOut: { x: 0.05, y: 0 } },
    { id: "b", x: 0.8, y: 0.9 },
  ],
  easing: MOTION_PATH_DEFAULT_EASING,
  startHoldSec: 0,
  endHoldSec: 0,
};

function drawnOn(startFrame: AssetRef) {
  return { ...definition.defaults, startFrame, paths: [path] };
}

function changeStartFrame(startFrame: AssetRef) {
  return createFeatureFieldChange<IcLoraTrackSchema, "startFrame">({
    kind: "image-asset",
    fieldId: "path",
    dataKey: "startFrame",
    value: startFrame,
  });
}

function changePath(next: MotionPath[]) {
  return createFeatureFieldChange<IcLoraTrackSchema, "paths">({
    kind: "motion-path",
    fieldId: "path",
    dataKey: "paths",
    value: next,
  });
}

describe("motion track definition", () => {
  it("clears the path when the start frame changes, not when only the path changes", () => {
    const drawn = drawnOn({ assetId: "frog" });

    const newImage = definition.applyChange(drawn, changeStartFrame({ assetId: "cat" }), context);
    assert.deepEqual(newImage.startFrame, { assetId: "cat" });
    assert.deepEqual(newImage.paths, []);

    const edited = { ...path, points: path.points.slice(0, 1) };
    const sameImage = definition.applyChange(drawn, changePath([edited]), context);
    assert.deepEqual(sameImage.paths, [edited]);
  });

  it("clamps the path pauses when the duration is shortened below their sum", () => {
    const longPauses = {
      ...drawnOn({ assetId: "frog" }),
      duration: 6 as const,
      paths: [{ ...path, startHoldSec: 2, endHoldSec: 2 }],
    };
    const shortened = definition.applyChange(
      longPauses,
      createFeatureFieldChange<IcLoraTrackSchema, "duration">({
        kind: "options",
        fieldId: "duration",
        dataKey: "duration",
        value: 3,
      }),
      context,
    );
    const [clamped] = shortened.paths;
    assert.ok(clamped != null);
    assert.ok(clamped.startHoldSec + clamped.endHoldSec < shortened.duration);
    assert.equal(clamped.startHoldSec, 2);
    assert.equal(clamped.endHoldSec, 0.5);
  });

  it("sends no aspect ratio and no editor point id", () => {
    const body = definition.toCreateBody(drawnOn({ assetId: "frog" }), context);
    assert.equal("aspectRatio" in body.params, false);
    assert.deepEqual(body.inputs.paths[0]?.points, [
      { x: 0.1, y: 0.2, handleOut: { x: 0.05, y: 0 } },
      { x: 0.8, y: 0.9 },
    ]);
  });

  it("blocks a path of fewer than two points and a blank prompt, and keeps Generate off", () => {
    const ready = definition.normalize(drawnOn({ assetId: "frog" }), context);
    assert.equal(definition.isReady?.(ready, context), true);
    assert.equal(definition.isReady?.({ ...ready, prompt: "   " }, context), false);
    assert.equal(definition.isReady?.({ ...ready, startFrame: null }, context), false);
    assert.equal(definition.isReady?.({ ...ready, paths: [] }, context), false);

    const onePoint = { ...ready, paths: [{ ...path, points: path.points.slice(0, 1) }] };
    assert.deepEqual(
      definition.validate(onePoint, context).map((issue) => issue.id),
      ["path-required"],
    );
    assert.deepEqual(
      definition.validate({ ...ready, prompt: "  " }, context).map((issue) => issue.id),
      ["prompt-required"],
    );
    assert.throws(() => definition.toCreateBody(onePoint, context));
  });

  it("reopens a saved 2160p run at 1080p", () => {
    const normalized = definition.normalize(
      { ...drawnOn({ assetId: "frog" }), resolution: "2160p" },
      context,
    );
    assert.equal(normalized.resolution, "1080p");
  });

  it("reuses a saved path whose pause is not one the editor lists and rejects invalid paths", () => {
    const base = {
      points: [
        { x: 0.1, y: 0.2 },
        { x: 0.8, y: 0.9 },
      ],
      easing: "linear",
      startHoldSec: 0.25,
      endHoldSec: 2.75,
    };
    const track = readTrack({ inputs: { image: { assetId: "frog" }, paths: [base] } });
    assert.equal(track?.paths[0]?.startHoldSec, 0.25);
    assert.equal(track?.paths[0]?.endHoldSec, 2.75);

    const readWith = (change: object) =>
      readTrack({ inputs: { image: { assetId: "frog" }, paths: [{ ...base, ...change }] } });
    assert.equal(readWith({ easing: "wobble" }), null);
    assert.equal(readWith({ startHoldSec: 3.5 }), null);
    assert.equal(readWith({ points: base.points.slice(0, 1) }), null);
  });

  it("sends every track in inputs.paths", () => {
    const second: MotionPath = {
      ...path,
      points: [
        { id: "c", x: 0.3, y: 0.7 },
        { id: "d", x: 0.6, y: 0.1 },
      ],
      easing: "linear",
    };
    const body = definition.toCreateBody(
      { ...drawnOn({ assetId: "frog" }), paths: [path, second] },
      context,
    );
    assert.equal(body.inputs.paths.length, 2);
    assert.deepEqual(body.inputs.paths[1], {
      points: [
        { x: 0.3, y: 0.7 },
        { x: 0.6, y: 0.1 },
      ],
      easing: "linear",
      startHoldSec: 0,
      endHoldSec: 0,
    });
  });

  it("blocks a track of one point even when another track is drawn", () => {
    const ready = definition.normalize(drawnOn({ assetId: "frog" }), context);
    const partial = { ...ready, paths: [path, { ...path, points: path.points.slice(0, 1) }] };
    assert.equal(definition.isReady?.(partial, context), false);
    assert.throws(() => definition.toCreateBody(partial, context));
  });

  it("reuses every path of a saved spec and rejects a spec with no valid paths", () => {
    const wire = {
      points: [
        { x: 0.1, y: 0.2 },
        { x: 0.8, y: 0.9 },
      ],
      easing: "linear",
      startHoldSec: 0,
      endHoldSec: 0,
    };
    const image = { assetId: "frog" };
    const readPaths = (paths: unknown) => readTrack({ inputs: { image, paths } });
    assert.equal(readPaths([wire])?.paths.length, 1);
    assert.equal(readPaths([wire, wire, wire])?.paths.length, 3);
    assert.equal(readPaths([]), null);
    assert.equal(readPaths([wire, wire, wire, wire]), null);
    assert.equal(readPaths([wire, { ...wire, easing: "wobble" }]), null);
    assert.equal(readTrack({ inputs: { image, path: wire } }), null);
  });
});
