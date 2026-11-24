import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { describe, it } from "node:test";
import { fileURLToPath } from "node:url";

import {
  MOTION_PATH_EASING_OPTIONS,
  MOTION_PATH_MAX_HOLD_SEC,
  MOTION_PATH_MAX_POINTS,
  MOTION_PATH_MAX_TRACKS,
  MOTION_PATH_TRACK_DOT_COLORS,
  motionPathDotColor,
  sampleMotionPathFrames,
  type MotionPath,
} from "./motionPathSampler.ts";

type GoldenCase = {
  name: string;
  duration: number;
  frames: number;
  path: Omit<MotionPath, "points"> & {
    points: { x: number; y: number; handleIn?: { x: number; y: number }; handleOut?: { x: number; y: number } }[];
  };
  positions: [number, number][];
};

const GOLDEN: { step: number; cases: GoldenCase[] } = JSON.parse(
  readFileSync(
    fileURLToPath(new URL("../../../shared/motion-path-golden.json", import.meta.url)),
    "utf8",
  ),
);

const OPENAPI_SCHEMA: {
  components: {
    schemas: {
      IcLoraTrackInputs: { properties: { paths: { maxItems: number } } };
      MotionPath: {
        properties: {
          easing: { enum: string[] };
          points: { maxItems: number };
          startHoldSec: { maximum: number };
          endHoldSec: { maximum: number };
        };
      };
    };
  };
} = JSON.parse(
  readFileSync(
    fileURLToPath(new URL("../../generated/backend-openapi.json", import.meta.url)),
    "utf8",
  ),
);

describe("motion path sampler", () => {
  it("follows the frames of the ltx.io sampler, the same frames the backend draws", () => {
    for (const { name, duration, frames, path, positions } of GOLDEN.cases) {
      const editorPath: MotionPath = {
        ...path,
        points: path.points.map((point, index) => ({ id: `p${index}`, ...point })),
      };
      const sampled = sampleMotionPathFrames(editorPath, duration, frames);
      assert.equal(sampled.length, frames, name);
      // The golden file keeps every step-th frame. The frame count is 8k+1, so the last stays.
      sampled
        .filter((_, index) => index % GOLDEN.step === 0)
        .forEach((frame, index) => {
          assert.ok(Math.abs(frame.x - positions[index][0]) < 1e-9, `${name} frame ${index} x`);
          assert.ok(Math.abs(frame.y - positions[index][1]) < 1e-9, `${name} frame ${index} y`);
        });
    }
  });

  it("pins the point limit, the pause limit, and the easing list to the OpenAPI schema", () => {
    const schema = OPENAPI_SCHEMA.components.schemas.MotionPath.properties;
    assert.equal(MOTION_PATH_MAX_POINTS, schema.points.maxItems);
    assert.equal(MOTION_PATH_MAX_HOLD_SEC, schema.startHoldSec.maximum);
    assert.equal(MOTION_PATH_MAX_HOLD_SEC, schema.endHoldSec.maximum);
    assert.deepEqual([...MOTION_PATH_EASING_OPTIONS], schema.easing.enum);
  });

  it("pins the track limit to the OpenAPI schema", () => {
    const { paths } = OPENAPI_SCHEMA.components.schemas.IcLoraTrackInputs.properties;
    assert.equal(MOTION_PATH_MAX_TRACKS, paths.maxItems);
  });

  it("has a different dot color for each track the form allows", () => {
    const colors = Array.from({ length: MOTION_PATH_MAX_TRACKS }, (_, index) =>
      motionPathDotColor(index),
    );
    assert.equal(new Set(colors).size, MOTION_PATH_MAX_TRACKS);
    assert.equal(motionPathDotColor(MOTION_PATH_TRACK_DOT_COLORS.length), colors[0]);
  });
});
