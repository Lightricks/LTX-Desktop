import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { resolveFetcherLanding } from "./fetcher-landing.ts";
import { HOME_FEATURES } from "./home-features.ts";
import type { VideoGenerationOfferingCapabilities } from "./video-generation-model-specs.ts";

function homeFeaturePath(slug: string): string {
  const feature = HOME_FEATURES.find(
    (entry) => "fetcherTool" in entry && entry.fetcherTool === slug,
  );
  assert.ok(feature, `missing Home feature for ${slug}`);
  return feature.path;
}

const local25: VideoGenerationOfferingCapabilities = {
  t2v: true,
  i2v: true,
  a2v: true,
  ic_lora: true,
  retake: false,
  extend: false,
  multi_keyframe: true,
  multi_keyframe_max_count: 4,
  user_loras: true,
  camera_motion: true,
  auto_duration: true,
};

describe("resolveFetcherLanding", () => {
  it("blocks unknown slugs without waiting on support", () => {
    assert.deepEqual(
      resolveFetcherLanding({
        slug: "nope",
        isLoading: true,
        errorMessage: null,
        capabilities: null,
        features: HOME_FEATURES,
      }),
      { status: "blocked" },
    );
    assert.deepEqual(
      resolveFetcherLanding({
        slug: "assets",
        isLoading: true,
        errorMessage: null,
        capabilities: null,
        features: HOME_FEATURES,
      }),
      { status: "blocked" },
    );
  });

  it("stays resolving while the local offering is loading", () => {
    assert.deepEqual(
      resolveFetcherLanding({
        slug: "t2v",
        isLoading: true,
        errorMessage: null,
        capabilities: null,
        features: HOME_FEATURES,
      }),
      { status: "resolving" },
    );
  });

  it("blocks when the offering cannot be loaded", () => {
    assert.deepEqual(
      resolveFetcherLanding({
        slug: "t2v",
        isLoading: false,
        errorMessage: "offline",
        capabilities: null,
        features: HOME_FEATURES,
      }),
      { status: "blocked" },
    );
    assert.deepEqual(
      resolveFetcherLanding({
        slug: "t2v",
        isLoading: false,
        errorMessage: null,
        capabilities: null,
        features: HOME_FEATURES,
      }),
      { status: "blocked" },
    );
  });

  it("blocks when the local offering says the tool is unsupported", () => {
    assert.deepEqual(
      resolveFetcherLanding({
        slug: "t2v",
        isLoading: false,
        errorMessage: null,
        capabilities: { ...local25, t2v: false },
        features: HOME_FEATURES,
      }),
      { status: "blocked" },
    );
  });

  it("blocks when Home has no claimant for the tool", () => {
    assert.deepEqual(
      resolveFetcherLanding({
        slug: "t2v",
        isLoading: false,
        errorMessage: null,
        capabilities: local25,
        features: HOME_FEATURES.filter(
          (feature) => "fetcherTool" in feature && feature.fetcherTool === "i2v",
        ),
      }),
      { status: "blocked" },
    );
  });

  it("blocks cozy-felt when t2v or user LoRAs are unsupported", () => {
    assert.deepEqual(
      resolveFetcherLanding({
        slug: "cozy-felt",
        isLoading: false,
        errorMessage: null,
        capabilities: { ...local25, t2v: false },
        features: HOME_FEATURES,
      }),
      { status: "blocked" },
    );
    assert.deepEqual(
      resolveFetcherLanding({
        slug: "cozy-felt",
        isLoading: false,
        errorMessage: null,
        capabilities: { ...local25, user_loras: false },
        features: HOME_FEATURES,
      }),
      { status: "blocked" },
    );
  });

  it("blocks dolly-in when i2v or user LoRAs are unsupported", () => {
    assert.deepEqual(
      resolveFetcherLanding({
        slug: "dolly-in",
        isLoading: false,
        errorMessage: null,
        capabilities: { ...local25, i2v: false },
        features: HOME_FEATURES,
      }),
      { status: "blocked" },
    );
    assert.deepEqual(
      resolveFetcherLanding({
        slug: "dolly-in",
        isLoading: false,
        errorMessage: null,
        capabilities: { ...local25, user_loras: false },
        features: HOME_FEATURES,
      }),
      { status: "blocked" },
    );
  });

  it("still lands dolly-in when t2v is unsupported as long as i2v and user LoRAs are on", () => {
    assert.deepEqual(
      resolveFetcherLanding({
        slug: "dolly-in",
        isLoading: false,
        errorMessage: null,
        capabilities: { ...local25, t2v: false },
        features: HOME_FEATURES,
      }),
      { status: "redirect", path: homeFeaturePath("dolly-in") },
    );
  });

  it("redirects T2V, I2V, A2V, Retake, Extend, and listed LoRA recipes to the existing Home feature routes", () => {
    assert.deepEqual(
      resolveFetcherLanding({
        slug: "t2v",
        isLoading: false,
        errorMessage: null,
        capabilities: local25,
        features: HOME_FEATURES,
      }),
      { status: "redirect", path: homeFeaturePath("t2v") },
    );
    assert.deepEqual(
      resolveFetcherLanding({
        slug: "i2v",
        isLoading: false,
        errorMessage: null,
        capabilities: local25,
        features: HOME_FEATURES,
      }),
      { status: "redirect", path: homeFeaturePath("i2v") },
    );
    assert.deepEqual(
      resolveFetcherLanding({
        slug: "a2v",
        isLoading: false,
        errorMessage: null,
        capabilities: local25,
        features: HOME_FEATURES,
      }),
      { status: "redirect", path: homeFeaturePath("a2v") },
    );
    assert.deepEqual(
      resolveFetcherLanding({
        slug: "retake",
        isLoading: false,
        errorMessage: null,
        capabilities: { ...local25, retake: true },
        features: HOME_FEATURES,
      }),
      { status: "redirect", path: homeFeaturePath("retake") },
    );
    assert.deepEqual(
      resolveFetcherLanding({
        slug: "extend",
        isLoading: false,
        errorMessage: null,
        capabilities: { ...local25, extend: true },
        features: HOME_FEATURES,
      }),
      { status: "redirect", path: homeFeaturePath("extend") },
    );
    assert.deepEqual(
      resolveFetcherLanding({
        slug: "cozy-felt",
        isLoading: false,
        errorMessage: null,
        capabilities: local25,
        features: HOME_FEATURES,
      }),
      { status: "redirect", path: homeFeaturePath("cozy-felt") },
    );
    assert.deepEqual(
      resolveFetcherLanding({
        slug: "dolly-in",
        isLoading: false,
        errorMessage: null,
        capabilities: local25,
        features: HOME_FEATURES,
      }),
      { status: "redirect", path: homeFeaturePath("dolly-in") },
    );
  });
});
