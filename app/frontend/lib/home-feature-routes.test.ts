import assert from "node:assert/strict";
import { describe, it } from "node:test";
import type { ComponentType } from "react";
import { matchRoutes } from "react-router";
import { HOME_FEATURES, type HomeFeatureId } from "./home-features.ts";
import { paths } from "../paths.ts";
import {
  createHomeFeatureRoutes,
  selectDesktopHomeFeatureRouteComponents,
  selectRemoteHomeFeatureScreens,
  type HomeFeatureHostEntry,
} from "./home-feature-routes.ts";

function StubRoute() {
  return null;
}

function TextToVideoRoute() {
  return null;
}

function AudioToVideoRoute() {
  return null;
}

function DesktopOnlyRoute() {
  return null;
}

function FeatureChromeStub() {
  return null;
}

// Derived from HOME_FEATURES so a newly listed feature/recipe can't silently skip
// this completeness check. (A hardcoded map would go stale: *.test.ts is excluded
// from tsc, so a missing key wouldn't fail typecheck — production completeness is
// enforced by the Record<HomeFeatureId, ComponentType> type on router.tsx.)
const HOME_FEATURE_ROUTE_COMPONENTS = Object.fromEntries(
  HOME_FEATURES.map((feature) => [feature.id, StubRoute]),
) as Record<HomeFeatureId, ComponentType>;

describe("home feature routes", () => {
  it("registers every Home feature path with a route component", () => {
    const routes = createHomeFeatureRoutes(
      HOME_FEATURE_ROUTE_COMPONENTS,
      FeatureChromeStub,
    );

    assert.equal(routes.length, 1);
    assert.equal(routes[0]?.path, undefined);
    assert.deepEqual(
      routes[0]?.children?.map((route) => route.path),
      HOME_FEATURES.map((feature) => feature.path),
    );

    for (const feature of HOME_FEATURES) {
      assert.ok(matchRoutes(routes, feature.path), feature.path);
      const leaf = matchRoutes(routes, feature.path)?.at(-1)?.route;
      assert.equal(
        (leaf?.handle as { homeFeatureId?: string } | undefined)?.homeFeatureId,
        feature.id,
      );
    }
  });

  it("returns no routes when the host has no feature screens", () => {
    assert.deepEqual(createHomeFeatureRoutes({}, FeatureChromeStub), []);
  });

  it("omits features without a host screen from the route table", () => {
    const routes = createHomeFeatureRoutes(
      {
        "text-to-video": TextToVideoRoute,
      },
      FeatureChromeStub,
    );

    assert.deepEqual(
      routes[0]?.children?.map((route) => route.path),
      [paths.textToVideo],
    );
    assert.ok(matchRoutes(routes, paths.textToVideo));
    assert.equal(matchRoutes(routes, paths.imageToVideo), null);
  });
});

describe("Explore feature registry projections", () => {
  it("reuses one screen per feature and omits remote: false from the phone map", () => {
    const registry = {
      ...Object.fromEntries(
        HOME_FEATURES.map((feature) => [
          feature.id,
          { screen: DesktopOnlyRoute, remote: false as const },
        ]),
      ),
      "text-to-video": {
        screen: TextToVideoRoute,
        remote: true as const,
      },
      "audio-to-video": {
        screen: AudioToVideoRoute,
        remote: true as const,
      },
      retake: {
        screen: DesktopOnlyRoute,
        remote: false,
      },
      extend: {
        screen: DesktopOnlyRoute,
        remote: false,
      },
    } satisfies Record<HomeFeatureId, HomeFeatureHostEntry>;

    const desktop = selectDesktopHomeFeatureRouteComponents(registry);
    const remote = selectRemoteHomeFeatureScreens(registry);

    assert.deepEqual(
      Object.keys(desktop),
      HOME_FEATURES.map((feature) => feature.id),
    );
    assert.equal(desktop["text-to-video"], TextToVideoRoute);
    assert.equal(desktop["image-to-video"], DesktopOnlyRoute);
    assert.equal(desktop["audio-to-video"], AudioToVideoRoute);
    assert.deepEqual(Object.keys(remote), ["text-to-video", "audio-to-video"]);
    assert.equal(remote["text-to-video"], TextToVideoRoute);
    assert.equal(remote["text-to-video"], desktop["text-to-video"]);
    assert.equal(remote["audio-to-video"], AudioToVideoRoute);
    assert.equal(remote["audio-to-video"], desktop["audio-to-video"]);

    const remoteRoutes = createHomeFeatureRoutes(remote, FeatureChromeStub);
    assert.deepEqual(
      remoteRoutes[0]?.children?.map((route) => route.path),
      [paths.textToVideo, paths.audioToVideo],
    );
    assert.ok(matchRoutes(remoteRoutes, paths.audioToVideo));
  });
});
