import assert from "node:assert/strict";
import { createElement } from "react";
import { describe, it } from "node:test";
import { matchRoutes } from "react-router";
import type { ComponentType } from "react";
import { HOME_FEATURES, type HomeFeatureId } from "../lib/home-features.ts";
import { paths } from "../paths.ts";
import { createRemoteAppRoutes } from "./remote-routes.ts";

function RouteStub() {
  return null;
}

const screens = Object.fromEntries(
  HOME_FEATURES.map((feature) => [feature.id, RouteStub]),
) as Record<HomeFeatureId, ComponentType>;

describe("remote app routes", () => {
  const routes = createRemoteAppRoutes({
    errorElement: createElement(RouteStub),
    pairing: createElement(RouteStub),
    pairedLayout: createElement(RouteStub),
    homeLayoutPages: {
      home: createElement(RouteStub),
      assets: createElement(RouteStub),
      dashboard: createElement(RouteStub),
    },
    screens,
    featureLayout: RouteStub,
  });

  it("keeps a root error boundary, pairing route, and a wildcard fallback", () => {
    assert.equal(routes.length, 1);
    assert.ok(routes[0]?.errorElement);

    const pairing = routes[0]?.children?.find((route) => route.path === "pairing");
    assert.ok(pairing);

    const paired = routes[0]?.children?.find((route) => route.children);
    const splat = paired?.children?.find((route) => route.path === "*");
    assert.ok(splat);
  });

  it("matches /pairing instead of the unknown-path splat", () => {
    const matched = matchRoutes(routes, "/pairing");
    assert.equal(matched?.at(-1)?.route.path, "pairing");
  });

  it("still matches Home and every Explore feature path", () => {
    assert.ok(matchRoutes(routes, "/"));
    for (const feature of HOME_FEATURES) {
      assert.ok(matchRoutes(routes, feature.path), feature.path);
    }

    const unknown = matchRoutes(routes, "/renamed-explore");
    assert.equal(unknown?.at(-1)?.route.path, "*");
  });

  it("matches /activity-dashboard instead of the wildcard", () => {
    const matched = matchRoutes(routes, "/activity-dashboard");
    assert.equal(matched?.at(-1)?.route.path, paths.dashboard);
  });

  it("matches the Assets path instead of the wildcard", () => {
    const matched = matchRoutes(routes, "/assets");
    assert.equal(matched?.at(-1)?.route.path, paths.assets);
    assert.notEqual(matched?.at(-1)?.route.path, "*");
  });
});
