import assert from "node:assert/strict";
import { createElement } from "react";
import { describe, it } from "node:test";
import { matchRoutes } from "react-router";
import {
  createDesktopAppRoutes,
  DESKTOP_HOME_LAYOUT_ROUTE_HANDLE,
} from "./desktop-routes.ts";
import { createHomeFeatureRoutes } from "./lib/home-feature-routes.ts";
import { HOME_FEATURES, type HomeFeatureId } from "./lib/home-features.ts";
import { paths } from "./paths.ts";

function RouteStub() {
  return null;
}

const homeFeatureScreens = Object.fromEntries(
  HOME_FEATURES.map((feature) => [feature.id, RouteStub]),
) as Record<HomeFeatureId, typeof RouteStub>;

describe("desktop hash router", () => {
  const routes = createDesktopAppRoutes({
    appShell: createElement(RouteStub),
    desktopHomeLayout: createElement(RouteStub),
    homeLayoutPages: {
      home: createElement(RouteStub),
      assets: createElement(RouteStub),
      dashboard: createElement(RouteStub),
      remote: createElement(RouteStub),
      projects: createElement(RouteStub),
      settings: createElement(RouteStub),
    },
    fetcherTool: createElement(RouteStub),
    homeFeatureRoutes: createHomeFeatureRoutes(homeFeatureScreens, RouteStub),
    projectLoader: async () => null,
    project: createElement(RouteStub),
  });

  it("routes #/projects beneath DesktopHomeLayout", () => {
    const matched = matchRoutes(routes, "/projects");
    assert.ok(matched);
    assert.equal(matched.at(-1)?.route.path, paths.projects);
    const layoutMatch = matched.find(
      (match) => match.route.handle === DESKTOP_HOME_LAYOUT_ROUTE_HANDLE,
    );
    assert.ok(layoutMatch);
  });

  it("routes #/projects/:projectId beneath DesktopHomeLayout", () => {
    const matched = matchRoutes(routes, "/projects/abc");
    assert.ok(matched);
    assert.equal(matched.at(-1)?.route.path, paths.project);
    const layoutMatch = matched.find(
      (match) => match.route.handle === DESKTOP_HOME_LAYOUT_ROUTE_HANDLE,
    );
    assert.ok(layoutMatch);
  });

  it("routes #/assets to AssetsLibraryScreen beneath DesktopHomeLayout", () => {
    const matched = matchRoutes(routes, "/assets");
    assert.ok(matched);

    const leaf = matched.at(-1);
    assert.equal(leaf?.route.path, paths.assets);
    assert.equal(leaf?.route.element?.type, RouteStub);
    assert.notEqual(leaf?.route.path, "*");

    const layoutMatch = matched.find(
      (match) => match.route.handle === DESKTOP_HOME_LAYOUT_ROUTE_HANDLE,
    );
    assert.ok(
      layoutMatch,
      "expected DesktopHomeLayout route handle in the matched route chain",
    );
    assert.equal(layoutMatch.route.handle?.id, "DesktopHomeLayout");
  });

  it("routes #/fetcher/:tool beneath DesktopHomeLayout", () => {
    const matched = matchRoutes(routes, "/fetcher/t2v");
    assert.ok(matched);

    const leaf = matched.at(-1);
    assert.equal(leaf?.route.path, paths.fetcherTool);
    assert.equal(leaf?.params.tool, "t2v");

    const layoutMatch = matched.find(
      (match) => match.route.handle === DESKTOP_HOME_LAYOUT_ROUTE_HANDLE,
    );
    assert.ok(layoutMatch);
  });
});
