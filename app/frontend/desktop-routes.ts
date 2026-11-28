import { createElement, type ReactNode } from "react";
import { Navigate, type RouteObject } from "react-router";
import { homeLayoutRouteSpecs, type HomeLayoutPageId } from "./lib/home-shell.ts";
import { paths } from "./paths.ts";

export const DESKTOP_HOME_LAYOUT_ROUTE_HANDLE = {
  id: "DesktopHomeLayout",
} as const;

export function createDesktopAppRoutes(options: {
  appShell: ReactNode;
  desktopHomeLayout: ReactNode;
  homeLayoutPages: Record<HomeLayoutPageId, ReactNode>;
  fetcherTool: ReactNode;
  homeFeatureRoutes: RouteObject[];
  projectLoader: RouteObject["loader"];
  project: ReactNode;
}): RouteObject[] {
  return [
    {
      element: options.appShell,
      children: [
        {
          element: options.desktopHomeLayout,
          handle: DESKTOP_HOME_LAYOUT_ROUTE_HANDLE,
          children: [
            ...homeLayoutRouteSpecs("desktop").map((spec) => ({
              path: spec.path,
              element: options.homeLayoutPages[spec.id],
            })),
            { path: paths.fetcherTool, element: options.fetcherTool },
            ...options.homeFeatureRoutes,
            {
              path: paths.project,
              loader: options.projectLoader,
              element: options.project,
            },
          ],
        },
        { path: "*", element: createElement(Navigate, { to: paths.home, replace: true }) },
      ],
    },
  ];
}
