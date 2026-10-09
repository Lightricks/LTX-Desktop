import { createElement, type ComponentType, type ReactNode } from "react";
import { Navigate, type RouteObject } from "react-router";
import { createHomeFeatureRoutes } from "../lib/home-feature-routes.ts";
import type { HomeFeatureId } from "../lib/home-features.ts";
import { homeLayoutRouteSpecs } from "../lib/home-shell.ts";
import { paths } from "../paths.ts";

export function RemoteUnknownPathRedirect() {
  return createElement(Navigate, {
    to: paths.home,
    replace: true,
  });
}

export function createRemoteAppRoutes(options: {
  errorElement: ReactNode;
  pairing: ReactNode;
  pairedLayout: ReactNode;
  homeLayoutPages: Record<"home" | "assets" | "dashboard", ReactNode>;
  screens: Partial<Record<HomeFeatureId, ComponentType>>;
  featureLayout: ComponentType;
}): RouteObject[] {
  return [
    {
      errorElement: options.errorElement,
      children: [
        { path: "pairing", element: options.pairing },
        {
          element: options.pairedLayout,
          children: [
            ...homeLayoutRouteSpecs("remote").map((spec) => {
              if (spec.id !== "home" && spec.id !== "assets" && spec.id !== "dashboard") {
                throw new Error(`Remote host does not render ${spec.id}`);
              }
              return {
                path: spec.path,
                element: options.homeLayoutPages[spec.id],
              };
            }),
            ...createHomeFeatureRoutes(options.screens, options.featureLayout),
            { path: "*", element: createElement(RemoteUnknownPathRedirect) },
          ],
        },
      ],
    },
  ];
}
