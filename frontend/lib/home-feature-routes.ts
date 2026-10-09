import { createElement, type ComponentType } from "react";
import type { RouteObject } from "react-router";
import { HOME_FEATURES, type HomeFeatureId } from "./home-features.ts";

export type HomeFeatureHostEntry = {
  readonly screen: ComponentType;
  readonly remote: boolean;
};

export type RemoteHomeFeatureIdOf<
  T extends Record<HomeFeatureId, HomeFeatureHostEntry>,
> = {
  [K in HomeFeatureId]: T[K]["remote"] extends true ? K : never;
}[HomeFeatureId];

export function selectDesktopHomeFeatureRouteComponents(
  registry: Record<HomeFeatureId, HomeFeatureHostEntry>,
): Record<HomeFeatureId, ComponentType> {
  const components = {} as Record<HomeFeatureId, ComponentType>;
  for (const feature of HOME_FEATURES) {
    components[feature.id] = registry[feature.id].screen;
  }
  return components;
}

export function selectRemoteHomeFeatureScreens(
  registry: Record<HomeFeatureId, HomeFeatureHostEntry>,
): Partial<Record<HomeFeatureId, ComponentType>> {
  const screens: Partial<Record<HomeFeatureId, ComponentType>> = {};
  for (const feature of HOME_FEATURES) {
    const entry = registry[feature.id];
    if (entry.remote) {
      screens[feature.id] = entry.screen;
    }
  }
  return screens;
}

export type HomeFeatureRouteHandle = { readonly homeFeatureId: HomeFeatureId };

export function createHomeFeatureRoutes(
  components: Partial<Record<HomeFeatureId, ComponentType>>,
  layout: ComponentType,
): RouteObject[] {
  const children = HOME_FEATURES.flatMap((feature) => {
    const Component = components[feature.id];
    if (!Component) {
      return [];
    }
    return [
      {
        path: feature.path,
        element: createElement(Component),
        handle: { homeFeatureId: feature.id } satisfies HomeFeatureRouteHandle,
      },
    ];
  });
  if (children.length === 0) {
    return [];
  }
  return [{ element: createElement(layout), children }];
}
