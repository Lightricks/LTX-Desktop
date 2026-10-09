import type { HomeFeatureId } from "./home-features.ts";
import { IC_LORA_RECIPES, type IcLoraRecipeId } from "./ic-lora-recipes.ts";
import { LORA_RECIPE_HOME_FEATURES } from "./lora-recipes.ts";
import {
  HOME_FEATURE_REGISTRY,
  type HomeFeatureRegistry,
  type RemoteHomeFeatureId,
} from "./home-feature-registry.ts";
import type {
  HomeFeatureHostEntry,
  RemoteHomeFeatureIdOf,
} from "./home-feature-routes.ts";

type Equal<A, B> =
  (<T>() => T extends A ? 1 : 2) extends <T>() => T extends B ? 1 : 2
    ? true
    : false;
type Expect<T extends true> = T;

function HostStub() {
  return null;
}

const registryKeysMatch: Expect<
  Equal<keyof HomeFeatureRegistry, HomeFeatureId>
> = true;
const currentRemoteIdsMatch: Expect<
  Equal<RemoteHomeFeatureId, HomeFeatureId>
> = true;
const registrySatisfiesHostEntries: Expect<
  HomeFeatureRegistry extends Record<HomeFeatureId, HomeFeatureHostEntry>
    ? true
    : false
> = true;

const desktopOnlyAllowed: HomeFeatureHostEntry = {
  screen: HostStub,
  remote: false,
};
const remoteSupportedAllowed: HomeFeatureHostEntry = {
  screen: HostStub,
  remote: true,
};

const loraDesktopOnly = Object.fromEntries(
  LORA_RECIPE_HOME_FEATURES.map((feature) => [
    feature.id,
    { screen: HostStub, remote: false as const },
  ]),
) as {
  [K in (typeof LORA_RECIPE_HOME_FEATURES)[number]["id"]]: {
    readonly screen: typeof HostStub;
    readonly remote: false;
  };
};

const icLoraDesktopOnly = Object.fromEntries(
  IC_LORA_RECIPES.map((recipe) => [
    recipe.id,
    { screen: HostStub, remote: false as const },
  ]),
) as {
  [K in IcLoraRecipeId]: {
    readonly screen: typeof HostStub;
    readonly remote: false;
  };
};

const optOutRegistry = {
  "text-to-video": { screen: HostStub, remote: true },
  "image-to-video": { screen: HostStub, remote: false },
  "audio-to-video": { screen: HostStub, remote: false },
  retake: { screen: HostStub, remote: false },
  extend: { screen: HostStub, remote: false },
  ...icLoraDesktopOnly,
  ...loraDesktopOnly,
} as const satisfies Record<HomeFeatureId, HomeFeatureHostEntry>;

const optOutOmitsDesktopOnly: Expect<
  Equal<RemoteHomeFeatureIdOf<typeof optOutRegistry>, "text-to-video">
> = true;

const productionRemoteCoversEveryFeature: Expect<
  Equal<RemoteHomeFeatureIdOf<HomeFeatureRegistry>, HomeFeatureId>
> = true;

// @ts-expect-error Remote support must be explicit (`remote: true` or `remote: false`).
const omittedRemote: HomeFeatureHostEntry = {
  screen: HostStub,
};
// @ts-expect-error Shared screen is required.
const omittedScreen: HomeFeatureHostEntry = {
  remote: true,
};
// @ts-expect-error Registry must include every HomeFeatureId.
const incompleteRegistry: Record<HomeFeatureId, HomeFeatureHostEntry> = {
  "text-to-video": { screen: HostStub, remote: true },
};

void [
  registryKeysMatch,
  currentRemoteIdsMatch,
  registrySatisfiesHostEntries,
  desktopOnlyAllowed,
  remoteSupportedAllowed,
  optOutOmitsDesktopOnly,
  productionRemoteCoversEveryFeature,
  omittedRemote,
  omittedScreen,
  incompleteRegistry,
  HOME_FEATURE_REGISTRY,
];
