import type { HomeFeatureId } from "./home-features.ts";
import { IC_LORA_RECIPES, type IcLoraRecipeId } from "./ic-lora-recipes.ts";
import {
  LORA_RECIPE_HOME_FEATURES,
  type LoraRecipeId,
} from "./lora-recipes.ts";
import {
  createHomeFeatureRoutes,
  selectDesktopHomeFeatureRouteComponents,
  selectRemoteHomeFeatureScreens,
  type HomeFeatureHostEntry,
  type RemoteHomeFeatureIdOf,
} from "./home-feature-routes.ts";
import { AudioToVideoFeatureScreen } from "@/ltx-io/screens/Feature/AudioToVideoFeatureScreen";
import { ExtendFeatureScreen } from "@/ltx-io/screens/Feature/ExtendFeatureScreen";
import { ImageToVideoFeatureScreen } from "@/ltx-io/screens/Feature/ImageToVideoFeatureScreen";
import { makeIcLoraRecipeScreen } from "../views/home/IcLoraRecipePage.tsx";
import { RetakeFeatureScreen } from "@/ltx-io/screens/Feature/RetakeFeatureScreen";
import { TextToVideoFeatureScreen } from "@/ltx-io/screens/Feature/TextToVideoFeatureScreen";
import { FeatureChromeLayout } from "@/ltx-io/screens/Feature/chrome/FeatureChromeLayout";
import { makeLoraRecipeScreen } from "../views/home/LoraRecipePage.tsx";

const IC_LORA_RECIPE_REGISTRY = Object.fromEntries(
  IC_LORA_RECIPES.map((recipe) => [
    recipe.id,
    { screen: makeIcLoraRecipeScreen(recipe.id), remote: true as const },
  ]),
) as {
  [K in IcLoraRecipeId]: {
    readonly screen: ReturnType<typeof makeIcLoraRecipeScreen>;
    readonly remote: true;
  };
};

const LORA_RECIPE_REGISTRY = Object.fromEntries(
  LORA_RECIPE_HOME_FEATURES.map((recipe) => [
    recipe.id,
    { screen: makeLoraRecipeScreen(recipe.id), remote: true as const },
  ]),
) as {
  [K in LoraRecipeId]: {
    readonly screen: ReturnType<typeof makeLoraRecipeScreen>;
    readonly remote: true;
  };
};

export const HOME_FEATURE_REGISTRY = {
  "text-to-video": {
    screen: TextToVideoFeatureScreen,
    remote: true,
  },
  "image-to-video": {
    screen: ImageToVideoFeatureScreen,
    remote: true,
  },
  "audio-to-video": {
    screen: AudioToVideoFeatureScreen,
    remote: true,
  },
  retake: {
    screen: RetakeFeatureScreen,
    remote: true,
  },
  extend: {
    screen: ExtendFeatureScreen,
    remote: true,
  },
  ...IC_LORA_RECIPE_REGISTRY,
  ...LORA_RECIPE_REGISTRY,
} as const satisfies Record<HomeFeatureId, HomeFeatureHostEntry>;

export type HomeFeatureRegistry = typeof HOME_FEATURE_REGISTRY;
export type RemoteHomeFeatureId = RemoteHomeFeatureIdOf<HomeFeatureRegistry>;

export const DESKTOP_HOME_FEATURE_ROUTE_COMPONENTS =
  selectDesktopHomeFeatureRouteComponents(HOME_FEATURE_REGISTRY);

export const REMOTE_HOME_FEATURE_SCREENS =
  selectRemoteHomeFeatureScreens(HOME_FEATURE_REGISTRY);

export const DESKTOP_HOME_FEATURE_ROUTES = createHomeFeatureRoutes(
  DESKTOP_HOME_FEATURE_ROUTE_COMPONENTS,
  FeatureChromeLayout,
);
