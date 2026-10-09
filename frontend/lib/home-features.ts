import { matchPath } from "react-router";
import { paths } from "../paths.ts";
import type { FetcherToolId } from "../../shared/deep-link.ts";
import { IC_LORA_RECIPES } from "./ic-lora-recipes.ts";
import { LORA_RECIPE_HOME_FEATURES } from "./lora-recipes.ts";

type AppPath = (typeof paths)[keyof typeof paths];
type HomeFeaturePath = Exclude<
  AppPath,
  | typeof paths.home
  | typeof paths.projects
  | typeof paths.project
  | typeof paths.assets
  | typeof paths.fetcherTool
>;

type HomeFeatureDefinitionShape = {
  readonly id: string;
  readonly path: HomeFeaturePath;
  readonly title: string;
  readonly description: string;
  readonly fetcherTool?: FetcherToolId;
};

/** LTX.io FeatureDetails copy. Home card `description` stays the short line. */
export type HomeFeatureListing = {
  readonly blurb: string;
  readonly categories: string;
  readonly typeLabel: string;
  /** Model card. Shown as the details “Hugging Face” button when set. */
  readonly huggingfaceUrl?: string;
};

/**
 * Home-feature rows for every IC-LoRA recipe, derived from IC_LORA_RECIPES so
 * title / description / path / listing have a single source of truth.
 * `fetcherTool: recipe.id` is checked against FETCHER_TOOL_SLUGS by the type.
 */
const IC_LORA_HOME_FEATURES = IC_LORA_RECIPES.map((recipe) => ({
  id: recipe.id,
  path: recipe.path,
  title: recipe.title,
  description: recipe.description,
  fetcherTool: recipe.id,
  listing: recipe.listing,
}));

export const FIRST_CLASS_HOME_FEATURES = [
  {
    id: "text-to-video",
    path: paths.textToVideo,
    title: "Text to Video",
    description: "Generate a video from a text prompt with matching audio.",
    fetcherTool: "t2v",
    listing: {
      blurb:
        "Generate a video from a text prompt. Receive a high-quality video with synchronized audio, refined motion, and rich detail, optimized for portrait or landscape formats.",
      categories: "Workflow",
      typeLabel: "Text to Video",
    },
  },
  {
    id: "image-to-video",
    path: paths.imageToVideo,
    title: "Image to Video",
    description: "Animate a still image into a short clip.",
    fetcherTool: "i2v",
    listing: {
      blurb:
        "Bring a still image to life by describing how it should move. Provide a single frame or both a start and end frame, and receive smooth, natural animation that stays true to your original composition.",
      categories: "Motion",
      typeLabel: "Image to Video",
    },
  },
  {
    id: "audio-to-video",
    path: paths.audioToVideo,
    title: "Audio to Video",
    description: "Generate a video synchronized to an audio track.",
    fetcherTool: "a2v",
    listing: {
      blurb:
        "Turn any audio track into video. Provide music, dialogue, or sound effects, and the visuals you describe will be generated in perfect sync with the rhythm, timing, and energy of your audio.",
      categories: "Audio, Animation",
      typeLabel: "Audio to Video",
    },
  },
  {
    id: "retake",
    path: paths.retake,
    title: "Retake",
    description: "Replace a region of an existing clip.",
    fetcherTool: "retake",
    listing: {
      blurb:
        "Refine a video without starting over. Select the segment that needs work, describe the change, and receive a regenerated version that blends seamlessly into the surrounding footage.",
      categories: "Post production",
      typeLabel: "Retake video",
    },
  },
  {
    id: "extend",
    path: paths.extend,
    title: "Extend",
    description: "Grow an existing clip forward or backward.",
    fetcherTool: "extend",
    listing: {
      blurb:
        "Continue a video beyond its original length. Feed in a short clip and receive a natural extension that preserves motion and scene continuity, chainable up to 60 seconds.",
      categories: "Post production",
      typeLabel: "Extend Video",
    },
  },
  ...IC_LORA_HOME_FEATURES,
] as const satisfies readonly (HomeFeatureDefinitionShape & {
  readonly listing: HomeFeatureListing;
})[];

type FirstClassFeature = (typeof FIRST_CLASS_HOME_FEATURES)[number];
type FirstClassFeatureId = FirstClassFeature["id"];

/** AlphaGen is hidden until it ships. Past results stay in the queue and Assets. */
const ALPHA_GEN_ENABLED = false;

/**
 * False for a feature kept off Home, the sidebar, Explore Tools, and the Settings and Setup
 * LoRA lists (an IC-LoRA catalog id is its feature id). Its route still opens.
 */
export function isHomeFeatureEnabled(id: string): boolean {
  return id !== "alpha-gen" || ALPHA_GEN_ENABLED;
}

function firstClassFeature(id: FirstClassFeatureId): FirstClassFeature {
  const feature = FIRST_CLASS_HOME_FEATURES.find((item) => item.id === id);
  if (!feature) {
    throw new Error(`Unknown first-class Home feature: ${id}`);
  }
  return feature;
}

/**
 * Gallery and quick-search rows for first-class features. Every first-class
 * feature belongs to exactly one section (enforced below and in tests).
 */
const HOME_FEATURE_SECTION_DEFINITIONS = [
  {
    id: "workflows",
    heading: "Workflows",
    featureIds: ["text-to-video", "image-to-video", "audio-to-video"],
  },
  {
    id: "vfx",
    heading: "VFX",
    featureIds: IC_LORA_RECIPES.filter((recipe) => recipe.section === "vfx").map(
      (recipe) => recipe.id,
    ),
  },
  {
    id: "post-production",
    heading: "Post production",
    featureIds: [
      "retake",
      "extend",
      ...IC_LORA_RECIPES.filter((recipe) => recipe.section === "post-production").map(
        (recipe) => recipe.id,
      ),
    ],
  },
] as const satisfies readonly {
  readonly id: string;
  readonly heading: string;
  readonly featureIds: readonly FirstClassFeatureId[];
}[];

type SectionedFeatureId =
  (typeof HOME_FEATURE_SECTION_DEFINITIONS)[number]["featureIds"][number];
type AssertNever<T extends never> = T;
/** `never` when every first-class feature is in a section. A missing one fails typecheck. */
export type HomeFeatureSectionsMissingFeature = AssertNever<
  Exclude<FirstClassFeatureId, SectionedFeatureId>
>;

export const HOME_FEATURE_SECTIONS = HOME_FEATURE_SECTION_DEFINITIONS.map(
  (section) => ({
    id: section.id,
    heading: section.heading,
    features: section.featureIds.filter(isHomeFeatureEnabled).map(firstClassFeature),
  }),
);

const HOME_SIDEBAR_FEATURED_FEATURE_IDS = [
  "image-to-video",
  "audio-to-video",
  "extend",
  "alpha-gen",
] as const satisfies readonly FirstClassFeatureId[];

/** Sidebar “Featured” list — first-class features only (not LoRA recipes, not Day to Night). */
export const HOME_SIDEBAR_FEATURED_FEATURES = HOME_SIDEBAR_FEATURED_FEATURE_IDS.filter(
  isHomeFeatureEnabled,
).map(firstClassFeature);

export const HOME_FEATURES = [
  ...FIRST_CLASS_HOME_FEATURES,
  ...LORA_RECIPE_HOME_FEATURES,
] as const satisfies readonly HomeFeatureDefinitionShape[];

export type HomeFeatureId = (typeof HOME_FEATURES)[number]["id"];
export type HomeFeatureDefinition = (typeof HOME_FEATURES)[number];

export function isHomeFeatureId(
  value: string | undefined,
): value is HomeFeatureId {
  return (
    value !== undefined && HOME_FEATURES.some((feature) => feature.id === value)
  );
}

export function getHomeFeature(id: HomeFeatureId): HomeFeatureDefinition {
  const feature = HOME_FEATURES.find((candidate) => candidate.id === id);
  if (!feature) {
    throw new Error(`Unknown Home feature: ${id}`);
  }
  return feature;
}

export function matchHomeFeature(
  pathname: string,
): HomeFeatureDefinition | undefined {
  return HOME_FEATURES.find(
    (feature) => matchPath({ path: feature.path, end: true }, pathname) != null,
  );
}
