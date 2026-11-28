import type { LoraRecipeId } from "./lora-recipes.ts";

export type HomeLoraGallerySection = {
  readonly heading: string;
  readonly featureIds: readonly LoraRecipeId[];
};

/** LoRA home rows — groupings mirror ltx.io `LORA_UI_CONFIG` (`style` / `motion` / `general`). */
export const HOME_LORA_GALLERY_SECTIONS = [
  {
    heading: "Styles",
    featureIds: [
      "cozy-felt",
      "claymation",
      "fantasy-painterly",
      "paper-cut-out-style",
      "cinemagraph",
    ],
  },
  {
    heading: "Camera & motion",
    featureIds: [
      "jib-up",
      "jib-down",
      "dolly-in",
      "dolly-out",
      "fpv-motion",
      "openwheel-t-cam",
      "transition",
      "vbvr",
    ],
  },
] as const satisfies readonly HomeLoraGallerySection[];

type GalleryRecipeId =
  (typeof HOME_LORA_GALLERY_SECTIONS)[number]["featureIds"][number];
type AssertNever<T extends never> = T;
/** `never` when every recipe is listed. A missing recipe fails typecheck. */
export type HomeLoraGalleryMissingRecipe = AssertNever<
  Exclude<LoraRecipeId, GalleryRecipeId>
>;
