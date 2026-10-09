import { paths } from "../paths.ts";
import type { PackagedSeedKind } from "../../shared/explore-seed-filenames.ts";
import {
  icLoraSeedFilename,
  LAYOUT_TO_RENDER_IMAGE_SEED_FILENAME,
  LAYOUT_TO_RENDER_IMAGE_SEED_PREVIOUS_FILENAMES,
  LAYOUT_TO_RENDER_IMAGE_SEED_REVISION,
  RESTORE_IMAGE_SEED_FILENAME,
  RESTORE_IMAGE_SEED_PREVIOUS_FILENAMES,
  RESTORE_IMAGE_SEED_REVISION,
} from "../../shared/explore-seed-filenames.ts";

type SeedIdOf<Kind> = Kind extends `${infer Id}-video` ? Id : never;
/** Packaged `<id>-video` kinds of recipes. Retake and Extend have video kinds but no recipe row. */
type IcLoraSeedId = Exclude<SeedIdOf<PackagedSeedKind>, "retake" | "extend">;

/** The packaged reference clip of a recipe. Bump `revision` when the clip changes. */
function icLoraSeed<const Id extends IcLoraSeedId>(id: Id, revision = 1) {
  return {
    filename: icLoraSeedFilename(id),
    previousFilenames: [] as readonly string[],
    revision,
    storageKey: `ltx.${id}-seed-revision`,
    kind: `${id}-video` as const,
  };
}

/** The Home section that lists a recipe. A row without a valid section fails typecheck. */
export type IcLoraRecipeSection = "post-production" | "vfx";

/** Form controls a recipe fixes to its catalog defaults instead of showing. */
export type IcLoraRecipeHiddenField = "prompt" | "audioMode" | "loraStrength";

/**
 * FE mirror of backend/services/features/ic_lora_recipes.py.
 * `referenceImage` adds an image input. Whether the image is required comes from the
 * catalog `reference_image_required`, not from this table. `seed` is the packaged
 * default image, or null when the form starts with no image (an optional image has none).
 * shared/ic-lora-recipes.json pins id and catalogId. This table adds the copy,
 * the packaged reference clip, and the details listing. Style strength and
 * audio mode stay in the catalog default_settings.
 */
export const IC_LORA_RECIPES = [
  {
    id: "day-to-night",
    catalogId: "day-to-night",
    path: paths.dayToNight,
    title: "Day to Night",
    description: "Relight a clip from day into night.",
    demoPrompt:
      "A realistic nighttime scene of a fisherman fishing in a lake under a dark midnight sky with a bright crescent moon casting a silvery reflection on the water. Cool blue ambient moonlight and deep natural shadows surround the serene landscape, with a single warm glow from a lantern nearby illuminating the water's edge. Only the lighting changes from day to night; identical composition, framing, camera movement and motion.",
    placeholder: "Describe the night light you want...",
    hiddenFields: [],
    cutout: false,
    section: "post-production",
    seed: icLoraSeed("day-to-night"),
    listing: {
      blurb:
        "Turn a daytime clip into night. Keep the motion and the framing, and describe the light you want.",
      categories: "Post production",
      typeLabel: "Day to Night",
      huggingfaceUrl:
        "https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Day-To-Night",
    },
  },
  {
    id: "alpha-gen",
    catalogId: "alpha-gen",
    path: paths.alphaGen,
    title: "AlphaGen",
    description: "Cut a subject out of a video without a green screen.",
    // AlphaGen takes no prompt. The model picks the foreground on its own.
    demoPrompt: "",
    placeholder: "Leave empty. AlphaGen needs no prompt.",
    // The model needs no prompt, keeps the clip audio, and has one useful strength.
    hiddenFields: ["prompt", "audioMode", "loraStrength"],
    // The result is a WebM with alpha, shown over a backdrop the user picks.
    cutout: true,
    section: "vfx",
    seed: icLoraSeed("alpha-gen"),
    listing: {
      blurb:
        "Isolate a person or object from any video, with no green screen. Hair, smoke, glass and water keep their soft edges. The result is a transparent cutout that you can place on any background.",
      categories: "VFX",
      typeLabel: "AlphaGen",
      huggingfaceUrl:
        "https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Alpha-Gen",
    },
  },
  {
    id: "deblur",
    catalogId: "deblur",
    path: paths.deblur,
    title: "Deblur",
    description: "Restore a blurry clip to sharp focus.",
    demoPrompt:
      "DEBLUR The same scene in sharp focus with crisp detail and clean edges.",
    placeholder: "Start with DEBLUR, then describe the sharp scene...",
    hiddenFields: [],
    cutout: false,
    section: "post-production",
    seed: icLoraSeed("deblur"),
    listing: {
      blurb:
        "Recover sharpness from soft or out-of-focus footage. Keep the subject and the framing, and receive a clean, sharp version of the clip.",
      categories: "Post production",
      typeLabel: "Deblur",
      huggingfaceUrl:
        "https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Deblur",
    },
  },
  {
    id: "colorization",
    catalogId: "colorization",
    path: paths.colorization,
    title: "Colorization",
    description: "Add natural color to a grayscale clip.",
    demoPrompt:
      "COLORIZE The tiger is colored with rich burnt-orange and golden-amber fur with deep black stripes, a keen amber eye. Dusty grey nose, prominent white whiskers catching the light, and creamy white facial markings on the muzzle. The background shows soft, blurred green and grey-brown foliage suggesting a natural habitat. The lighting is warm and directional.",
    placeholder: "Start with COLORIZE, then describe the colors you want...",
    hiddenFields: [],
    cutout: false,
    section: "post-production",
    seed: icLoraSeed("colorization"),
    listing: {
      blurb:
        "Colorize grayscale or faded video. Receive natural color that respects lighting, shadows and identity, and keeps the original motion and composition.",
      categories: "Post production",
      typeLabel: "Colorization",
      huggingfaceUrl:
        "https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Colorization",
    },
  },
  {
    id: "clean-plate",
    catalogId: "clean-plate",
    path: paths.cleanPlate,
    title: "Clean Plate",
    description: "Remove people and vehicles from a clip.",
    demoPrompt:
      "The source video, with no people and no vehicles anywhere in the frame.",
    placeholder: "Describe the empty scene...",
    hiddenFields: [],
    cutout: false,
    section: "post-production",
    seed: icLoraSeed("clean-plate", 2),
    listing: {
      blurb:
        "Make a clean background plate for VFX and compositing. Remove people, pedestrians and vehicles, and rebuild the empty scene with the static background kept intact.",
      categories: "Post production",
      typeLabel: "Clean Plate",
      huggingfaceUrl:
        "https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Clean-Plate",
    },
  },
  {
    id: "decompression",
    catalogId: "decompression",
    path: paths.decompression,
    title: "Decompression",
    description: "Remove compression artifacts from a clip.",
    demoPrompt:
      "ENHANCE QUALITY The same scene restored to high quality with sharp detail, clean edges, and no compression artifacts.",
    placeholder: "Start with ENHANCE QUALITY, then describe the clean scene...",
    hiddenFields: [],
    cutout: false,
    section: "post-production",
    seed: icLoraSeed("decompression"),
    listing: {
      blurb:
        "Restore quality to compressed video. Remove macroblocking, chroma bleed, ringing and banding from low-bitrate footage, and keep the identity and framing.",
      categories: "Post production",
      typeLabel: "Decompression",
      huggingfaceUrl:
        "https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Decompression",
    },
  },
  {
    id: "water-simulation",
    catalogId: "water-simulation",
    path: paths.waterSimulation,
    title: "Water Simulation",
    description: "Add rivers, rain and splashes to a clip.",
    demoPrompt:
      "ADD WATER A clear, meandering river winds through the middle distance of the wheat fields, its surface catching the golden hour light with amber and cream reflections. The water flows calm and steady, partially visible between the grain, with darker banks lined by reeds and grasses. The river curves gently toward the tree line, reflecting the warm overcast sky.",
    placeholder: "Start with ADD WATER, then describe the water you want...",
    hiddenFields: [],
    cutout: false,
    section: "post-production",
    seed: icLoraSeed("water-simulation"),
    listing: {
      blurb:
        "Add water to any shot, such as rivers, surf, rain, floods and splashes. The subject, the framing and the camera move stay exactly as filmed.",
      categories: "Post production",
      typeLabel: "Water Simulation",
      huggingfaceUrl:
        "https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Water-Simulation",
    },
  },
  {
    id: "layout-to-render",
    catalogId: "layout-to-render",
    path: paths.layoutToRender,
    title: "Layout to Render",
    description: "Turn a plain 3D layout into a finished, photoreal shot.",
    // Replace with the ltx.io Layout to Render prompt when it is confirmed.
    demoPrompt:
      "A green locomotive pulls red and blue freight cars along a curved track through a snowy mountain valley, past snow-covered pines and over a stone viaduct, under an overcast sky.",
    placeholder: "Describe the finished shot in one sentence...",
    hiddenFields: [],
    cutout: false,
    // The still sets the look. The model reads it as a frame before the clip.
    referenceImage: {
      label: "Look Image",
      seed: {
        kind: "layout-to-render-image" satisfies PackagedSeedKind,
        filename: LAYOUT_TO_RENDER_IMAGE_SEED_FILENAME,
        previousFilenames: LAYOUT_TO_RENDER_IMAGE_SEED_PREVIOUS_FILENAMES,
        revision: LAYOUT_TO_RENDER_IMAGE_SEED_REVISION,
      },
    },
    section: "vfx",
    seed: icLoraSeed("layout-to-render"),
    listing: {
      blurb:
        "Turn a plain 3D layout into a finished shot. The layout sets the motion and the framing. A still image sets the colors, the materials and the light.",
      categories: "VFX",
      typeLabel: "Layout to Render",
      huggingfaceUrl:
        "https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Layout-To-Render",
    },
  },
  {
    id: "restore",
    catalogId: "restore",
    path: paths.restore,
    title: "Restore",
    description: "Restore old footage as a clean, colour video.",
    // The 1896 example prompt of the Restore page. Replace with the ltx.io prompt when the
    // recipe exists there.
    demoPrompt:
      "a steam locomotive arriving at a small rural railway station in 1896, passengers in period coats and hats waiting on the platform, wooden station buildings, natural colour",
    placeholder: "Describe the period, the place and the light...",
    hiddenFields: [],
    cutout: false,
    // A low-resolution source is the intended input (the model trained on 240p to 360p
    // scans). The form offers every resolution cell whatever the source size.
    upscalesSource: true,
    // An optional look image. The model reads it as the first frame of the clip. The
    // packaged one is the restored first frame of the packaged clip.
    referenceImage: {
      label: "Look Image",
      seed: {
        kind: "restore-image" satisfies PackagedSeedKind,
        filename: RESTORE_IMAGE_SEED_FILENAME,
        previousFilenames: RESTORE_IMAGE_SEED_PREVIOUS_FILENAMES,
        revision: RESTORE_IMAGE_SEED_REVISION,
      },
    },
    section: "vfx",
    seed: icLoraSeed("restore"),
    listing: {
      blurb:
        "Restore archive footage. Remove compression damage, tape and sepia casts, flicker, dirt and scratches, and render the same shot in natural colour. Deinterlace an interlaced clip first.",
      categories: "VFX",
      typeLabel: "Restore",
      huggingfaceUrl:
        "https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Restore",
    },
  },
] as const satisfies readonly {
  readonly section: IcLoraRecipeSection;
  readonly [key: string]: unknown;
}[];

export type IcLoraRecipeId = (typeof IC_LORA_RECIPES)[number]["id"];
/** An image input of a recipe. A row without one omits `referenceImage`. */
export interface IcLoraReferenceImage {
  readonly label: string;
  /** The packaged default image, or null when the form starts with no image. */
  readonly seed: {
    readonly kind: PackagedSeedKind;
    readonly filename: string;
    readonly previousFilenames: readonly string[];
    /** Bump when the packaged image changes. */
    readonly revision: number;
  } | null;
}

export type IcLoraRecipe = (typeof IC_LORA_RECIPES)[number] & {
  readonly referenceImage?: IcLoraReferenceImage;
  /**
   * The recipe runs one tiled stage and upscales a small source. The resolution control
   * then offers every cell, not only the cells the source can fill. A row without it
   * keeps the source-size limit.
   */
  readonly upscalesSource?: boolean;
};

/**
 * The image field label. An optional image says so.
 * `required` is the catalog `reference_image_required`. It is undefined until the catalog loads.
 */
export function icLoraImageLabel(
  image: IcLoraReferenceImage,
  required: boolean | undefined,
): string {
  return required === false ? `${image.label} (optional)` : image.label;
}

/** The value of the image input. It is null when the recipe has no image input. */
export function icLoraImageValue<TImage>(
  recipe: IcLoraRecipe,
  value: TImage | null | undefined,
): TImage | null {
  return recipe.referenceImage == null ? null : (value ?? null);
}

/**
 * The image input the recipe needs and `value` does not give. Null when nothing is missing.
 * `required` is the catalog `reference_image_required`.
 */
export function missingIcLoraImage(
  recipe: IcLoraRecipe,
  value: unknown,
  required: boolean | undefined,
): IcLoraReferenceImage | null {
  const image = recipe.referenceImage;
  return image != null && required === true && value == null ? image : null;
}

/** False while a recipe with an image input waits for the catalog flag. No default is assumed. */
export function isIcLoraImageRequirementKnown(
  recipe: IcLoraRecipe,
  required: boolean | undefined,
): boolean {
  return recipe.referenceImage == null || required !== undefined;
}

export function isIcLoraFieldHidden(
  recipe: IcLoraRecipe,
  field: IcLoraRecipeHiddenField,
): boolean {
  const hidden: readonly IcLoraRecipeHiddenField[] = recipe.hiddenFields;
  return hidden.includes(field);
}

export function isIcLoraRecipeId(value: string | undefined): value is IcLoraRecipeId {
  return IC_LORA_RECIPES.some((recipe) => recipe.id === value);
}

export function getIcLoraRecipe(id: IcLoraRecipeId): IcLoraRecipe {
  const recipe = IC_LORA_RECIPES.find((candidate) => candidate.id === id);
  if (!recipe) {
    throw new Error(`Unknown IC-LoRA recipe: ${id}`);
  }
  return recipe;
}
