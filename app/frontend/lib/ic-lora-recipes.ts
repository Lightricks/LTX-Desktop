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
  VIDEO_GUIDE_IMAGE_SEED,
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

/** The most images an image-list recipe takes. The catalog `input.max_count` is the same number. */
export const IC_LORA_SHEET_MAX_IMAGES = 10;

/** The Home section that lists a recipe. A row without a valid section fails typecheck. */
export type IcLoraRecipeSection = "post-production" | "vfx" | "marketing";

/**
 * What a recipe takes as input: one clip (`video`), a list of images that join into one
 * sheet (`sheet`), or one image with a path drawn on it (`track`).
 */
export type IcLoraRecipeKind = "video" | "sheet" | "track";

/** Form controls a recipe fixes to its catalog defaults instead of showing. */
export type IcLoraRecipeHiddenField = "prompt" | "audioMode" | "loraStrength";

/**
 * FE mirror of backend/services/features/ic_lora_recipes.py.
 * `referenceImage` adds an image input. Whether the image is required comes from the
 * catalog `reference_image_required`, not from this table. `seed` is the packaged
 * default image, or null when the form starts with no image (an optional image has none).
 * shared/ic-lora-recipes.json pins id, catalogId and kind. This table adds the copy,
 * the packaged reference clip, and the details listing. Style strength and
 * audio mode stay in the catalog default_settings. A sheet or track row sets its own
 * Home defaults (`homeDefaults`), so GenSpace keeps the catalog value.
 */
export const IC_LORA_RECIPES = [
  {
    id: "day-to-night",
    kind: "video",
    catalogId: "day-to-night",
    path: paths.dayToNight,
    title: "Day to Night",
    description: "Relight a clip from day into night.",
    demoPrompt:
      "A realistic nighttime scene of a fisherman fishing in a lake under a dark midnight sky with a bright crescent moon casting a silvery reflection on the water. Cool blue ambient moonlight and deep natural shadows surround the serene landscape, with a single warm glow from a lantern nearby illuminating the water's edge. Only the lighting changes from day to night; identical composition, framing, camera movement and motion.",
    demoNegativePrompt:
      "daytime, bright sunlight, blue sky, harsh sun glare, overexposed water, pitch black crushed shadows, blurry fisherman, jittery motion",
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
    kind: "video",
    catalogId: "alpha-gen",
    path: paths.alphaGen,
    title: "AlphaGen",
    description: "Cut a subject out of a video without a green screen.",
    // AlphaGen takes no prompt. The model picks the foreground on its own.
    demoPrompt: "",
    demoNegativePrompt: "",
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
    kind: "video",
    catalogId: "deblur",
    path: paths.deblur,
    title: "Deblur",
    description: "Restore a blurry clip to sharp focus.",
    demoPrompt:
      "DEBLUR The same scene in sharp focus with crisp detail and clean edges.",
    demoNegativePrompt:
      "blurry, out of focus, defocused, soft, hazy, smeared detail, oversharpened halos, ringing, jittery motion",
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
    kind: "video",
    catalogId: "colorization",
    path: paths.colorization,
    title: "Colorization",
    description: "Add natural color to a grayscale clip.",
    demoPrompt:
      "COLORIZE The tiger is colored with rich burnt-orange and golden-amber fur with deep black stripes, a keen amber eye. Dusty grey nose, prominent white whiskers catching the light, and creamy white facial markings on the muzzle. The background shows soft, blurred green and grey-brown foliage suggesting a natural habitat. The lighting is warm and directional.",
    demoNegativePrompt:
      "grayscale, black and white, monochrome, desaturated fur, sepia tint, neon unnatural colors, color bleeding across stripes, blurry whiskers",
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
    kind: "video",
    catalogId: "clean-plate",
    path: paths.cleanPlate,
    title: "Clean Plate",
    description: "Remove people and vehicles from a clip.",
    demoPrompt:
      "The source video, with no people and no vehicles anywhere in the frame.",
    demoNegativePrompt:
      "person, people, human, figure, pedestrian, vehicle, car, limb, shadow of a person, ghosting, semi-transparent remnants, blurry background",
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
    kind: "video",
    catalogId: "decompression",
    path: paths.decompression,
    title: "Decompression",
    description: "Remove compression artifacts from a clip.",
    demoPrompt:
      "ENHANCE QUALITY The same scene restored to high quality with sharp detail, clean edges, and no compression artifacts.",
    demoNegativePrompt:
      "macroblocking, blocky compression, mosquito noise, chroma bleed, ringing, color banding, pixelated edges, smeared texture, blurry",
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
    kind: "video",
    catalogId: "water-simulation",
    path: paths.waterSimulation,
    title: "Water Simulation",
    description: "Add rivers, rain and splashes to a clip.",
    demoPrompt:
      "ADD WATER A clear, meandering river winds through the middle distance of the wheat fields, its surface catching the golden hour light with amber and cream reflections. The water flows calm and steady, partially visible between the grain, with darker banks lined by reeds and grasses. The river curves gently toward the tree line, reflecting the warm overcast sky.",
    demoNegativePrompt:
      "deformed water, unnatural fluid, frozen water, dry cracked riverbed, flooded wheat stalks, warped horizon, jittery ripples, blurry",
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
    kind: "video",
    catalogId: "layout-to-render",
    path: paths.layoutToRender,
    title: "Layout to Render",
    description: "Turn a plain 3D layout into a finished, photoreal shot.",
    // Replace with the ltx.io Layout to Render prompt when it is confirmed.
    demoPrompt:
      "A green locomotive pulls red and blue freight cars along a curved track through a snowy mountain valley, past snow-covered pines and over a stone viaduct, under an overcast sky.",
    demoNegativePrompt:
      "grey clay untextured mesh, wireframe, low-poly blockout, floating train cars, derailed wheels, melted viaduct arches, blurry, watermark",
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
    kind: "video",
    catalogId: "restore",
    path: paths.restore,
    title: "Restore",
    description: "Restore old footage as a clean, colour video.",
    // The 1896 example prompt of the Restore page. Replace with the ltx.io prompt when the
    // recipe exists there.
    demoPrompt:
      "a steam locomotive arriving at a small rural railway station in 1896, passengers in period coats and hats waiting on the platform, wooden station buildings, natural colour",
    demoNegativePrompt:
      "shipping containers, logos, signage, lettering, ladders, modern vehicles, graffiti, modern buildings, blocky compression, banding, mosquito noise, ringing, blurry, soft, plastic, smeared detail, oversharpened halos, warped faces",
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
  {
    id: "ingredients",
    // The input is a list of images, not a clip. The backend joins them into one sheet.
    kind: "sheet",
    catalogId: "ingredients",
    path: paths.ingredients,
    title: "Ingredients",
    description: `Combine up to ${IC_LORA_SHEET_MAX_IMAGES} images into one video.`,
    // The catalog prompt template: panels first, then the action. It describes the packaged sheet.
    demoPrompt:
      "Reference sheet: Top Row Left (Character): A 3D animated orange fox with white fur on its snout, belly, and tail tip, large expressive brown eyes, and dark paws. It wears a small green neckerchief and is shown from multiple turnaround angles including front close-up, front full body, side, and back views. Top Row Right (Prop): A fishing rod with a cork handle, a spinning reel, a thin green fiberglass pole with metal eyelets, and a fishing line ending in a metal hook. Bottom Row Left (Setting): A serene outdoor park with a calm lake, a small wooden dock, lush green grass, leafy trees, and lily pads under a bright sunny sky with soft clouds. Bottom Row Right (Prop): A glossy bright red rain boot with pull loops at the top and a thick black grooved rubber sole, with water droplets scattered on its surface. Generated video: A bright, high-quality 3D animated scene set beside a serene, sunlit lake in a lush park. The camera frames a steady medium shot of the orange fox standing on the grassy bank near the water's edge, wearing its small green neckerchief and the bright red rain boot on one foot while holding the fishing rod with both paws. The fox stands facing the calm lake, holding the cork handle of the fishing rod, intently watching the water's surface as the fishing line dips near some lily pads. Suddenly, the fox's eyes widen in surprise as the rod bends sharply, indicating a heavy catch. With an eager and determined expression, the fox pulls the fishing rod upward with all its might, straining slightly. Breaking the surface of the water, it hoists up the glossy bright red rain boot attached to the hook, water droplets splashing around it. The fox blinks in sheer disbelief, staring at the boot dangling from the line. A playful, mischievous grin spreads across the fox's face, and it tosses its head back, opening its mouth wide in a burst of hearty, joyous laughter. The warm afternoon sunlight casts soft shadows across the grassy landscape, highlighting the shimmering ripples on the lake and the vibrant colors of the scene.",
    demoNegativePrompt:
      "split screen grid, reference sheet panels visible, inconsistent fox markings, missing green neckerchief, deformed fishing rod, blurry, jittery, distorted",
    placeholder: "Describe the scene that uses your images...",
    hiddenFields: [],
    cutout: false,
    // The Home defaults. The split is on purpose: the shared catalog default (1) is the GenSpace
    // value and stays. The Home form always sends this strength as `scale`. A direct API call
    // without `scale` gets the catalog value. 5 s at 24 fps is 121 frames, the trained length.
    homeDefaults: { loraStrength: 1.35, fps: 24, seconds: 5 },
    section: "marketing",
    listing: {
      // The ltx.io info dialog text.
      blurb:
        `Upload up to ${IC_LORA_SHEET_MAX_IMAGES} reference images (characters, props, locations). At generate time they are stitched into an Ingredients sheet that drives a short clip.`,
      categories: "Marketing",
      typeLabel: "Image to Video",
      huggingfaceUrl:
        "https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Ingredients",
    },
  },
  {
    id: "refine-details",
    kind: "video",
    catalogId: "refine-details",
    path: paths.refineDetails,
    title: "Refine Details",
    description: "Rebuild fine detail in a soft or upscaled clip.",
    // The first example prompt of the model card. It describes the rendering, not the subject.
    demoPrompt:
      "Locked-off wide shot of a clear mountain stream running over mossy granite boulders under a beech forest canopy, thousands of individual leaves, sunlight dappled evenly across the scene, water sparkling with fine ripples, ferns and lichen in the foreground, deep focus, tack sharp everywhere, nature documentary, photoreal, 8K, sharp photographic detail, crisp faces and clothing texture, natural grain, high resolution footage",
    demoNegativePrompt:
      "blurry, soft focus, shallow depth of field, bokeh, motion blur, low resolution, plastic, smeared detail, oversharpened halos, painting, fog, haze, text, watermark",
    placeholder: "Describe the sharpness, texture and light...",
    hiddenFields: [],
    cutout: false,
    // The clip is scaled up to the cell with Lanczos. The clip size and the sizes above it are offered.
    upscalesSource: "source-and-above",
    // The recipe offers 1440p and 4K without the `highResVideoToVideo` flag.
    highResWithoutFlag: true,
    tilingControls: true,
    tilingTooltips: {
      tileSize:
        "A larger tile means fewer tiles and a shorter render. On a 16:9 frame this starts at 720p. At 540p and below the frame fits in one tile, so every size is the same. At 720p, Large and Extra large both cover the frame.\n\nIt needs more VRAM, so a size that does not fit is disabled. The LoRA is trained on the standard size. A larger tile can change the detail.",
    },
    section: "post-production",
    // Revision 2: the packaged clip is the real reference, not the Restore placeholder.
    seed: icLoraSeed("refine-details", 2),
    listing: {
      blurb:
        "Add fine detail to a soft, compressed or upscaled clip. Texture, edges and grain come back, and the framing, the exposure and the colour stay the same. Raise the size up to 4K in one pass, for example from 1080p web video.",
      categories: "Post production",
      typeLabel: "Refine Details",
      huggingfaceUrl:
        "https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Refine-Details",
    },
  },
  {
    id: "motion-track",
    // The input is one image and a path drawn on it. The backend draws the control clip.
    kind: "track",
    catalogId: "motion-track",
    path: paths.motionTrack,
    title: "Motion Track Control",
    description:
      "Draw paths on a reference image to guide how objects and the camera move through the scene.",
    // The ltx.io seed. The backend wraps it with the camera lock, so it names the scene only.
    demoPrompt:
      "The conductor frog smoothly but forcefully moves his baton-holding arm and hand along the provided motion track, precisely following the path from point to point. The shoulder, elbow, wrist, hand, and baton move together naturally in a powerful, commanding conducting gesture. He maintains a firm grip on the baton throughout. Sharp, energetic movements with controlled acceleration and natural joint articulation. The frog's body reacts subtly to the intensity of the conducting. The orchestra remains visually stable. Audio: violent, aggressive orchestral music with pounding percussion",
    demoNegativePrompt:
      "camera movement, zoom, panning, reframing, cuts, visible colored tracking dots, detached baton, extra frog arms, blurry orchestra",
    placeholder: "Describe the scene — object motion follows the tracks and trajectories you set.",
    hiddenFields: [],
    cutout: false,
    // The Home defaults. 5 s at 24 fps, as the ltx.io form starts.
    homeDefaults: { loraStrength: 1, fps: 24, seconds: 5 },
    section: "vfx",
    listing: {
      blurb:
        "Draw paths on a reference image to guide how objects and the camera move through the scene.",
      categories: "VFX, Animation",
      typeLabel: "IC-LoRA",
      baseModel: "ltx-2.3",
      // Only the LTX-2.3 file is published. It also runs on LTX-2.5, as on ltx.io.
      huggingfaceUrl:
        "https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Motion-Track-Control",
    },
  },
  {
    id: "video-guide",
    kind: "video",
    catalogId: "video-guide",
    path: paths.videoGuide,
    title: "Video Guide",
    description:
      "Direct your generation with visual guide video like pose, depth, or canny, and a start frame.",
    // ltx.io has no seed prompt for Video Guide. This text describes the start frame and the
    // packaged pose clip, in the order of the catalog prompt template (scene, action, camera).
    demoPrompt:
      "A 3D animated lioness in a plush purple fur coat sits at a wooden table in a cozy cafe, with a purple textured wall, a framed picture, a tall palm plant and two iced drinks behind her. She rests her chin on her gloved hand and glances to the side with a gentle smile. Then she sits up, turns to the camera and speaks with lively, expressive hand gestures, her ears and eyes animated. The camera holds a steady medium shot at eye level.",
    demoNegativePrompt:
      "human face, flat 2D cartoon, extra paws, deformed gloves, floating cups, jerky head snap, camera shake, blurry fur, watermark",
    placeholder: "Describe your scene based on your image and reference video.",
    hiddenFields: [],
    cutout: false,
    // The form picks how the guide clip is read: pose, depth or Canny edges.
    controlMode: true,
    // An optional start frame. The form opens with the start frame.
    referenceImage: { label: "Start Frame", seed: VIDEO_GUIDE_IMAGE_SEED },
    section: "vfx",
    seed: icLoraSeed("video-guide", 2),
    listing: {
      blurb:
        "Direct your generation with visual guides. Combine pose, depth, and edge references in a single model to control movement and composition while keeping full creative freedom over style.",
      categories: "VFX, Animation",
      typeLabel: "IC-LoRA",
      baseModel: "ltx-2.3",
      // Only the LTX-2.3 file is published. It also runs on LTX-2.5, as on ltx.io.
      huggingfaceUrl:
        "https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control",
    },
  },
] as const satisfies readonly {
  readonly kind: IcLoraRecipeKind;
  readonly section: IcLoraRecipeSection;
  readonly [key: string]: unknown;
}[];

export type IcLoraRecipeId = (typeof IC_LORA_RECIPES)[number]["id"];
/** The recipes with an image list as input. The other recipes take one clip. */
export type IcLoraSheetRecipeId = IcLoraSheetRecipe["id"];
/** The recipes with an image and a path as input. */
export type IcLoraTrackRecipeId = IcLoraTrackRecipe["id"];
export type IcLoraVideoRecipeId = Exclude<IcLoraRecipeId, IcLoraSheetRecipeId | IcLoraTrackRecipeId>;
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

export type IcLoraSheetRecipe = Extract<
  (typeof IC_LORA_RECIPES)[number],
  { readonly kind: "sheet" }
>;

export type IcLoraTrackRecipe = Extract<
  (typeof IC_LORA_RECIPES)[number],
  { readonly kind: "track" }
>;

export type IcLoraRecipe = Exclude<
  (typeof IC_LORA_RECIPES)[number],
  { readonly kind: "sheet" | "track" }
> & {
  readonly referenceImage?: IcLoraReferenceImage;
  /**
   * The form has a Control mode field, and puts the start frame above the video.
   * The catalog entry declares the `control_type` control.
   */
  readonly controlMode?: boolean;
  /**
   * The recipe runs one tiled stage and upscales a small source. The resolution control
   * then offers every cell, not only the cells the source can fill. A row without it
   * keeps the source-size limit. `"source-and-above"` lists the Original cell and every
   * higher cell, 1440p and 2160p included. The backend then scales the clip up to
   * the cell (`upscales_to_cell` in ic_lora_recipes.py).
   */
  readonly upscalesSource?: boolean | "source-and-above";
  /**
   * The recipe offers 1440p and 4K without the `highResVideoToVideo` flag. Other recipes
   * need the flag for these cells. shared/ic-lora-recipes.json pins it, and a backend test
   * pins the same set.
   */
  readonly highResWithoutFlag?: boolean;
  /** The form has an Advanced section with the tile overlap and the chunk carry. */
  readonly tilingControls?: boolean;
  /**
   * Replaces the generic Advanced tooltip for this recipe. An omitted key keeps the generic text.
   */
  readonly tilingTooltips?: {
    readonly overlap?: string;
    readonly carry?: string;
    readonly chunkLength?: string;
    readonly tileSize?: string;
  };
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

export function isIcLoraVideoRecipeId(value: string | undefined): value is IcLoraVideoRecipeId {
  return IC_LORA_RECIPES.some((recipe) => recipe.id === value && recipe.kind === "video");
}

export function isIcLoraSheetRecipeId(value: string | undefined): value is IcLoraSheetRecipeId {
  return IC_LORA_RECIPES.some((recipe) => recipe.id === value && recipe.kind === "sheet");
}

export function isIcLoraTrackRecipeId(value: string | undefined): value is IcLoraTrackRecipeId {
  return IC_LORA_RECIPES.some((recipe) => recipe.id === value && recipe.kind === "track");
}

function findRecipeRow(id: IcLoraRecipeId): (typeof IC_LORA_RECIPES)[number] {
  const recipe = IC_LORA_RECIPES.find((candidate) => candidate.id === id);
  if (!recipe) {
    throw new Error(`Unknown IC-LoRA recipe: ${id}`);
  }
  return recipe;
}

export function getIcLoraRecipe(id: IcLoraVideoRecipeId): IcLoraRecipe {
  const recipe = findRecipeRow(id);
  if (recipe.kind !== "video") {
    throw new Error(`Not a video IC-LoRA recipe: ${id}`);
  }
  return recipe;
}

/** The catalog entry of any recipe. */
export function icLoraCatalogId(id: IcLoraRecipeId): string {
  return findRecipeRow(id).catalogId;
}

export function getIcLoraSheetRecipe(id: IcLoraSheetRecipeId): IcLoraSheetRecipe {
  const recipe = findRecipeRow(id);
  if (recipe.kind !== "sheet") {
    throw new Error(`Not an image-list IC-LoRA recipe: ${id}`);
  }
  return recipe;
}

export function getIcLoraTrackRecipe(id: IcLoraTrackRecipeId): IcLoraTrackRecipe {
  const recipe = findRecipeRow(id);
  if (recipe.kind !== "track") {
    throw new Error(`Not an image-and-path IC-LoRA recipe: ${id}`);
  }
  return recipe;
}
