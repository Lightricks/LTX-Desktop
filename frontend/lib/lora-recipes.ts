import { paths } from "../paths.ts";

const LTX_LICENSE_URL = "https://github.com/Lightricks/LTX-2/blob/main/LICENSE";
const LTX_HF_LICENSE_URL =
  "https://huggingface.co/Lightricks/LTX-2/blob/main/LICENSE";

export type LoraRecipeListing = {
  readonly blurb: string;
  readonly categories: string;
  readonly typeLabel: string;
  readonly authorName: string;
  readonly affiliation: "ltx" | "community";
  readonly baseModel: string;
  readonly licenseName: string;
  readonly licenseUrl: string;
  readonly huggingfaceUrl: string;
};

type LoraRecipeDefinitionShape = {
  readonly id: string;
  readonly catalogId: string;
  readonly mode: string;
  readonly path: string;
  readonly title: string;
  readonly description: string;
  readonly seedPrompt: string;
  readonly placeholder: string;
  readonly defaultStrength: number;
  readonly listed: boolean;
  readonly listing: LoraRecipeListing;
  readonly requiresEndFrame?: true;
};

/**
 * FE mirror of the backend LoRA-recipe table (backend
 * services/features/lora_recipes.py). The shared id/catalogId/mode/devices/listed
 * contract lives in `shared/lora-recipes.json`; both this table and the backend
 * table are asserted against that file (see lora-recipes.test.ts /
 * test_lora_recipe.py) so they can't drift. This table adds the FE-only copy
 * (title, description, placeholder, seed, path, default strength, listing).
 * `catalogId` is what the create body carries; the backend resolves the
 * installed path from it.
 *
 * Seed prompts match infinity/src/utils/loraSeedPrompts.ts LORA_PROMPTS_BY_ID
 * (or the gallery example prompt when that map has no entry). Placeholders match
 * LORA_PROMPT_PLACEHOLDER_BY_ID. Home titles match ltx.io gallery labels.
 * `listing` is the FeatureDetails payload so #213 does not need a 13-key chrome
 * map.
 */
export const LORA_RECIPES = [
  {
    id: "cozy-felt",
    catalogId: "cozy-felt-style",
    mode: "t2v",
    path: paths.cozyFelt,
    title: "Cozy Felt",
    description:
      "Turn a prompt into a handcrafted felt-cutout diorama with matching audio.",
    seedPrompt:
      "A big felt T-rex plays golf on a sunny felt golf course, comically gripping a tiny felt golf club in its little short arms with intense concentration. The T-rex is built from soft green layered felt with a big rounded body, a huge toothy stitched grin, powerful plush legs, a long thick plush tail, and comically tiny little arms, wearing a jaunty felt golf cap and a little argyle felt sweater vest. It lines up a tiny white felt golf ball on a felt tee, waggles the club with its stubby arms, then takes a big swing — twisting its whole body and swinging its long tail around for momentum since its arms can barely reach — and taps the felt ball, which rolls across the green as the T-rex watches hopefully with a big grin, shielding its eyes with a tiny arm. Around it, a cozy felt golf course with rolling stitched green fairways, a little red felt flag on the putting green, felt sand bunkers, leafy felt trees, and fluffy felt clouds in a bright blue felt sky. The camera holds in a gentle push-in on the golfing T-rex. Cheerful, funny, charming storybook mood, with a soft felt club tap, cheerful birdsong, and a gentle breeze.",
    placeholder:
      "Describe a scene to render as a handmade felt scene. Use words like embroidered felt details, felt cutout style, handcrafted felt diorama.",
    defaultStrength: 1,
    listed: true,
    listing: {
      blurb:
        "Render your scene as a handcrafted felt world, where characters, objects, and environments appear soft, stitched, and tactile, perfect for warm, storybook visuals.",
      categories: "Community",
      typeLabel: "Text to Video",
      authorName: "vrgamedevgirl84",
      affiliation: "community",
      baseModel: "ltx-2.3-fast",
      licenseName: "LTX",
      licenseUrl: LTX_LICENSE_URL,
      huggingfaceUrl:
        "https://huggingface.co/vrgamedevgirl84/LTX2.3_Cozy_Felt_Style_LoRa",
    },
  },
  {
    id: "claymation",
    catalogId: "claymation-style",
    mode: "t2v",
    path: paths.claymation,
    title: "Claymation",
    description: "Generate scenes with a tactile clay animation look.",
    seedPrompt:
      "claymation style, stop motion clay, a handcrafted clay diorama of two cute clay polar bears relaxing on a sunny tropical beach, sipping cocktails. The polar bears have soft white sculpted clay fur with visible fingerprints and hand-shaped textures, round friendly faces, and little black noses, lounging back in tiny clay beach chairs wearing cool clay sunglasses. Each holds a colorful clay cocktail in a little clay coconut cup with a tiny umbrella and a slice of clay fruit, sipping happily and giving content smiles. They sit on golden clay sand with a clay beach umbrella shading them, gentle clay ocean waves rolling in behind, a clay palm tree swaying, and soft clay clouds in a bright blue sky. All sculpted from clay with rounded stylized shapes and tactile handmade textures. Warm sunny cinematic lighting, shallow depth of field, cozy playful storybook mood, with gentle rolling waves, a soft breeze, and cheerful beach ambience.",
    placeholder:
      "Describe a scene to bring to life in tactile stop-motion clay — try a charming character in a cozy, miniature world.",
    defaultStrength: 1,
    listed: true,
    listing: {
      blurb:
        "Render your scene as a handcrafted claymation diorama, with sculpted clay, visible fingerprints, and miniature sets.",
      categories: "Community",
      typeLabel: "Text to Video",
      authorName: "vrgamedevgirl84",
      affiliation: "community",
      baseModel: "ltx-2.3-fast",
      licenseName: "LTX",
      licenseUrl: LTX_LICENSE_URL,
      huggingfaceUrl:
        "https://huggingface.co/vrgamedevgirl84/LTX_2.3_Clay_Mation_Style_LoRa",
    },
  },
  {
    id: "fantasy-painterly",
    catalogId: "fantasy-painterly-style",
    mode: "t2v",
    path: paths.fantasyPainterly,
    title: "Fantasy Painterly",
    description:
      "Render a fantasy scene with hand-painted brushwork and luminous lighting.",
    seedPrompt:
      "the opening frame is an extreme cinematic close-up of an elderly forest keeper, with only half of their face visible while the rest fades into soft negative space. Deep expressive wrinkles, kind eyes, silver hair, and a moss-green hooded cloak are bathed in warm golden light, creating an intimate portrait. After a quiet pause, they slowly pull the hood over their head, turn away from the camera, and begin walking deeper into the enchanted forest. As they disappear into the distance, tiny glowing forest spirits drift between the ancient trees, carried by a gentle breeze. The camera begins with an intimate close-up, then slowly pulls back and follows behind them with a graceful cinematic tracking shot, revealing more of the magical forest while maintaining soft parallax and shallow depth of field. Rich fantasy brushstrokes, visible canvas texture, luminous painterly lighting, atmospheric mist, and timeless storybook fantasy. Audio: natural enchanted forest ambience only.",
    placeholder:
      "Describe a fantasy scene to render with hand-painted brushwork. Use words like fantasy painterly style, cinematic fantasy, and oil painting.",
    defaultStrength: 1,
    listed: true,
    listing: {
      blurb:
        "Render a fantasy scene with hand-painted brushwork, oil-paint texture, and luminous magical lighting.",
      categories: "Community",
      typeLabel: "Text to Video",
      authorName: "vrgamedevgirl84",
      affiliation: "community",
      baseModel: "ltx-2.3-fast",
      licenseName: "LTX",
      licenseUrl: LTX_LICENSE_URL,
      huggingfaceUrl:
        "https://huggingface.co/vrgamedevgirl84/LTX_2.3_Fantasy_Painterly_Style_LoRa",
    },
  },
  {
    id: "paper-cut-out-style",
    catalogId: "paper-cutout-style",
    mode: "t2v",
    path: paths.paperCutOutStyle,
    title: "Paper Cut-Out",
    description: "Turn a prompt into a layered paper-cutout cinematic diorama.",
    seedPrompt:
      "A beautifully composed wide cinematic establishing shot of a peaceful desert at golden hour, with towering saguaro cacti, blooming barrel cacti, scattered rocks, warm textured sand, layered mesas, and a glowing sky fading from amber to turquoise. After a brief moment, a curious desert tortoise enters from the left, slowly walks between the cacti, pauses to lift its head and look directly at the viewer, then continues walking and exits the frame on the right. The camera starts as a still wide shot before smoothly transitioning into a slow eye-level tracking shot with gentle parallax. Warm golden-hour lighting creates a peaceful, nostalgic cinematic atmosphere. Audio: Only authentic desert ambience with soft wind, dry grass rustling, gentle tortoise footsteps, occasional pebbles shifting, and natural silence.",
    placeholder:
      "Describe a scene to render as a handmade paper cut-out diorama. Use words like paper cutout style, handcrafted paper and layered paper scene.",
    defaultStrength: 1,
    listed: true,
    listing: {
      blurb:
        "Turn a prompt into a layered paper-cutout cinematic diorama, with stacked paper, clean cut edges, and tape seams.",
      categories: "Community",
      typeLabel: "Text to Video",
      authorName: "vrgamedevgirl84",
      affiliation: "community",
      baseModel: "ltx-2.3-fast",
      licenseName: "LTX",
      licenseUrl: LTX_LICENSE_URL,
      huggingfaceUrl:
        "https://huggingface.co/vrgamedevgirl84/LTX_2.3_Paper_Cut_Out_Style_LoRa",
    },
  },
  {
    id: "cinemagraph",
    catalogId: "cinemagraph-motion",
    mode: "i2v",
    path: paths.cinemagraph,
    title: "Cinemagraph",
    description:
      "Freeze a still except for one looping detail — a locked-off living photograph.",
    seedPrompt:
      "Mountains, and foreground remain completely frozen, only the night sky moves, the Milky Way and stars rotate across the sky like Earth-rotation time-lapse, everything below the horizon stays still, colors stay the same",
    placeholder:
      "Describe a static scene and the single element that should move. Emphasize what stays frozen and what animates in a seamless loop.",
    defaultStrength: 2,
    listed: true,
    listing: {
      blurb:
        "Freeze a still except for one looping detail — a locked-off living photograph.",
      categories: "Camera",
      typeLabel: "Image to Video",
      authorName: "LTX Team",
      affiliation: "ltx",
      baseModel: "ltx-2.3-fast",
      licenseName: "LTX-2 Community License",
      licenseUrl: LTX_HF_LICENSE_URL,
      huggingfaceUrl:
        "https://huggingface.co/Lightricks/LTX-2.3-22b-LoRA-Cinemagraph",
    },
  },
  {
    id: "jib-up",
    catalogId: "jib-up",
    mode: "i2v",
    path: paths.jibUp,
    title: "Jib Up",
    description:
      "Turn any scene into an epic reveal with a classic crane sweep upward.",
    seedPrompt:
      "The camera slowly rises, revealing the view. As it moves, the character gently lifts their gaze. Soft, peaceful sounds of nature.",
    placeholder:
      "Describe the scene and the destination of the camera movement. Try to explain what the camera will focus on as it moves upward.",
    defaultStrength: 1,
    listed: true,
    listing: {
      blurb:
        "Turn any scene into an epic reveal with a classic crane sweep upward.",
      categories: "Camera",
      typeLabel: "Image to Video",
      authorName: "LTX Team",
      affiliation: "ltx",
      baseModel: "ltx-2.3-fast",
      licenseName: "LTX-2 Community License",
      licenseUrl: LTX_HF_LICENSE_URL,
      huggingfaceUrl:
        "https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Jib-Up",
    },
  },
  {
    id: "jib-down",
    catalogId: "jib-down",
    mode: "i2v",
    path: paths.jibDown,
    title: "Jib Down",
    description:
      "Turn any scene into an epic reveal with a classic crane sweep downward.",
    seedPrompt:
      "The camera slowly moves down, revealing the character's long hiking pants and hiking shoes. The character stays still. Soft sounds of nature in the background.",
    placeholder:
      "Describe the scene and the destination of the camera movement. Try to explain what the camera will focus on as it moves downward.",
    defaultStrength: 1,
    listed: true,
    listing: {
      blurb:
        "Turn any scene into an epic reveal with a classic crane sweep downward.",
      categories: "Camera",
      typeLabel: "Image to Video",
      authorName: "LTX Team",
      affiliation: "ltx",
      baseModel: "ltx-2.3-fast",
      licenseName: "LTX-2 Community License",
      licenseUrl: LTX_HF_LICENSE_URL,
      huggingfaceUrl:
        "https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Jib-Down",
    },
  },
  {
    id: "dolly-in",
    catalogId: "dolly-in",
    mode: "i2v",
    path: paths.dollyIn,
    title: "Dolly In",
    description:
      "Smooth cinematic push toward your subject. No rig, instant tension.",
    seedPrompt:
      "The camera slowly zooms in on the character's face. As the camera zooms in, she looks directly into the camera. A soft wind moves through her hair, with gentle wind sounds in the background.",
    placeholder:
      "Describe a scene and the destination of the dolly-in — what the camera arrives at as it pushes forward.",
    defaultStrength: 1,
    listed: true,
    listing: {
      blurb:
        "Push the camera toward your subject. Smooth forward motion builds focus and draws the viewer into the scene.",
      categories: "Camera",
      typeLabel: "Image to Video",
      authorName: "LTX Team",
      affiliation: "ltx",
      baseModel: "ltx-2.3-fast",
      licenseName: "LTX-2 Community License",
      licenseUrl: LTX_HF_LICENSE_URL,
      huggingfaceUrl:
        "https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-In",
    },
  },
  {
    id: "dolly-out",
    catalogId: "dolly-out",
    mode: "i2v",
    path: paths.dollyOut,
    title: "Dolly Out",
    description: "Smooth cinematic push out from your subject.",
    seedPrompt:
      "The camera slowly moves out, revealing the character in the black suit stands still with her hands loosely clasped in front of her waist. Soft wind moves through her hair, with gentle wind sounds in the background.",
    placeholder:
      "Describe the scene and the destination of the camera movement. Try to explain what the camera will focus on as it moves backward.",
    defaultStrength: 1,
    listed: true,
    listing: {
      blurb:
        "Pull the camera away from your subject. Smooth backward motion reveals the space around them.",
      categories: "Camera",
      typeLabel: "Image to Video",
      authorName: "LTX Team",
      affiliation: "ltx",
      baseModel: "ltx-2.3-fast",
      licenseName: "LTX-2 Community License",
      licenseUrl: LTX_HF_LICENSE_URL,
      huggingfaceUrl:
        "https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Out",
    },
  },
  {
    id: "fpv-motion",
    catalogId: "fpv-motion",
    mode: "t2v",
    path: paths.fpvMotion,
    title: "Drone Flythrough",
    description:
      "Add smooth, low-speed drone-style camera motion to generated video.",
    seedPrompt:
      "A cinematic FPV drone shot flying smoothly through a cozy wooden cabin in the woods and out into a stunning natural landscape, ending with a dramatic upward reveal. The drone glides gently and steadily forward toward the cabin nestled among tall pine trees, drifting smoothly through an open doorway into the warm rustic interior — passing a stone fireplace, wooden furniture, and soft light through the windows. It floats steadily across the room and out through a large open window on the far side, emerging into the open air to reveal a majestic waterfall cascading down a moss-covered cliff into a calm crystal-clear lake below, surrounded by lush green forest. The drone glides smoothly out over the glistening lake, gentle mist rising from the waterfall in the golden sunlight. Then the camera rapidly ascends, shooting quickly upward in a fast rising motion while remaining angled down and focused on the scene beneath — the lake, waterfall, and forest shrinking below as the camera climbs, revealing the full sweeping landscape and the surrounding mountains and treetops from a soaring high vantage point. The early flythrough moves with smooth, unhurried motion, building to the fast dramatic upward climb at the end. Warm natural lighting, soft shafts through the trees, light atmospheric haze, rich green and gold tones. Immersive first-person FPV drone perspective, wide-angle lens, cinematic, photorealistic, 4K. Gentle wind, a soft drone hum rising as it climbs, the roar of the waterfall, flowing water, and birdsong.",
    placeholder:
      "Describe a scene and a flight path - the camera glides through it in smooth, low-speed drone motion, e.g. 'through a canyon and out over the sea.'",
    defaultStrength: 1,
    listed: true,
    listing: {
      blurb:
        "Add smooth, low-speed drone-style camera motion so the shot glides through the scene.",
      categories: "Community",
      typeLabel: "Text to Video",
      authorName: "chsengni",
      affiliation: "community",
      baseModel: "ltx-2.3-fast",
      licenseName: "Apache-2.0",
      licenseUrl: "https://huggingface.co/chsengni/ltx2.3-fpv-motion",
      huggingfaceUrl: "https://huggingface.co/chsengni/ltx2.3-fpv-motion",
    },
  },
  {
    id: "openwheel-t-cam",
    catalogId: "openwheel-tcam-style",
    mode: "t2v",
    path: paths.openwheelTCam,
    title: "OpenWheel T-Cam",
    description:
      "Generate open-wheel motorsport video from an in-car T-cam cockpit perspective.",
    seedPrompt:
      "T-cam onboard view looking over the silver and cyan central halo structure of a red 16LV6HybridF1 car racing in a tight pack on a wet track in heavy rain. The red car features cyan and white sponsor decals. Speeding down a long straight through clouds of spray, it trails closely behind a yellow 16LV6HybridF1 car just ahead, while a blue 16LV6HybridF1 car runs alongside on the left and a green car looms in the mirrors behind. The camera car pulls out of the spray to draw alongside the yellow car, the two racing wheel to wheel down the straight, before both brake hard for an approaching corner. The yellow car defends the inside line while the camera car sweeps around the outside through the turn, spray flying, then both accelerate hard out of the corner side by side, fighting for position. A gold and black driver's helmet is visible at the bottom of the frame.",
    placeholder:
      "Describe an open-wheel race from the cockpit. Focus on the car, track conditions, and racing action.",
    defaultStrength: 1,
    listed: true,
    listing: {
      blurb:
        "Generate open-wheel motorsport video from an in-car T-cam cockpit perspective.",
      categories: "Community",
      typeLabel: "Text to Video",
      authorName: "mxturbo",
      affiliation: "community",
      baseModel: "ltx-2.3-fast",
      licenseName: "Apache-2.0",
      licenseUrl:
        "https://huggingface.co/mxturbo/Openwheel-motorsports-Cockpit-T-Cam-LTX2.3",
      huggingfaceUrl:
        "https://huggingface.co/mxturbo/Openwheel-motorsports-Cockpit-T-Cam-LTX2.3",
    },
  },
  {
    id: "transition",
    catalogId: "transition",
    mode: "i2v",
    path: paths.transition,
    title: "Transition",
    description:
      "Morph between two frames with a seamless, continuous camera move.",
    seedPrompt:
      "A high aerial shot glides low over a churning turquoise ocean as a massive wave rises and curls, its crest breaking into white foaming spray against a hazy, overcast sky. As the wave crashes forward, its curling white foam morphs seamlessly into a snow-capped mountain peak, the churning water resolving into jagged rock and packed snow, and the sea mist thinning into drifting alpine cloud below the summit. The color grade shifts gradually from cool teal-blue ocean light to the warm golden-hour glow of sunrise catching the mountain's ridges. The camera holds a steady high-angle drift throughout, tracking the continuous motion from crashing water to towering peak.",
    placeholder:
      "Describe how the start frame transforms into the end frame — shot type, the change over time, then lighting and atmosphere.",
    defaultStrength: 1,
    listed: true,
    requiresEndFrame: true,
    listing: {
      blurb:
        "Morph between two frames with a seamless, continuous camera move.",
      categories: "Community",
      typeLabel: "Image to Video",
      authorName: "joyfox",
      affiliation: "community",
      baseModel: "ltx-2.3-fast",
      licenseName: "Apache-2.0",
      licenseUrl: "https://huggingface.co/joyfox/LTX-2.3-Transition-LORA",
      huggingfaceUrl: "https://huggingface.co/joyfox/LTX-2.3-Transition-LORA",
    },
  },
  {
    id: "vbvr",
    catalogId: "vbvr",
    mode: "i2v",
    path: paths.vbvr,
    title: "VBVR",
    description:
      "Improve multi-object motion, physical interactions, and prompt accuracy in generated video.",
    seedPrompt:
      "The soccer player sprints across the field with powerful strides. The camera tracks smoothly, keeping him centered in the frame. Fast footsteps, heavy breathing, and distant crowd cheers fill the scene.",
    placeholder:
      "Describe what happens over time — object motion, collisions, and how the camera stays framed.",
    defaultStrength: 1,
    listed: true,
    listing: {
      blurb:
        "Improve multi-object motion, physical interactions, and prompt accuracy in generated video.",
      categories: "Community",
      typeLabel: "Image to Video",
      authorName: "LiconStudio",
      affiliation: "community",
      baseModel: "ltx-2.3-fast",
      licenseName: "LTX-2 Community License",
      licenseUrl: LTX_LICENSE_URL,
      huggingfaceUrl: "https://huggingface.co/LiconStudio/Ltx2.3-VBVR-lora-I2V",
    },
  },
] as const satisfies readonly LoraRecipeDefinitionShape[];

/**
 * Home-feature rows for the *listed* recipes only, derived from LORA_RECIPES so
 * title / description / path have a single source of truth. Because LORA_RECIPES
 * is `as const`, each `id` stays a string literal here, so `HomeFeatureId` remains
 * a closed union when these are spread into HOME_FEATURES.
 */
export const LORA_RECIPE_HOME_FEATURES = LORA_RECIPES.filter(
  (recipe) => recipe.listed,
).map((recipe) => ({
  id: recipe.id,
  path: recipe.path,
  title: recipe.title,
  description: recipe.description,
  fetcherTool: recipe.id,
}));

export type LoraRecipeId = (typeof LORA_RECIPES)[number]["id"];
export type LoraRecipeDefinition = (typeof LORA_RECIPES)[number];
export type LoraI2vRecipeId = Extract<LoraRecipeDefinition, { mode: "i2v" }>["id"];
export type LoraEndFrameRecipeId = Extract<
  LoraRecipeDefinition,
  { requiresEndFrame: true }
>["id"];

export function isLoraRecipeId(value: string | undefined): value is LoraRecipeId {
  return value !== undefined && LORA_RECIPES.some((recipe) => recipe.id === value);
}

export function getLoraRecipe(id: LoraRecipeId): LoraRecipeDefinition {
  const recipe = LORA_RECIPES.find((candidate) => candidate.id === id);
  if (!recipe) {
    throw new Error(`Unknown LoRA recipe: ${id}`);
  }
  return recipe;
}

export function recipeHasEndFrame(
  recipe: LoraRecipeDefinition,
): recipe is Extract<LoraRecipeDefinition, { requiresEndFrame: true }> {
  return "requiresEndFrame" in recipe && recipe.requiresEndFrame === true;
}
