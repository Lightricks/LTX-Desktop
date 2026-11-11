import {
  resolveVariantChoice,
  unsupportedModelReason,
  variantChoiceOptions,
  variantChoiceWarning,
} from "../../../lib/catalogVariantChoice.ts";
import {
  icLoraImageLabel,
  isIcLoraFieldHidden,
  type IcLoraRecipe,
} from "../../../../lib/ic-lora-recipes.ts";
import {
  IC_LORA_CONTROL_COPY,
  IC_LORA_CONTROL_TYPES,
  type IcLoraControlType,
} from "../../../lib/icLoraControlType.ts";
import type {
  AdvancedFormField,
  FeatureFormField,
  FeatureFormModel,
  FieldOption,
} from "../types.ts";
import { icLoraFpsOptions, icLoraMaxInputSeconds, icLoraOutputFps } from "./icLoraFps.ts";
import { icLoraFitsModel, icLoraMeterModel } from "./icLoraMemory.ts";
import { icLoraResolutionOptions } from "./icLoraResolution.ts";
import {
  AUTO_NUMBER,
  AUTO_TILE,
  CHUNK_CARRY_TOOLTIP,
  CHUNK_LENGTH_TOOLTIP,
  TILE_SIZE_TOOLTIP,
  TILING_OVERLAP_TOOLTIP,
} from "./icLoraTilingControls.ts";
import {
  carryOptions,
  overlapOptions,
  tileSizeOptions,
  windowFrameOptions,
} from "./icLoraTileOptions.ts";
import type {
  IcLoraRecipeContext,
  IcLoraRecipeSchema,
  IcLoraRecipeValues,
} from "./icLoraRecipe.ts";
import { negativePromptField } from "./negativePrompt.ts";
import { capabilityModelOptions } from "./videoFeature.ts";

export const IC_LORA_AUDIO_MODES = ["source", "generated", "off"] as const;

export type IcLoraRecipeAudioMode = (typeof IC_LORA_AUDIO_MODES)[number];

const IC_LORA_CONTROL_OPTIONS: FieldOption<IcLoraControlType>[] = IC_LORA_CONTROL_TYPES.map(
  (value) => ({ value, ...IC_LORA_CONTROL_COPY[value] }),
);

export const STRENGTH_MIN = 0;
export const STRENGTH_MAX = 2;
export const STRENGTH_STEP = 0.05;

export const IC_LORA_AUDIO_OPTIONS: FieldOption<IcLoraRecipeAudioMode>[] = [
  { value: "off", label: "Off" },
  { value: "source", label: "Source" },
  { value: "generated", label: "Generated" },
];

export function icLoraModelOfferingIds(context: Pick<IcLoraRecipeContext, "specs">): string[] {
  return capabilityModelOptions(context.specs, "ic_lora").map((option) => option.value);
}

/** The Model field, and the Variant field when the catalog has variants. */
export function icLoraModelFields(
  values: Pick<IcLoraRecipeValues, "model" | "variant">,
  context: Pick<IcLoraRecipeContext, "specs" | "catalogVariants" | "supportedModels">,
): FeatureFormField<Pick<IcLoraRecipeSchema, "model" | "variant">>[] {
  const choice = resolveVariantChoice(
    context.catalogVariants ?? [],
    values.variant,
    values.model,
    icLoraModelOfferingIds(context),
  );
  const variantOptions = variantChoiceOptions(choice, values.model, "IC-LoRA");
  return [
    {
      kind: "options",
      id: "model",
      label: "Model",
      dataKey: "model",
      forcePicker: true,
      options: capabilityModelOptions(context.specs, "ic_lora").map((option) => {
        const disabledReason = unsupportedModelReason(
          context.supportedModels,
          option.value,
          "IC-LoRA",
        );
        if (disabledReason != null) return { ...option, disabledReason };
        const warning = variantChoiceWarning(choice, option.value, "IC-LoRA");
        return warning == null ? option : { ...option, warning };
      }),
    },
    ...shownUnless(variantOptions == null, {
      kind: "options" as const,
      id: "variant" as const,
      label: "Variant",
      dataKey: "variant" as const,
      forcePicker: true,
      options: variantOptions ?? [],
    }),
  ];
}

/** One field, or none when the recipe fixes that control to its catalog default. */
function shownUnless<TField>(hidden: boolean, field: TField): TField[] {
  return hidden ? [] : [field];
}

/** The recipe text when it sets one, otherwise the generic Advanced tooltip. */
function tilingTooltip(
  recipe: IcLoraRecipe,
  key: keyof NonNullable<IcLoraRecipe["tilingTooltips"]>,
  generic: string,
): string {
  return recipe.tilingTooltips?.[key] ?? generic;
}

/**
 * The Advanced section of a recipe with `tilingControls`. Empty until the catalog says the window
 * is adaptive. Every control starts on Auto, which the form stores as null (the tile size stores
 * "auto"), so the control shows its `defaultValue`.
 */
function tilingFields(
  recipe: IcLoraRecipe,
  context: IcLoraRecipeContext,
): AdvancedFormField<IcLoraRecipeSchema>[] {
  if (recipe.tilingControls !== true || context.tilingAdaptive !== true) return [];
  return [
    {
      kind: "options",
      id: "tilingOverlap",
      label: "Tile overlap",
      dataKey: "tilingOverlap",
      tooltip: tilingTooltip(recipe, "overlap", TILING_OVERLAP_TOOLTIP),
      defaultValue: AUTO_NUMBER,
      forcePicker: true,
      options: overlapOptions(context),
    },
    {
      kind: "options",
      id: "carryFrames",
      label: "Chunk carry",
      dataKey: "carryFrames",
      tooltip: tilingTooltip(recipe, "carry", CHUNK_CARRY_TOOLTIP),
      defaultValue: AUTO_NUMBER,
      forcePicker: true,
      options: carryOptions(context),
    },
    {
      kind: "options",
      id: "maxWindowFrames",
      label: "Maximum chunk length",
      dataKey: "maxWindowFrames",
      tooltip: tilingTooltip(recipe, "chunkLength", CHUNK_LENGTH_TOOLTIP),
      defaultValue: AUTO_NUMBER,
      forcePicker: true,
      options: windowFrameOptions(context),
    },
    {
      kind: "options",
      id: "tilingSize",
      label: "Tile size",
      dataKey: "tilingSize",
      tooltip: tilingTooltip(recipe, "tileSize", TILE_SIZE_TOOLTIP),
      defaultValue: AUTO_TILE,
      forcePicker: true,
      options: tileSizeOptions(context),
    },
  ];
}

export function icLoraRecipeForm(
  recipe: IcLoraRecipe,
  values: IcLoraRecipeValues,
  context: IcLoraRecipeContext,
): FeatureFormModel<IcLoraRecipeSchema> {
  const fpsOptions = context.videoFps == null ? [] : icLoraFpsOptions(context.videoFps);
  const advice = context.recipeMemory ?? null;
  // No clip yet: the job's size and length are unknown, so the form says which cells fit.
  const hasClip =
    values.video != null && context.videoWidth != null && context.videoHeight != null;
  const videoField = {
    kind: "video-asset" as const,
    id: "video" as const,
    label: "Reference Video",
    dataKey: "video" as const,
    maxDurationSeconds: icLoraMaxInputSeconds(icLoraOutputFps(values.fps, context.videoFps)),
  };
  const imageFields: FeatureFormField<IcLoraRecipeSchema>[] =
    recipe.referenceImage == null
      ? []
      : [
          {
            kind: "image-asset",
            id: "image",
            // ltx.io labels the start frame of a control-mode recipe without "(optional)".
            label: recipe.controlMode
              ? recipe.referenceImage.label
              : icLoraImageLabel(recipe.referenceImage, context.referenceImageRequired),
            dataKey: "image",
          },
        ];
  return {
    advancedFields: tilingFields(recipe, context),
    fields: [
      // The start frame comes first on ltx.io's control-mode form, then the guide clip.
      ...(recipe.controlMode ? [...imageFields, videoField] : [videoField, ...imageFields]),
      ...shownUnless(isIcLoraFieldHidden(recipe, "prompt"), {
        kind: "textarea" as const,
        id: "prompt" as const,
        label: "Prompt",
        dataKey: "prompt" as const,
        placeholder: recipe.placeholder,
      }),
      ...shownUnless(
        isIcLoraFieldHidden(recipe, "prompt"),
        negativePromptField<IcLoraRecipeSchema>(),
      ),
      {
        kind: "options",
        id: "resolution",
        label: "Resolution",
        dataKey: "resolution",
        forcePicker: true,
        options: icLoraResolutionOptions(context, recipe.upscalesSource ?? false),
      },
      ...shownUnless(fpsOptions.length <= 1, {
        kind: "options" as const,
        id: "fps" as const,
        label: "FPS",
        dataKey: "fps" as const,
        // Original is stored as null, so the control must show it, not the first rate.
        defaultValue: fpsOptions.find((option) => option.original)?.value,
        forcePicker: true,
        options: fpsOptions.map((option) => ({
          value: option.value,
          label: option.label,
        })),
      }),
      ...shownUnless(isIcLoraFieldHidden(recipe, "audioMode"), {
        kind: "options" as const,
        id: "audioMode" as const,
        label: "Audio",
        dataKey: "audioMode" as const,
        options: IC_LORA_AUDIO_OPTIONS,
      }),
      ...icLoraModelFields(values, context),
      ...shownUnless(isIcLoraFieldHidden(recipe, "loraStrength"), {
        kind: "slider" as const,
        id: "loraStrength" as const,
        label: "Style strength",
        dataKey: "loraStrength" as const,
        min: STRENGTH_MIN,
        max: STRENGTH_MAX,
        step: STRENGTH_STEP,
      }),
      ...shownUnless(!recipe.controlMode, {
        kind: "options" as const,
        id: "controlType" as const,
        label: "Select control mode",
        dataKey: "controlType" as const,
        forcePicker: true,
        options: IC_LORA_CONTROL_OPTIONS.map((option) => ({
          ...option,
          thumbnail: context.controlThumbnails?.[option.value],
        })),
      }),
    ],
    memory: advice != null && hasClip ? icLoraMeterModel(advice) : null,
    memoryFits:
      advice != null && !hasClip
        ? icLoraFitsModel(advice, values.resolution, icLoraOutputFps(values.fps, context.videoFps))
        : null,
  };
}
