import type { components } from "../../../../generated/backend-openapi";
import {
  restoreOfferingId,
  withDefaultOffering,
  type OfferingId,
} from "../../../../lib/video-generation-model-specs.ts";
import { roundDurationSeconds } from "../../../lib/roundDuration.ts";
import { readGenerationParams } from "../../../lib/resultsFeedModel.ts";
import {
  applyFeatureFieldChange,
  type AssetRef,
  type FeatureDefinition,
  type FeatureFieldChange,
  type FeatureFormModel,
  type FeatureValues,
  type FieldOption,
  type ValidationIssue,
} from "../types.ts";
import {
  capabilityModelOptions,
  editResolutionFieldOptions,
  editResolutionSize,
  editResolutionKeyFromParams,
  type EditResolutionKey,
  hasAdvertisedCapability,
  normalizeEditResolution,
  readPrompt,
  readVideoGenerationInput,
  resolveCapableModel,
  type VideoInputContext,
  videoLengthUnknownIssue,
  videoRequiredIssue,
} from "./videoFeature.ts";

type CreateExtendRequest = components["schemas"]["CreateExtendRequest"];
type ExtendMode = components["schemas"]["ExtendParams"]["mode"];

export const EXTEND_DURATION_PRESETS = [4, 6, 8, 10, 12] as const;
export type ExtendDuration = (typeof EXTEND_DURATION_PRESETS)[number];

export type ExtendSchema = {
  video: { kind: "video-asset"; value: AssetRef | null };
  prompt: { kind: "textarea" };
  model: { kind: "options"; value: OfferingId };
  resolution: { kind: "options"; value: EditResolutionKey };
  mode: { kind: "options"; value: ExtendMode };
  duration: { kind: "options"; value: ExtendDuration };
};

export type ExtendValues = FeatureValues<ExtendSchema>;

export type ExtendContext = VideoInputContext;

export const EXTEND_EXAMPLE_PROMPT = "The camera pulls back as the scene continues";

const MODE_OPTIONS: FieldOption<ExtendMode>[] = [
  { value: "start", label: "Before the clip" },
  { value: "end", label: "After the clip" },
];

const DURATION_OPTIONS: FieldOption<ExtendDuration>[] = EXTEND_DURATION_PRESETS.map(
  (seconds) => ({ value: seconds, label: `${seconds}s` }),
);

export const EXTEND_DEFAULTS: ExtendValues = {
  video: null,
  prompt: EXTEND_EXAMPLE_PROMPT,
  model: "ltx-2.5-fast",
  resolution: "original",
  mode: "end",
  duration: 4,
};

function isExtendMode(value: unknown): value is ExtendMode {
  return value === "start" || value === "end";
}

function isExtendDuration(value: unknown): value is ExtendDuration {
  return EXTEND_DURATION_PRESETS.some((preset) => preset === value);
}

export function extendModelOptions(
  specs: ExtendContext["specs"],
): FieldOption<OfferingId>[] {
  return capabilityModelOptions(specs, "extend");
}

export function hasAdvertisedExtend(specs: ExtendContext["specs"]): boolean {
  return hasAdvertisedCapability(specs, "extend");
}

function resolveExtendModel(
  model: unknown,
  context: ExtendContext,
): OfferingId {
  return resolveCapableModel(model, context.specs, "extend", EXTEND_DEFAULTS.model);
}

export function fromGeneration(
  spec: unknown,
  context: ExtendContext = {
    specs: null,
    seed: null,
    videoDurationSeconds: null,
    videoDurationPending: false,
    videoWidth: null,
    videoHeight: null,
    videoFps: null,
  },
): ExtendValues | null {
  const params = readGenerationParams(spec);
  const inputs = readVideoGenerationInput(spec);
  if (!params || !inputs) return null;
  return {
    video: inputs.video,
    prompt: readPrompt(spec),
    model: restoreOfferingId(params.model, context.specs) ?? EXTEND_DEFAULTS.model,
    resolution: editResolutionKeyFromParams(params),
    mode: isExtendMode(params.mode) ? params.mode : EXTEND_DEFAULTS.mode,
    duration: isExtendDuration(params.duration)
      ? params.duration
      : EXTEND_DEFAULTS.duration,
  };
}

function videoLengthIssue(
  values: ExtendValues,
  context: ExtendContext,
): ValidationIssue | null {
  if (values.video == null) return null;
  if (context.videoDurationPending) return null;
  if (context.videoDurationSeconds == null) {
    return videoLengthUnknownIssue();
  }
  return null;
}

export function validateExtend(
  values: ExtendValues,
  context: ExtendContext,
): ValidationIssue[] {
  const issues: ValidationIssue[] = [];
  if (values.video == null) {
    issues.push(videoRequiredIssue());
  } else {
    const lengthIssue = videoLengthIssue(values, context);
    if (lengthIssue) issues.push(lengthIssue);
  }
  return issues;
}

export function toCreateBody(
  values: ExtendValues,
  context: ExtendContext = {
    specs: null,
    seed: null,
    videoDurationSeconds: null,
    videoDurationPending: false,
    videoWidth: null,
    videoHeight: null,
    videoFps: null,
  },
): CreateExtendRequest {
  if (values.video == null) {
    throw new Error("Video is required.");
  }
  return {
    contract_version: 1,
    inputs: { video: { assetId: values.video.assetId } },
    params: {
      prompt: values.prompt.trim(),
      model: values.model,
      duration: roundDurationSeconds(values.duration),
      mode: values.mode,
      promptProvenance: "typed",
      ...editResolutionParam(values, context),
    },
  };
}

function editResolutionParam(
  values: ExtendValues,
  context: ExtendContext,
): { resolution: { width: number; height: number } } | Record<string, never> {
  const resolution = editResolutionSize(
    values.resolution,
    context.videoWidth,
    context.videoHeight,
  );
  return resolution ? { resolution } : {};
}

export function buildExtendForm(
  _values: ExtendValues,
  context: ExtendContext,
): FeatureFormModel<ExtendSchema> {
  const resolutionOptions = editResolutionFieldOptions(
    context.videoWidth,
    context.videoHeight,
    context.specs?.low_performance_machine,
  );
  return {
    fields: [
      {
        kind: "video-asset",
        id: "video",
        label: "Video",
        dataKey: "video",
      },
      {
        kind: "textarea",
        id: "prompt",
        label: "Prompt",
        dataKey: "prompt",
        placeholder: "Describe what should happen in the extension...",
      },
      {
        kind: "options",
        id: "duration",
        label: "Duration",
        dataKey: "duration",
        options: DURATION_OPTIONS,
      },
      {
        kind: "options",
        id: "mode",
        label: "Add frames",
        dataKey: "mode",
        options: MODE_OPTIONS,
      },
      ...(resolutionOptions.length > 0
        ? [
            {
              kind: "options" as const,
              id: "resolution",
              label: "Resolution",
              dataKey: "resolution" as const,
              forcePicker: true,
              options: resolutionOptions,
            },
          ]
        : []),
      {
        kind: "options",
        id: "model",
        label: "Model",
        dataKey: "model",
        forcePicker: true,
        options: extendModelOptions(context.specs),
      },
    ],
  };
}

export function applyExtendChange(
  values: ExtendValues,
  change: FeatureFieldChange<ExtendSchema>,
): ExtendValues {
  return applyFeatureFieldChange(values, change);
}

export function normalizeExtend(
  values: ExtendValues,
  context: ExtendContext,
): ExtendValues {
  const model = resolveExtendModel(values.model, context);
  const resolution = normalizeEditResolution(
    values.resolution,
    context.videoWidth,
    context.videoHeight,
  );
  if (model === values.model && resolution === values.resolution) return values;
  return { ...values, model, resolution };
}

export function seedExtendValues(context: ExtendContext): ExtendValues {
  return {
    ...withDefaultOffering(context.specs, { ...EXTEND_DEFAULTS }),
    video: context.seed,
  };
}

export function resetExtendValues(context: ExtendContext): ExtendValues {
  return withDefaultOffering(context.specs, { ...EXTEND_DEFAULTS });
}

export const extendDefinition: FeatureDefinition<
  ExtendSchema,
  CreateExtendRequest,
  ExtendContext,
  "extend"
> = {
  id: "extend",
  title: "Extend",
  defaults: EXTEND_DEFAULTS,
  initialValues: seedExtendValues,
  resetValues: resetExtendValues,
  form: buildExtendForm,
  applyChange: applyExtendChange,
  normalize: normalizeExtend,
  fromGeneration,
  validate: validateExtend,
  toCreateBody,
  isReady: (values, context) => {
    if (values.video == null) return false;
    if (context.videoDurationPending) return false;
    return videoLengthIssue(values, context) == null;
  },
  isUnavailable: (_values, context) => !hasAdvertisedExtend(context.specs),
};
