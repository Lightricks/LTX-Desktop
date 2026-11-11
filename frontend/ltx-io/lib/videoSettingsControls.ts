import {
  isConfirmableBudgetTier,
  riskOptionFields,
} from "../../lib/budget-risk.ts";
import { askOptionMemory, optionMemory } from "../../lib/memory-meter.ts";
import {
  meterCell,
  passShapeChoice,
  selectedFastRisk,
  type ForcedPassShape,
  type ImageSize,
  type PassShapeSelection,
} from "../../lib/pass-shape.ts";
import {
  GENSPACE_MIN_SELECTABLE_DURATION_S,
  getOfferingVideoGenerationModelSpecs,
  offeringSpecsAsPipelineItems,
  resolveVideoGenerationOptions,
  VIDEO_GENERATION_RESOLUTION_ORDER,
  untrainedAspectRatioWarning,
  orderAspectRatiosForPicker,
  type OfferingId,
  type ResolvedVideoGenerationOptions,
  type VideoGenerationDuration,
  type VideoGenerationFps,
  type VideoGenerationModelSpecsResponse,
  type VideoGenerationResolution,
  type WarnedVideoGenerationOption,
} from "../../lib/video-generation-model-specs.ts";
import { videoGenerationResolutionLabel } from "../../lib/video-resolution.ts";
import { chunkedRunNotice, generationResolutionWarning } from "./lowPerformanceWarning.ts";
import type {
  FeatureFieldChange,
  FeatureFieldSchema,
  FeatureValues,
  FieldOption,
  ValidationIssue,
} from "../screens/Feature/types.ts";
import {
  clampVideoFields,
  collectModelDisabledFpsOptions,
  collectModelFpsOptions,
  retargetVideoFields,
  type ClampedVideoFields,
  type VideoFeatureAspectRatio,
} from "./videoFieldPolicy.ts";

/** Single source: the shared canonical order (one list, not two). */
export const VIDEO_SETTING_RESOLUTIONS: readonly VideoGenerationResolution[] =
  VIDEO_GENERATION_RESOLUTION_ORDER;
export const VIDEO_SETTING_FPS_VALUES: readonly VideoGenerationFps[] = [
  24, 25, 48, 50,
];
export const VIDEO_SETTING_DURATIONS: readonly VideoGenerationDuration[] = [
  2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20,
];

export type VideoSettingsFieldSchema = FeatureFieldSchema & {
  model: { kind: "options"; value: OfferingId };
  aspectRatio: { kind: "options"; value: VideoFeatureAspectRatio };
  resolution: { kind: "options"; value: VideoGenerationResolution };
  duration: { kind: "options"; value: VideoGenerationDuration };
  fps: { kind: "options"; value: VideoGenerationFps };
};

export type VideoSettingsValues<TSchema extends VideoSettingsFieldSchema> =
  FeatureValues<TSchema> & ClampedVideoFields;

type VideoSettingsModelField = {
  kind: "options";
  id: string;
  label: string;
  dataKey: "model";
  forcePicker: true;
  options: FieldOption<OfferingId>[];
};

type VideoSettingsAspectField<TAspect extends VideoFeatureAspectRatio> = {
  kind: "options";
  id: string;
  label: string;
  dataKey: "aspectRatio";
  options: FieldOption<TAspect>[];
};

type VideoSettingsResolutionField = {
  kind: "options";
  id: string;
  label: string;
  dataKey: "resolution";
  maxOptionCount: number;
  options: FieldOption<VideoGenerationResolution>[];
};

type VideoSettingsDurationField = {
  kind: "options";
  id: string;
  label: string;
  dataKey: "duration";
  forcePicker: true;
  options: FieldOption<VideoGenerationDuration>[];
};

type VideoSettingsFpsField = {
  kind: "options";
  id: string;
  label: string;
  dataKey: "fps";
  maxOptionCount: number;
  options: FieldOption<VideoGenerationFps>[];
};

type PassShapeField = {
  kind: "options";
  id: string;
  label: string;
  dataKey: "passShape";
  info: string;
  defaultValue: ForcedPassShape;
  options: FieldOption<ForcedPassShape>[];
};

const PASS_SHAPE_INFO =
  "All at once makes the whole clip at once, so there are no joins, but it needs more memory. " +
  "In parts, the clip is made in overlapping pieces that each fit. Lighting or subjects can " +
  "drift a little where they meet.";

/**
 * "Make it": all at once or in parts, each with its memory. Empty wherever the choice isn't real
 * (see `passShapeChoice`) or the priced cell is still unknown, so callers spread it as the last
 * settings row.
 */
export function createPassShapeFields(
  cell: PassShapeSelection | null,
  specs: VideoGenerationModelSpecsResponse | null | undefined,
): PassShapeField[] {
  const choice = cell == null ? null : passShapeChoice(specs, cell);
  if (choice == null) return [];
  const memory = (ask: typeof choice.onePass): Pick<FieldOption, "memory"> => {
    const option = askOptionMemory(ask, specs?.memory);
    return option === undefined ? {} : { memory: option };
  };
  return [
    {
      kind: "options",
      id: "passShape",
      label: "Make it",
      dataKey: "passShape",
      info: PASS_SHAPE_INFO,
      defaultValue: choice.automatic,
      options: [
        { value: "one_pass", label: "All at once", ...memory(choice.onePass) },
        { value: "parts", label: "In parts", ...memory(choice.parts) },
      ],
    },
  ];
}

export type VideoSettingsOptionField<
  TAspect extends VideoFeatureAspectRatio = VideoFeatureAspectRatio,
> =
  | VideoSettingsModelField
  | VideoSettingsAspectField<TAspect>
  | VideoSettingsResolutionField
  | VideoSettingsDurationField
  | VideoSettingsFpsField;

/** The "runs in N windows" notice of the option's cell, when the backend splits that cell. */
function chunkedNotice<T extends string | number>(
  chunked: readonly { value: T; windows: number }[],
  value: T,
): string | undefined {
  const match = chunked.find((option) => option.value === value);
  return match ? chunkedRunNotice(match.windows) : undefined;
}

/**
 * What the backend says about a resolution's cell at the selected fps and duration: its windows,
 * `null` for a single pass, or `undefined` when unknown. It is unknown with an automatic
 * duration, and where the resolution does not offer the selected duration.
 */
function resolutionChunk(
  resolved: ResolvedVideoGenerationOptions,
  resolution: VideoGenerationResolution,
): { windows: number } | null | undefined {
  const run = resolved.chunkedResolutionOptions.find((option) => option.value === resolution);
  if (run != null) return run;
  return resolved.singlePassResolutionOptions.includes(resolution) ? null : undefined;
}

/** With the memory meter on, memory lives in the meter: options keep only their quality caveat. */
function optionWarning<T extends string | number>(
  memoryMeter: boolean,
  value: T,
  caveat: string | undefined,
  warned: readonly WarnedVideoGenerationOption<T>[],
): Pick<FieldOption, "warning" | "warningTone" | "budget" | "budgetNote"> {
  if (!memoryMeter) return withBudgetWarning(value, caveat, warned);
  return caveat != null ? { warning: caveat } : {};
}

function disabledFields(
  memoryMeter: boolean,
  option: { message: string; readout?: FieldOption["budget"] },
): Pick<FieldOption, "disabledReason" | "budget"> {
  return {
    disabledReason: option.message,
    ...(!memoryMeter && option.readout ? { budget: option.readout } : {}),
  };
}

function withBudgetWarning<T extends string | number>(
  value: T,
  caveat: string | undefined,
  warned: readonly WarnedVideoGenerationOption<T>[],
): Pick<FieldOption, "warning" | "warningTone" | "budget" | "budgetNote"> {
  const match = warned.find((option) => option.value === value);
  return {
    ...riskOptionFields(match?.message, caveat),
    ...(match?.readout
      ? { budget: match.readout, ...(caveat != null ? { budgetNote: caveat } : {}) }
      : {}),
  };
}

export function isSupportedVideoResolution(
  value: unknown,
): value is VideoGenerationResolution {
  return VIDEO_SETTING_RESOLUTIONS.some((resolution) => resolution === value);
}

export function isSupportedVideoFps(value: unknown): value is VideoGenerationFps {
  return VIDEO_SETTING_FPS_VALUES.some((fps) => fps === value);
}

export function isSupportedVideoDuration(
  value: unknown,
): value is VideoGenerationDuration {
  return VIDEO_SETTING_DURATIONS.some((duration) => duration === value);
}

export function createVideoSettingsOptionFields<
  TSchema extends VideoSettingsFieldSchema,
>(
  values: Pick<
    FeatureValues<TSchema>,
    "model" | "aspectRatio" | "resolution" | "duration" | "fps"
  > & { passShape?: ForcedPassShape | null },
  specs: VideoGenerationModelSpecsResponse | null | undefined,
  input: {
    includeAuto?: boolean;
    memoryMeter?: boolean;
    /** Prices Auto at the start image's ratio, as the run does. */
    startFrameSize?: ImageSize | null;
  } = {},
): [
  VideoSettingsModelField,
  VideoSettingsAspectField<
    Extract<TSchema["aspectRatio"]["value"], VideoFeatureAspectRatio>
  >,
  VideoSettingsResolutionField,
  VideoSettingsDurationField,
  VideoSettingsFpsField,
] {
  const offeringSpecs = getOfferingVideoGenerationModelSpecs(specs);
  const resolved = resolveVideoGenerationOptions({
    settings: {
      model: values.model,
      duration: values.duration,
      videoResolution: values.resolution,
      fps: values.fps,
      aspectRatio: values.aspectRatio === "auto" ? "16:9" : values.aspectRatio,
    },
    modelSpecs: offeringSpecsAsPipelineItems(offeringSpecs),
    hasAudio: false,
    minimumDuration: GENSPACE_MIN_SELECTABLE_DURATION_S,
  });
  const selectedModel =
    resolved.modelOptions.find(
      (item) => item.pipeline === resolved.selectedModel,
    ) ?? resolved.modelOptions[0];
  const durationOptions = resolved.durationOptions.filter(isSupportedVideoDuration);
  const fpsOptions = collectModelFpsOptions(
    selectedModel,
    resolved.fpsOptions,
  ).filter(isSupportedVideoFps);
  // Resolution-scoped like the aspect picker: a duration the current cell
  // cannot run renders greyed with the backend refusal instead of staying
  // enabled (union) and silently retargeting the resolution on pick.
  const disabledDurations = resolved.disabledDurationOptions;
  // Withheld fps values append after keeps (union scope, numeric): the budget
  // grows monotonically, so drops sit above keeps.
  const disabledFps = collectModelDisabledFpsOptions(selectedModel, fpsOptions);
  const memoryMeter = input.memoryMeter ?? false;
  const selection: PassShapeSelection = {
    model: values.model,
    resolution: values.resolution,
    fps: values.fps,
    duration: values.duration,
    aspectRatio: values.aspectRatio,
    passShape: values.passShape,
  };
  const preview = (patch: Partial<typeof selection>): Pick<FieldOption, "memory"> => {
    if (!memoryMeter) return {};
    const memory = optionMemory(
      specs,
      meterCell(specs, { ...selection, ...patch }, input.startFrameSize),
    );
    return memory === undefined ? {} : { memory };
  };

  return [
    {
      kind: "options",
      id: "model",
      label: "Model",
      dataKey: "model",
      forcePicker: true,
      options: offeringSpecs
        .filter((row) =>
          resolved.modelOptions.some((item) => item.pipeline === String(row.model)),
        )
        .map((row) => ({
          value: row.model,
          label: row.spec.display_name,
          ...preview({ model: row.model }),
        })),
    },
    {
      kind: "options",
      id: "aspectRatio",
      label: "Aspect Ratio",
      dataKey: "aspectRatio",
      options: [
        ...(input.includeAuto
          ? [{ value: "auto" as const, label: "Auto" }]
          : []),
        ...orderAspectRatiosForPicker(resolved.aspectRatioOptions).map((aspectRatio) => {
          const disabled = resolved.disabledAspectRatioOptions.find(
            (option) => option.value === aspectRatio,
          );
          return {
            value: aspectRatio,
            label: aspectRatio,
            ...optionWarning(
              memoryMeter,
              aspectRatio,
              untrainedAspectRatioWarning(aspectRatio),
              resolved.warnedAspectRatioOptions,
            ),
            ...(disabled ? disabledFields(memoryMeter, disabled) : {}),
            ...preview({ aspectRatio }),
          };
        }),
      ] as VideoSettingsAspectField<
        Extract<TSchema["aspectRatio"]["value"], VideoFeatureAspectRatio>
      >["options"],
    },
    {
      kind: "options",
      id: "resolution",
      label: "Resolution",
      dataKey: "resolution",
      maxOptionCount: VIDEO_SETTING_RESOLUTIONS.length,
      options: [
        ...[...resolved.resolutionOptions]
          .sort(
            (left, right) =>
              VIDEO_SETTING_RESOLUTIONS.indexOf(left) -
              VIDEO_SETTING_RESOLUTIONS.indexOf(right),
          )
          .map((resolution) => ({
            value: resolution,
            label: videoGenerationResolutionLabel(resolution),
            ...optionWarning(
              memoryMeter,
              resolution,
              generationResolutionWarning(
                resolution,
                specs?.low_performance_machine,
                // The meter says when the picked cell runs in parts.
                memoryMeter ? undefined : resolutionChunk(resolved, resolution),
              ),
              resolved.warnedResolutionOptions,
            ),
            ...preview({ resolution }),
          })),
        ...resolved.disabledResolutionOptions.map((option) => ({
          value: option.value,
          label: videoGenerationResolutionLabel(option.value),
          ...disabledFields(memoryMeter, option),
          ...preview({ resolution: option.value }),
        })),
      ],
    },
    {
      kind: "options",
      id: "duration",
      label: "Duration",
      dataKey: "duration",
      forcePicker: true,
      options: [
        ...durationOptions.map((duration) => ({
          value: duration,
          label: `${duration}s`,
          ...optionWarning(
            memoryMeter,
            duration,
            memoryMeter ? undefined : chunkedNotice(resolved.chunkedDurationOptions, duration),
            resolved.warnedDurationOptions,
          ),
          ...preview({ duration }),
        })),
        ...disabledDurations.map((option) => ({
          value: option.value,
          label: `${option.value}s`,
          ...disabledFields(memoryMeter, option),
          ...preview({ duration: option.value }),
        })),
      ],
    },
    {
      kind: "options",
      id: "fps",
      label: "Frame Rate",
      dataKey: "fps",
      maxOptionCount: VIDEO_SETTING_FPS_VALUES.length,
      options: [
        ...fpsOptions.map((fps) => ({
          value: fps,
          label: String(fps),
          ...optionWarning(
            memoryMeter,
            fps,
            memoryMeter ? undefined : chunkedNotice(resolved.chunkedFpsOptions, fps),
            resolved.warnedFpsOptions,
          ),
          ...preview({ fps }),
        })),
        ...disabledFps.map((option) => ({
          value: option.value,
          label: String(option.value),
          ...disabledFields(memoryMeter, option),
          ...preview({ fps: option.value }),
        })),
      ],
    },
  ];
}

export function fastBudgetIssues(
  specs: VideoGenerationModelSpecsResponse | null | undefined,
  selection: PassShapeSelection | null,
): ValidationIssue[] {
  if (selection == null) return [];
  const risk = selectedFastRisk(specs, selection);
  if (risk == null || !isConfirmableBudgetTier(risk.tier)) return [];
  return [
    {
      id: "budget-risk",
      fieldId: "generate",
      message: risk.message,
      alwaysRevealed: true,
      blocksGenerate: false,
      tone: "risk",
      tier: risk.tier,
    },
  ];
}

export function applyVideoSettingsChange<
  TSchema extends VideoSettingsFieldSchema,
>(
  values: VideoSettingsValues<TSchema>,
  change: FeatureFieldChange<TSchema>,
  specs: VideoGenerationModelSpecsResponse | null | undefined,
  defaults: ClampedVideoFields,
): VideoSettingsValues<TSchema> {
  const merged = { ...values, [change.dataKey]: change.value };
  if (change.dataKey === "duration" || change.dataKey === "fps") {
    return clampVideoFields(
      retargetVideoFields(merged, specs, change.dataKey),
      specs,
      defaults,
    );
  }
  return clampVideoFields(merged, specs, defaults);
}
