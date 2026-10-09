import {
  GENSPACE_MIN_SELECTABLE_DURATION_S,
  getOfferingVideoGenerationModelSpecs,
  isOfferingId,
  offeringSpecsAsPipelineItems,
  OFFERING_IDS,
  resolveVideoGenerationOptions,
  untrainedAspectRatioWarning,
  orderAspectRatiosForPicker,
  type OfferingId,
  type VideoGenerationDuration,
  type VideoGenerationFps,
  type VideoGenerationModelSpecsResponse,
  type VideoGenerationResolution,
} from "../../lib/video-generation-model-specs.ts";
import type {
  FeatureFieldChange,
  FeatureFieldSchema,
  FeatureValues,
  FieldOption,
} from "../screens/Feature/types.ts";
import {
  clampVideoFields,
  collectModelDurationOptions,
  collectModelFpsOptions,
  retargetVideoFields,
  type ClampedVideoFields,
  type VideoFeatureAspectRatio,
} from "./videoFieldPolicy.ts";

export const VIDEO_SETTING_MODELS: readonly OfferingId[] = OFFERING_IDS;
export const VIDEO_SETTING_RESOLUTIONS: readonly VideoGenerationResolution[] = [
  "270p",
  "360p",
  "540p",
  "720p",
  "1080p",
  "1440p",
  "2160p",
];
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

export type VideoSettingsOptionField<
  TAspect extends VideoFeatureAspectRatio = VideoFeatureAspectRatio,
> =
  | VideoSettingsModelField
  | VideoSettingsAspectField<TAspect>
  | VideoSettingsResolutionField
  | VideoSettingsDurationField
  | VideoSettingsFpsField;

export function isSupportedVideoModel(
  value: unknown,
): value is OfferingId {
  return isOfferingId(value);
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
  >,
  specs: VideoGenerationModelSpecsResponse | null | undefined,
  input: { includeAuto?: boolean } = {},
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
  const durationOptions = collectModelDurationOptions(
    selectedModel,
    resolved.durationOptions,
  ).filter(isSupportedVideoDuration);
  const fpsOptions = collectModelFpsOptions(
    selectedModel,
    resolved.fpsOptions,
  ).filter(isSupportedVideoFps);

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
          const warning = untrainedAspectRatioWarning(aspectRatio);
          return {
            value: aspectRatio,
            label: aspectRatio,
            ...(warning ? { warning } : {}),
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
      options: [...resolved.resolutionOptions]
        .sort(
          (left, right) =>
            VIDEO_SETTING_RESOLUTIONS.indexOf(left) -
            VIDEO_SETTING_RESOLUTIONS.indexOf(right),
        )
        .map((resolution) => ({
          value: resolution,
          label: resolution,
        })),
    },
    {
      kind: "options",
      id: "duration",
      label: "Duration",
      dataKey: "duration",
      forcePicker: true,
      options: durationOptions.map((duration) => ({
        value: duration,
        label: `${duration}s`,
      })),
    },
    {
      kind: "options",
      id: "fps",
      label: "Frame Rate",
      dataKey: "fps",
      maxOptionCount: VIDEO_SETTING_FPS_VALUES.length,
      options: fpsOptions.map((fps) => ({
        value: fps,
        label: String(fps),
      })),
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
