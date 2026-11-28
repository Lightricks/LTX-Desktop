import { isH3Variant } from "../../lib/h3-models.ts";
import { paths } from "../../paths.ts";
import { isLocalVideoSelectionReady, localVideoHasCompatibleOptions, type VideoFeatureContext } from "./videoFieldPolicy.ts";
import {
  createVideoSettingsOptionFields, applyVideoSettingsChange,
  type VideoSettingsFieldSchema, type VideoSettingsValues,
} from "./videoSettingsControls.ts";
import type { FeatureFieldChange, FieldOption } from "../screens/Feature/types.ts";
import {
  defaultPickerModel, isExternalModel, pickerGrid, pickerOptions, rendererIssue, wireModel,
  type VideoPickerModel,
} from "./amdVideoPicker.ts";
import type { ImageSize } from "../../lib/pass-shape.ts";

export type PickerSchema = VideoSettingsFieldSchema & {
  model: { kind: "options"; value: VideoPickerModel };
};

export function pickerSettingsFields<T extends PickerSchema>(
  values: VideoSettingsValues<T>, context: VideoFeatureContext,
  options: { includeAuto?: boolean; memoryMeter?: boolean; startFrameSize?: ImageSize | null },
) {
  const model: VideoPickerModel = values.model;
  const grid = pickerGrid(context.specs, model);
  const [first, aspect, resolution, duration, fps] = createVideoSettingsOptionFields<T>(
    { ...values, model: wireModel(model) }, grid,
    { ...options, memoryMeter: !isExternalModel(model) && options.memoryMeter },
  );
  const native = isExternalModel(model)
    ? createVideoSettingsOptionFields<T>({ ...values, model: "ltx-2.5-fast" }, context.specs)[0]
    : first;
  const optionsForModel: FieldOption<VideoPickerModel>[] = context.amd
    ? pickerOptions(native.options, context.specs, context.amd, options.includeAuto === true)
    : native.options;
  const sizeLabel = (value: string) => model === "kandinsky" ? "480p SD" : value === "540p" ? "480p" : "768p";
  return [
    { ...first, options: optionsForModel,
      ...(context.amd?.desktop ? { relatedPage: { to: paths.projects, label: "Cloud Fast and Pro models in Gen Space" } } : {}),
    },
    aspect,
    model === "kandinsky" || isH3Variant(model) ? { ...resolution, options: resolution.options.map(o => ({ ...o, label: sizeLabel(o.value) })) } : resolution,
    duration, fps,
  ] as const;
}

export function pickerDefault<T extends { model: VideoPickerModel }>(values: T, context: VideoFeatureContext): T {
  return { ...values, model: defaultPickerModel(context.specs, context.amd) };
}

export function changePickerSettings<T extends PickerSchema>(
  values: VideoSettingsValues<T>, change: FeatureFieldChange<T>, context: VideoFeatureContext,
  defaults: VideoSettingsValues<T>,
): VideoSettingsValues<T> {
  const model = change.dataKey === "model" ? change.value as VideoPickerModel : values.model;
  const grid = pickerGrid(context.specs, model);
  const gridValues = { ...values, model: wireModel(model) };
  const gridChange = change.dataKey === "model" ? { ...change, value: wireModel(model) } : change;
  const result = applyVideoSettingsChange<T>(gridValues, gridChange, grid, defaults);
  return isExternalModel(model) ? { ...result, passShape: null, model } : result;
}

export function pickerReady(values: VideoSettingsValues<PickerSchema>, context: VideoFeatureContext): boolean {
  return !rendererIssue(values.model, context.amd) && isLocalVideoSelectionReady(
    { ...values, model: wireModel(values.model) }, pickerGrid(context.specs, values.model),
  );
}

export function pickerUnavailable(values: VideoSettingsValues<PickerSchema>, context: VideoFeatureContext): boolean {
  // Keep the model menu on screen even if the current choice has no installed weights.
  return !context.amd && !localVideoHasCompatibleOptions(values, context.specs);
}
