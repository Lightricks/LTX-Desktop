import {
  GENSPACE_MIN_SELECTABLE_DURATION_S,
  getOfferingVideoGenerationModelSpecs,
  offeringSpecsAsPipelineItems,
  resolveVideoGenerationOptions,
  isVideoGenerationAspectRatio,
  sanitizeVideoGenerationSettings,
  type VideoGenerationAspectRatio,
  type VideoGenerationFormSettings,
  type VideoGenerationModelSpecItem,
  type VideoGenerationModelSpecsResponse,
  type VideoGenerationSettingsShape,
} from "../../lib/video-generation-model-specs.ts";

export type VideoFeatureAspectRatio = VideoGenerationAspectRatio | "auto";

export type ClampedVideoFields = Omit<VideoGenerationFormSettings, "aspectRatio" | "model"> & {
  aspectRatio: VideoFeatureAspectRatio;
  model: string;
};

export type VideoFeatureContext = {
  specs: VideoGenerationModelSpecsResponse | null | undefined;
};

function pick<T>(current: T, fallback: T, options: readonly T[]): T | null {
  if (options.some((option) => option === current)) return current;
  if (options.some((option) => option === fallback)) return fallback;
  return options[0] ?? null;
}

function persistAspectRatio(
  value: VideoFeatureAspectRatio,
): VideoFeatureAspectRatio {
  if (value === "auto" || isVideoGenerationAspectRatio(value)) return value;
  return "16:9";
}

function offeringPipelineSpecs(
  specs: VideoGenerationModelSpecsResponse | null | undefined,
): VideoGenerationModelSpecItem[] {
  return offeringSpecsAsPipelineItems(getOfferingVideoGenerationModelSpecs(specs));
}

function toSettingsShape(values: ClampedVideoFields): VideoGenerationSettingsShape {
  const aspectRatio = persistAspectRatio(values.aspectRatio);
  return {
    model: values.model,
    duration: values.duration,
    videoResolution: values.resolution,
    fps: values.fps,
    // Option resolution has no Auto cell; keep persisted Auto out of the matrix.
    aspectRatio: aspectRatio === "auto" ? "16:9" : aspectRatio,
  };
}

/**
 * Keep current if still legal → else field default if in list → else first remaining.
 * Maps store `resolution` ↔ helper `videoResolution` at the boundary.
 */
export function clampVideoFields<T extends ClampedVideoFields>(
  values: T,
  specs: VideoGenerationModelSpecsResponse | null | undefined,
  defaults: ClampedVideoFields,
): T {
  const modelSpecs = offeringPipelineSpecs(specs);
  if (modelSpecs.length === 0) return values;

  const clampOptions = {
    hasAudio: false,
    minimumDuration: GENSPACE_MIN_SELECTABLE_DURATION_S,
  } as const;

  let mapped = toSettingsShape(values);

  let resolved = resolveVideoGenerationOptions({
    settings: mapped,
    modelSpecs,
    ...clampOptions,
  });
  const model =
    pick(
      mapped.model,
      defaults.model,
      resolved.modelOptions.map((item) => item.pipeline),
    ) ?? mapped.model;
  mapped = { ...mapped, model };

  resolved = resolveVideoGenerationOptions({
    settings: mapped,
    modelSpecs,
    ...clampOptions,
  });
  const videoResolution =
    pick(
      mapped.videoResolution,
      defaults.resolution,
      resolved.resolutionOptions,
    ) ?? mapped.videoResolution;
  mapped = { ...mapped, videoResolution };

  resolved = resolveVideoGenerationOptions({
    settings: mapped,
    modelSpecs,
    ...clampOptions,
  });
  const fps = pick(mapped.fps, defaults.fps, resolved.fpsOptions) ?? mapped.fps;
  mapped = { ...mapped, fps };

  resolved = resolveVideoGenerationOptions({
    settings: mapped,
    modelSpecs,
    ...clampOptions,
  });
  const duration =
    pick(mapped.duration, defaults.duration, resolved.durationOptions) ??
    mapped.duration;
  const persistedAspect = persistAspectRatio(values.aspectRatio);
  const defaultAspect = persistAspectRatio(defaults.aspectRatio);
  const aspectRatio =
    persistedAspect === "auto"
      ? "auto"
      : (pick(
          persistedAspect,
          defaultAspect === "auto" ? "16:9" : defaultAspect,
          resolved.aspectRatioOptions,
        ) ?? persistedAspect);
  mapped = {
    ...mapped,
    duration,
    aspectRatio: aspectRatio === "auto" ? "16:9" : aspectRatio,
  };

  const sanitized = sanitizeVideoGenerationSettings(
    mapped,
    modelSpecs,
    clampOptions,
  );
  if (!sanitized || sanitized.duration == null) {
    return {
      ...values,
      model,
      resolution: videoResolution,
      fps,
      duration,
      aspectRatio,
    };
  }

  return {
    ...values,
    model: sanitized.model,
    resolution: sanitized.videoResolution,
    fps: sanitized.fps,
    duration: sanitized.duration,
    aspectRatio,
  };
}

/**
 * Audio-conditioned features resolve against the model's A2V matrix instead of
 * the plain one, so they must opt in rather than inherit the silent-video path.
 */
export type LocalVideoSelectionOptions = {
  hasAudio?: boolean;
};

/**
 * True only when `values` already match a compatible local selection. False while
 * specs are absent (unclamped) — gate Generate on it so nothing ships pre-clamp.
 */
export function isLocalVideoSelectionReady(
  values: ClampedVideoFields,
  specs: VideoGenerationModelSpecsResponse | null | undefined,
  options: LocalVideoSelectionOptions = {},
): boolean {
  const modelSpecs = offeringPipelineSpecs(specs);
  if (modelSpecs.length === 0) return false;
  const resolved = resolveVideoGenerationOptions({
    settings: toSettingsShape(values),
    modelSpecs,
    hasAudio: options.hasAudio ?? false,
    minimumDuration: GENSPACE_MIN_SELECTABLE_DURATION_S,
  });
  return (
    resolved.hasCompatibleOptions &&
    resolved.selectedModel === values.model &&
    resolved.selectedResolution === values.resolution &&
    resolved.selectedFps === values.fps &&
    resolved.selectedDuration === values.duration
  );
}

export function localVideoHasCompatibleOptions(
  values: ClampedVideoFields,
  specs: VideoGenerationModelSpecsResponse | null | undefined,
  options: LocalVideoSelectionOptions = {},
): boolean {
  const modelSpecs = offeringPipelineSpecs(specs);
  return resolveVideoGenerationOptions({
    settings: toSettingsShape(values),
    modelSpecs,
    hasAudio: options.hasAudio ?? false,
    minimumDuration: GENSPACE_MIN_SELECTABLE_DURATION_S,
  }).hasCompatibleOptions;
}

function uniqueSortedNumbers(values: Iterable<number>): number[] {
  return [...new Set(values)].sort((left, right) => left - right);
}

function isLegalDuration(duration: number): boolean {
  return duration >= GENSPACE_MIN_SELECTABLE_DURATION_S;
}

function resolutionSpecMap(item: VideoGenerationModelSpecItem) {
  return item.spec.supported_resolutions_durations ?? {};
}

function durationsForFps(
  item: VideoGenerationModelSpecItem,
  resolution: string,
  fps: number,
): number[] {
  const list =
    resolutionSpecMap(item)[resolution]?.fps_to_durations[String(fps)] ?? [];
  return list.filter(isLegalDuration);
}

export function collectModelDurationOptions(
  item: VideoGenerationModelSpecItem | undefined,
  fallback: readonly number[] = [],
): number[] {
  if (!item) return uniqueSortedNumbers(fallback);
  const durations: number[] = [];
  for (const spec of Object.values(resolutionSpecMap(item))) {
    for (const list of Object.values(spec.fps_to_durations ?? {})) {
      durations.push(...list.filter(isLegalDuration));
    }
  }
  const unique = uniqueSortedNumbers(durations);
  return unique.length > 0 ? unique : uniqueSortedNumbers(fallback);
}

export function collectModelFpsOptions(
  item: VideoGenerationModelSpecItem | undefined,
  fallback: readonly number[] = [],
): number[] {
  if (!item) return uniqueSortedNumbers(fallback);
  const fpsValues: number[] = [];
  for (const spec of Object.values(resolutionSpecMap(item))) {
    for (const [fpsKey, list] of Object.entries(spec.fps_to_durations ?? {})) {
      if (list.some(isLegalDuration)) fpsValues.push(Number(fpsKey));
    }
  }
  const unique = uniqueSortedNumbers(fpsValues);
  return unique.length > 0 ? unique : uniqueSortedNumbers(fallback);
}

function selectedModelItem(
  specs: VideoGenerationModelSpecsResponse | null | undefined,
  model: string,
): VideoGenerationModelSpecItem | undefined {
  const modelSpecs = offeringPipelineSpecs(specs);
  return modelSpecs.find((item) => item.pipeline === model) ?? modelSpecs[0];
}

/**
 * When the user picks a duration or fps that isn't legal at the current
 * resolution, move resolution/fps to a combo on the same model that supports it.
 * Hydration clamp stays resolution-first; this is only for explicit field edits.
 */
export function retargetVideoFields<T extends ClampedVideoFields>(
  values: T,
  specs: VideoGenerationModelSpecsResponse | null | undefined,
  preserve: "duration" | "fps",
): T {
  const item = selectedModelItem(specs, values.model);
  if (!item) return values;

  const resolutions = Object.keys(resolutionSpecMap(item));
  const nearbyResolutions = [
    values.resolution,
    ...resolutions.filter((resolution) => resolution !== values.resolution),
  ];

  if (preserve === "duration") {
    for (const resolution of nearbyResolutions) {
      if (durationsForFps(item, resolution, values.fps).includes(values.duration)) {
        return { ...values, resolution };
      }
    }
    for (const resolution of nearbyResolutions) {
      const spec = resolutionSpecMap(item)[resolution];
      if (!spec) continue;
      for (const fpsKey of Object.keys(spec.fps_to_durations ?? {})) {
        const fps = Number(fpsKey);
        if (durationsForFps(item, resolution, fps).includes(values.duration)) {
          return { ...values, resolution, fps };
        }
      }
    }
    return values;
  }

  for (const resolution of nearbyResolutions) {
    const spec = resolutionSpecMap(item)[resolution];
    if (!spec?.fps_to_durations[String(values.fps)]) continue;
    const durations = durationsForFps(item, resolution, values.fps);
    if (durations.length === 0) continue;
    const duration = durations.includes(values.duration)
      ? values.duration
      : durations[0];
    return { ...values, resolution, duration };
  }
  return values;
}
