import { CUSTOM_LORA_HOME_FEATURE } from "../../lib/custom-lora.ts";
import { isIcLoraTrackRecipeId } from "../../lib/ic-lora-recipes.ts";
import { roundDurationSeconds } from "../lib/roundDuration.ts";

/** Desktop live setting wins; Remote uses the status fetched when the feature form mounts. */
export function resolveShowManualEnhance(
  liveAutoEnhance: boolean | undefined,
  statusShowManualEnhance: boolean | undefined,
): boolean {
  if (liveAutoEnhance !== undefined) return !liveAutoEnhance;
  return statusShowManualEnhance ?? false;
}

/** Desktop live setting wins; Remote uses the setting the status carries. Unknown is off. */
export function resolveShowNagStrengthControls(
  liveSetting: boolean | undefined,
  statusSetting: boolean | undefined,
): boolean {
  return liveSetting ?? statusSetting ?? false;
}

/** A still of a Keyframes form, as Enhance reads it. */
export type EnhanceKeyframe = { asset: { assetId: string }; frameIndex: number; strength: number };

/**
 * Enhance can run from a typed prompt or from media alone: a start frame is captioned, a
 * source video (IC-LoRA, Extend, Retake) is sampled into stills, the images of an
 * image-list recipe (Ingredients) are read as panels, and the stills of a Keyframes clip are
 * read at their times. Audio alone is not enough.
 */
export function hasManualEnhanceInput(input: {
  prompt: string;
  startFrame?: { assetId: string } | null;
  video?: { assetId: string } | null;
  images?: readonly { assetId: string }[] | null;
  keyframes?: readonly EnhanceKeyframe[] | null;
}): boolean {
  return (
    input.prompt.trim().length > 0 ||
    Boolean(input.startFrame?.assetId) ||
    Boolean(input.video?.assetId) ||
    (input.images?.length ?? 0) > 0 ||
    (input.keyframes?.length ?? 0) > 0
  );
}

/**
 * The stills of a Keyframes form as one text to compare: asset, frame and strength in time
 * order, then the clip length and frame rate. The rewrite reads all of them: it weights a still
 * by its strength and names its clock time, which comes from the frame and the frame rate.
 * Empty for no stills.
 */
export function keyframesSignature(
  keyframes?: readonly EnhanceKeyframe[] | null,
  timing?: { duration?: unknown; fps?: unknown },
): string {
  if (!keyframes || keyframes.length === 0) return "";
  const stills = [...keyframes]
    .sort((a, b) => a.frameIndex - b.frameIndex)
    .map((keyframe) => `${keyframe.asset.assetId}@${keyframe.frameIndex}x${keyframe.strength}`)
    .join(",");
  return `${stills}|${String(timing?.duration)}|${String(timing?.fps)}`;
}

/**
 * The stills an Enhance request sends for a Keyframes clip, in time order, with the clip timing
 * the rewrite needs for their clock times. Any other feature has no stills and sends none.
 */
export function enhanceKeyframeFields(input: {
  keyframes?: readonly EnhanceKeyframe[] | null;
  duration?: unknown;
  fps?: unknown;
}):
  | {
      keyframeAssets: { assetId: string; frameIndex: number; strength: number }[];
      duration?: number;
      fps?: number;
    }
  | undefined {
  if (!input.keyframes || input.keyframes.length === 0) return undefined;
  const timing =
    Number.isInteger(input.duration) &&
    (input.duration as number) > 0 &&
    Number.isInteger(input.fps) &&
    (input.fps as number) > 0
      ? { duration: input.duration as number, fps: input.fps as number }
      : {};
  return {
    keyframeAssets: [...input.keyframes]
      .sort((a, b) => a.frameIndex - b.frameIndex)
      .map((keyframe) => ({
        assetId: keyframe.asset.assetId,
        frameIndex: keyframe.frameIndex,
        strength: keyframe.strength,
      })),
    ...timing,
  };
}

/**
 * The panel images an image-list recipe sends to Enhance, in order. Any other feature sends
 * none: its images are a start frame or not part of Enhance.
 */
export function panelImageAssetIds(input: {
  isSheetRecipe: boolean;
  images?: readonly { assetId: string }[] | null;
}): string[] | undefined {
  if (!input.isSheetRecipe || !input.images || input.images.length === 0) return undefined;
  return input.images.map((image) => image.assetId);
}

/**
 * Whether an Enhance result still belongs to the panel images it was built from. `live` is
 * undefined when the form values are gone. The order counts: the rewrite names the panels in it.
 */
export function isPanelListCurrent(
  live: readonly string[] | undefined,
  source: readonly string[],
): boolean {
  return (
    live === undefined ||
    (live.length === source.length && live.every((assetId, index) => assetId === source[index]))
  );
}

/** The asset id of a form field value that holds an asset, or null when it holds none. */
export function assetIdOf(value: unknown): string | null {
  return value != null &&
    typeof value === "object" &&
    "assetId" in value &&
    typeof value.assetId === "string"
    ? value.assetId
    : null;
}

/**
 * Whether an Enhance result still belongs to the source video it was built from. `live` is
 * undefined when the form values are gone, `null` when the form has no video.
 */
export function isSourceVideoCurrent(
  live: string | null | undefined,
  source: string | null,
): boolean {
  return live === undefined || live === source;
}

/**
 * The look image an IC-LoRA recipe sends beside its source video. An IC-LoRA driven by an
 * image, or any other feature, sends none: its image is a start frame or not part of Enhance.
 */
export function lookImageAssetId(input: {
  isIcLoraRecipe: boolean;
  video?: { assetId: string } | null;
  image?: { assetId: string } | null;
}): string | undefined {
  return input.isIcLoraRecipe && input.video?.assetId ? input.image?.assetId : undefined;
}

/**
 * The single image an Enhance request captions: the start frame of Image-to-Video, Audio-to-Video,
 * Custom LoRA, or a Motion Track recipe, or the look image beside the source video of a video IC-LoRA.
 */
export function enhanceImageAssetId(input: {
  featureId: string;
  isIcLoraRecipe: boolean;
  startFrame?: { assetId: string } | null;
  video?: { assetId: string } | null;
  image?: { assetId: string } | null;
}): string | undefined {
  if (
    input.featureId === "image-to-video" ||
    input.featureId === "audio-to-video" ||
    // Only an image-to-video custom LoRA holds a start frame. A text-to-video one clears it.
    input.featureId === CUSTOM_LORA_HOME_FEATURE.id ||
    isIcLoraTrackRecipeId(input.featureId)
  ) {
    return input.startFrame?.assetId;
  }
  return lookImageAssetId(input);
}

/** The media an Enhance request was built from, as the form held it when the request started. */
export type EnhanceSourceMedia = {
  videoAssetId: string | null;
  startFrameAssetId: string | null;
  referenceImageAssetId: string | null;
  panelAssetIds: readonly string[];
  keyframes: string;
  replacedWindow: { start: number; end: number } | undefined;
};

/** The same media as the form holds it now. A field is undefined when the form values are gone. */
export type LiveEnhanceSourceMedia = {
  videoAssetId: string | null | undefined;
  startFrameAssetId: string | null | undefined;
  referenceImageAssetId: string | null | undefined;
  panelAssetIds: readonly string[] | undefined;
  keyframes: string | undefined;
  replacedWindow: { start: number; end: number } | undefined;
};

/**
 * Whether an Enhance result still belongs to the media it was built from. A rewrite of an old
 * video, start frame, look image, panel list, or Retake selection must not land on a new one.
 */
export function isEnhanceSourceCurrent(
  live: LiveEnhanceSourceMedia,
  source: EnhanceSourceMedia,
): boolean {
  return (
    isSourceVideoCurrent(live.videoAssetId, source.videoAssetId) &&
    isSourceVideoCurrent(live.startFrameAssetId, source.startFrameAssetId) &&
    isSourceVideoCurrent(live.referenceImageAssetId, source.referenceImageAssetId) &&
    isPanelListCurrent(live.panelAssetIds, source.panelAssetIds) &&
    (live.keyframes === undefined || live.keyframes === source.keyframes) &&
    (live.videoAssetId === undefined ||
      isSameReplacedWindow(live.replacedWindow, source.replacedWindow))
  );
}

/** Whether two Retake windows are the same selection. Both undefined means no window. */
export function isSameReplacedWindow(
  a: { start: number; end: number } | undefined,
  b: { start: number; end: number } | undefined,
): boolean {
  if (a === undefined || b === undefined) return a === b;
  return a.start === b.start && a.end === b.end;
}

/**
 * The part of the source video a Retake replaces, as the create body will send it. Other
 * features replace no part of their video, so they send none.
 */
export function replacedWindowFor(
  featureId: string,
  values: { startTime?: number; duration?: number },
): { start: number; end: number } | undefined {
  if (featureId !== "retake") return undefined;
  if (typeof values.startTime !== "number" || typeof values.duration !== "number") {
    return undefined;
  }
  return {
    start: values.startTime,
    end: values.startTime + roundDurationSeconds(values.duration),
  };
}
