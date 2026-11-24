import type { HomeFeatureId } from "../../../lib/home-features.ts";
import type { AudioToVideoSchema } from "./definitions/audioToVideo.ts";
import type { ImageToVideoSchema } from "./definitions/imageToVideo.ts";
import type { ExtendSchema } from "./definitions/extend.ts";
import type { RetakeSchema } from "./definitions/retake.ts";
import type { TextToVideoSchema } from "./definitions/textToVideo.ts";
import type { FeatureSchemaMap } from "../../stores/featureFormStore.ts";
import {
  createFeatureFieldChange,
  readAudioAssetFieldValue,
  readImageAssetFieldValue,
  readVideoAssetFieldValue,
  type AssetRef,
  type AudioAssetFieldKey,
  type FeatureFieldChange,
  type FeatureFieldKind,
  type FeatureFieldSchema,
  type FeatureFieldSpec,
  type FeatureValues,
  type ImageAssetFieldKey,
  type OptionsFieldKey,
  type TextareaFieldKey,
  type VideoAssetFieldKey,
} from "./types.ts";

type Equal<A, B> =
  (<T>() => T extends A ? 1 : 2) extends <T>() => T extends B ? 1 : 2
    ? true
    : false;
type Expect<T extends true> = T;

type RawTextareaSpec = { kind: "textarea" };
type DeclaredStringTextareaSpec = { kind: "textarea"; value: string };
type ConstrainedTextareaSpec = { kind: "textarea"; value: `keep-${string}` };

const rawTextareaAllowed: RawTextareaSpec extends FeatureFieldSpec
  ? true
  : never = true;
const declaredStringTextareaForbidden: DeclaredStringTextareaSpec extends FeatureFieldSpec
  ? never
  : true = true;
const constrainedTextareaForbidden: ConstrainedTextareaSpec extends FeatureFieldSpec
  ? never
  : true = true;

type MixedSchema = {
  prompt: { kind: "textarea" };
  tag: { kind: "options"; value: string };
  style: { kind: "options"; value: "plain" | "vivid" };
  duration: { kind: "options"; value: number };
  image: { kind: "image-asset"; value: AssetRef | null };
  audio: { kind: "audio-asset"; value: AssetRef | null };
  video: { kind: "video-asset"; value: AssetRef | null };
};

const fieldKindMatchesSpec: Expect<
  Equal<FeatureFieldKind, FeatureFieldSpec["kind"]>
> = true;
const mixedSchemaAllowed: Expect<
  MixedSchema extends FeatureFieldSchema ? true : false
> = true;
const mixedValuesMatch: Expect<
  Equal<
    FeatureValues<MixedSchema>,
    {
      prompt: string;
      tag: string;
      style: "plain" | "vivid";
      duration: number;
      image: AssetRef | null;
      audio: AssetRef | null;
      video: AssetRef | null;
    }
  >
> = true;

const mixedImageValues: FeatureValues<MixedSchema> = {
  prompt: "prompt",
  tag: "tag",
  style: "plain",
  duration: 4,
  image: { assetId: "img-1" },
  audio: { assetId: "aud-1" },
  video: { assetId: "vid-1" },
};
const mixedImageValue = readImageAssetFieldValue<MixedSchema, "image">(
  mixedImageValues,
  "image",
);
const mixedImageValueType: Expect<
  Equal<typeof mixedImageValue, AssetRef | null>
> = true;
const mixedAudioValue = readAudioAssetFieldValue<MixedSchema, "audio">(
  mixedImageValues,
  "audio",
);
const mixedAudioValueType: Expect<
  Equal<typeof mixedAudioValue, AssetRef | null>
> = true;
const mixedVideoValue = readVideoAssetFieldValue<MixedSchema, "video">(
  mixedImageValues,
  "video",
);
const mixedVideoValueType: Expect<
  Equal<typeof mixedVideoValue, AssetRef | null>
> = true;

const rawDomTextareaChange: FeatureFieldChange<MixedSchema> = {
  kind: "textarea",
  fieldId: "prompt",
  dataKey: "prompt",
  value: "drop-this",
};
const builtTagChange = createFeatureFieldChange<MixedSchema, "tag">({
  kind: "options",
  fieldId: "tag",
  dataKey: "tag",
  value: "runtime-tag",
});
const builtPromptChange = createFeatureFieldChange<MixedSchema, "prompt">({
  kind: "textarea",
  fieldId: "prompt",
  dataKey: "prompt",
  value: "drop-this",
});
const builtAudioChange = createFeatureFieldChange<MixedSchema, "audio">({
  kind: "audio-asset",
  fieldId: "audio",
  dataKey: "audio",
  value: { assetId: "a1" },
});
const builtVideoChange = createFeatureFieldChange<MixedSchema, "video">({
  kind: "video-asset",
  fieldId: "video",
  dataKey: "video",
  value: { assetId: "v1" },
});

const t2vPromptHasNoValue: "value" extends keyof TextToVideoSchema["prompt"]
  ? never
  : true = true;
const i2vPromptHasNoValue: "value" extends keyof ImageToVideoSchema["prompt"]
  ? never
  : true = true;

const homeSchemaMapMatch: Expect<
  Equal<keyof FeatureSchemaMap, HomeFeatureId>
> = true;
const t2vStoreValuesMatch: Expect<
  Equal<
    FeatureValues<FeatureSchemaMap["text-to-video"]>,
    FeatureValues<TextToVideoSchema>
  >
> = true;
const i2vStoreValuesMatch: Expect<
  Equal<
    FeatureValues<FeatureSchemaMap["image-to-video"]>,
    FeatureValues<ImageToVideoSchema>
  >
> = true;
const a2vStoreValuesMatch: Expect<
  Equal<
    FeatureValues<FeatureSchemaMap["audio-to-video"]>,
    FeatureValues<AudioToVideoSchema>
  >
> = true;
const a2vPromptHasNoValue: "value" extends keyof AudioToVideoSchema["prompt"]
  ? never
  : true = true;
const retakeStoreValuesMatch: Expect<
  Equal<
    FeatureValues<FeatureSchemaMap["retake"]>,
    FeatureValues<RetakeSchema>
  >
> = true;
const extendStoreValuesMatch: Expect<
  Equal<
    FeatureValues<FeatureSchemaMap["extend"]>,
    FeatureValues<ExtendSchema>
  >
> = true;

// @ts-expect-error options keys are not textarea fields
const tagAsTextareaKey: TextareaFieldKey<MixedSchema> = "tag";
// @ts-expect-error textarea keys are not option fields
const promptAsOptionKey: OptionsFieldKey<MixedSchema> = "prompt";
// @ts-expect-error image-asset keys are not textarea fields
const imageAsTextareaKey: TextareaFieldKey<MixedSchema> = "image";
// @ts-expect-error audio-asset keys are not textarea fields
const audioAsTextareaKey: TextareaFieldKey<MixedSchema> = "audio";
// @ts-expect-error audio-asset keys are not image-asset fields
const audioAsImageKey: ImageAssetFieldKey<MixedSchema> = "audio";
// @ts-expect-error image-asset keys are not audio-asset fields
const imageAsAudioKey: AudioAssetFieldKey<MixedSchema> = "image";
// @ts-expect-error video-asset keys are not textarea fields
const videoAsTextareaKey: TextareaFieldKey<MixedSchema> = "video";
// @ts-expect-error video-asset keys are not image-asset fields
const videoAsImageKey: ImageAssetFieldKey<MixedSchema> = "video";
// @ts-expect-error audio-asset keys are not video-asset fields
const audioAsVideoKey: VideoAssetFieldKey<MixedSchema> = "audio";
// @ts-expect-error options kind cannot be paired with a textarea key
const kindKeyMismatch: FeatureFieldChange<MixedSchema> = {
  kind: "options",
  fieldId: "prompt",
  dataKey: "prompt",
  value: "A dog running on a beach",
};
// @ts-expect-error textarea kind cannot be paired with an options key
const textareaOnTag: FeatureFieldChange<MixedSchema> = {
  kind: "textarea",
  fieldId: "tag",
  dataKey: "tag",
  value: "literally-any-string",
};
// @ts-expect-error textarea values are raw strings, not numbers
const invalidTextareaValue: FeatureFieldChange<MixedSchema> = {
  kind: "textarea",
  fieldId: "prompt",
  dataKey: "prompt",
  value: 42,
};
// @ts-expect-error option values must match the selected field
const invalidOptionValue: FeatureFieldChange<MixedSchema> = {
  kind: "options",
  fieldId: "style",
  dataKey: "style",
  value: "neon",
};

createFeatureFieldChange<MixedSchema, "tag">({
  // @ts-expect-error helper rejects a textarea kind for an options key
  kind: "textarea",
  fieldId: "tag",
  dataKey: "tag",
  value: "runtime-tag",
});
createFeatureFieldChange<MixedSchema, "prompt">({
  kind: "textarea",
  fieldId: "prompt",
  dataKey: "prompt",
  // @ts-expect-error helper rejects a numeric value for a textarea key
  value: 42,
});
createFeatureFieldChange<MixedSchema, "audio">({
  // @ts-expect-error helper rejects an image-asset kind for an audio-asset key
  kind: "image-asset",
  fieldId: "audio",
  dataKey: "audio",
  value: { assetId: "a1" },
});
readImageAssetFieldValue<
  MixedSchema,
  // @ts-expect-error helper rejects a textarea key for an image-asset read
  "prompt"
>(mixedImageValues, "prompt");
readAudioAssetFieldValue<
  MixedSchema,
  // @ts-expect-error helper rejects a textarea key for an audio-asset read
  "prompt"
>(mixedImageValues, "prompt");
readAudioAssetFieldValue<
  MixedSchema,
  // @ts-expect-error helper rejects an image-asset key for an audio-asset read
  "image"
>(mixedImageValues, "image");

const durableAssetRefRejectsPath: FeatureFieldChange<MixedSchema> = {
  kind: "image-asset",
  fieldId: "image",
  dataKey: "image",
  // @ts-expect-error durable asset refs cannot retain filesystem paths
  value: { assetId: "uploaded-1", path: "/tmp/input.jpg" },
};
const durableAssetRefRejectsObjectUrl: FeatureFieldChange<MixedSchema> = {
  kind: "image-asset",
  fieldId: "image",
  dataKey: "image",
  // @ts-expect-error durable asset refs cannot retain object URLs
  value: { assetId: "uploaded-1", previewUrl: "blob:https://ltx.local/1" },
};
const audioDurableAssetRefRejectsPath: FeatureFieldChange<MixedSchema> = {
  kind: "audio-asset",
  fieldId: "audio",
  dataKey: "audio",
  // @ts-expect-error durable asset refs cannot retain filesystem paths
  value: { assetId: "uploaded-1", path: "/tmp/input.wav" },
};
const audioDurableAssetRefRejectsObjectUrl: FeatureFieldChange<MixedSchema> = {
  kind: "audio-asset",
  fieldId: "audio",
  dataKey: "audio",
  // @ts-expect-error durable asset refs cannot retain object URLs
  value: { assetId: "uploaded-1", previewUrl: "blob:https://ltx.local/1" },
};

void [
  rawTextareaAllowed,
  declaredStringTextareaForbidden,
  constrainedTextareaForbidden,
  fieldKindMatchesSpec,
  mixedSchemaAllowed,
  mixedValuesMatch,
  mixedImageValue,
  mixedImageValueType,
  mixedAudioValue,
  mixedAudioValueType,
  mixedVideoValue,
  mixedVideoValueType,
  rawDomTextareaChange,
  builtTagChange,
  builtPromptChange,
  builtAudioChange,
  builtVideoChange,
  t2vPromptHasNoValue,
  i2vPromptHasNoValue,
  homeSchemaMapMatch,
  t2vStoreValuesMatch,
  i2vStoreValuesMatch,
  a2vStoreValuesMatch,
  a2vPromptHasNoValue,
  retakeStoreValuesMatch,
  extendStoreValuesMatch,
  tagAsTextareaKey,
  promptAsOptionKey,
  imageAsTextareaKey,
  audioAsTextareaKey,
  audioAsImageKey,
  imageAsAudioKey,
  videoAsTextareaKey,
  videoAsImageKey,
  audioAsVideoKey,
  kindKeyMismatch,
  textareaOnTag,
  invalidTextareaValue,
  invalidOptionValue,
  durableAssetRefRejectsPath,
  durableAssetRefRejectsObjectUrl,
  audioDurableAssetRefRejectsPath,
  audioDurableAssetRefRejectsObjectUrl,
];
