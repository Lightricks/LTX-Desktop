import type { ReactNode } from "react";

import { Button } from "@ds/Button/Button";

import type { ExploreAsset } from "@/lib/explore-contract";

import { useFeatureFlagEnabled } from "../../hooks/useFeatureFlagEnabled";
import { imageAspectForVideo } from "../../components/ImageGenerator/imageGeneratorRequest";
import { useImageGeneratorModal } from "../../components/ImageGenerator/useImageGeneratorModal";

const FRAME_FIELD_IDS: readonly string[] = ["startFrame", "endFrame"];

/**
 * Opens the image generator. The finished image fills the field.
 * It shows only when the `createWithAi` flag is on. Every frame and reference form reaches it through here.
 */
function CreateWithAiButton({
  useImageLabel,
  videoAspect,
  setAsset,
}: {
  useImageLabel: string;
  videoAspect: string;
  setAsset: (asset: ExploreAsset) => void;
}) {
  const openImageGenerator = useImageGeneratorModal();
  const enabled = useFeatureFlagEnabled("createWithAi");

  if (!enabled) return null;
  return (
    <Button
      appearance="neutral"
      hierarchy="secondary"
      size="sm"
      label="Create with AI"
      onClick={() =>
        openImageGenerator({
          initialAspect: imageAspectForVideo(videoAspect),
          onUseImage: setAsset,
          useImageLabel,
        })
      }
    />
  );
}

/** The `imageAssetAction` of a form whose start and end frames can be made with AI. */
export function frameImageAssetAction({
  fieldId,
  label,
  values,
  setAsset,
}: {
  fieldId: string;
  label: string;
  values: { aspectRatio: string };
  setAsset: (asset: ExploreAsset) => void;
}): ReactNode {
  if (!FRAME_FIELD_IDS.includes(fieldId)) return null;
  return (
    <CreateWithAiButton
      useImageLabel={`Use as ${label.toLowerCase()}`}
      videoAspect={values.aspectRatio}
      setAsset={setAsset}
    />
  );
}

/**
 * The `imageAssetAction` of a form whose one image is a reference, such as Motion Track.
 * The image is not a start or end frame, so the button says reference image.
 */
export function referenceImageAssetAction({
  values,
  setAsset,
}: {
  values: { aspectRatio: string };
  setAsset: (asset: ExploreAsset) => void;
}): ReactNode {
  return (
    <CreateWithAiButton
      useImageLabel="Use as reference image"
      videoAspect={values.aspectRatio}
      setAsset={setAsset}
    />
  );
}
