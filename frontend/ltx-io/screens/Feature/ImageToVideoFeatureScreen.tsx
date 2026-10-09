import { useMemo } from "react";

import { planExploreImageToVideoSeed } from "../../hooks/exploreImageToVideoSeed.ts";
import { useCreateImageToVideo } from "../../hooks/useCreateImageToVideo";
import { useExploreImageToVideoSeed } from "../../hooks/useExploreImageToVideoSeed";
import { useGenerations } from "../../hooks/useGenerations";
import { useVideoGenerationModelSpecs } from "../../hooks/useVideoGenerationModelSpecs";
import { useFeatureFormStore } from "../../stores/featureFormStore";
import {
  imageToVideoDefinition,
  type ImageToVideoContext,
} from "./definitions/imageToVideo";
import { FeatureFormScreen } from "./FeatureFormScreen";
import { imageToVideoFormPresentation } from "./ImageToVideoFormPresentation";
import { ImageToVideoExampleEmpty } from "./results/ImageToVideoExampleEmpty";
import { ResultFrame } from "./results/ResultFrame";

export function ImageToVideoFeatureScreen() {
  const specsQuery = useVideoGenerationModelSpecs();
  const hasStoredValues = useFeatureFormStore((state) =>
    state.hasValues("image-to-video"),
  );
  const generationsQuery = useGenerations("image-to-video");
  const seedPlan = planExploreImageToVideoSeed({
    hasStoredValues,
    generationsReady: generationsQuery.isSuccess,
    generationsFailed: generationsQuery.isError,
    lastGenerationSpec: generationsQuery.data?.[0]?.spec,
  });
  const seedQuery = useExploreImageToVideoSeed({
    enabled: seedPlan === "ingest",
  });
  const createGeneration = useCreateImageToVideo();
  const context = useMemo<ImageToVideoContext>(
    () => ({ specs: specsQuery.data, seed: seedQuery.data ?? null }),
    [seedQuery.data, specsQuery.data],
  );

  const contextReady =
    specsQuery.isSuccess &&
    (seedPlan === "skip" || (seedPlan === "ingest" && seedQuery.isSuccess));
  const resourceError =
    specsQuery.error ??
    (seedPlan === "ingest" ? seedQuery.error : null) ??
    (hasStoredValues ? null : generationsQuery.error);

  if (!contextReady && resourceError == null) {
    return null;
  }

  return (
    <FeatureFormScreen
      definition={imageToVideoDefinition}
      context={context}
      contextReady={contextReady}
      resourceError={resourceError}
      createGeneration={createGeneration}
      ResultFrameAdapter={ResultFrame}
      EmptyStateAdapter={ImageToVideoExampleEmpty}
      presentation={imageToVideoFormPresentation}
    />
  );
}
