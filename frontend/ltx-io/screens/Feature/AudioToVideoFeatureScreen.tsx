import { useMemo } from "react";

import { planExploreAudioToVideoSeed } from "../../hooks/exploreAudioToVideoSeed.ts";
import { useAsset } from "../../hooks/useAsset";
import { useCreateAudioToVideo } from "../../hooks/useCreateAudioToVideo";
import { useExploreAudioToVideoSeed } from "../../hooks/useExploreAudioToVideoSeed";
import { useGenerations } from "../../hooks/useGenerations";
import { useVideoGenerationModelSpecs } from "../../hooks/useVideoGenerationModelSpecs";
import { a2vTrimCapSeconds, A2V_TRIM_MIN_SECONDS } from "../../lib/a2vDurationPolicy.ts";
import { useFeatureFormStore } from "../../stores/featureFormStore";
import {
  audioToVideoDefinition,
  type AudioToVideoContext,
} from "./definitions/audioToVideo";
import { FeatureFormScreen } from "./FeatureFormScreen";
import { audioDurationSeconds } from "./fields/prepareDurationLimitedMediaImport";
import { audioToVideoFormPresentation } from "./AudioToVideoFormPresentation";
import { AudioToVideoExampleEmpty } from "./results/AudioToVideoExampleEmpty";
import { ResultFrame } from "./results/ResultFrame";

export function AudioToVideoFeatureScreen() {
  const specsQuery = useVideoGenerationModelSpecs();
  const hasStoredValues = useFeatureFormStore((state) =>
    state.hasValues("audio-to-video"),
  );
  const generationsQuery = useGenerations("audio-to-video");
  const seedPlan = planExploreAudioToVideoSeed({
    hasStoredValues,
    generationsReady: generationsQuery.isSuccess,
    generationsFailed: generationsQuery.isError,
    lastGenerationSpec: generationsQuery.data?.[0]?.spec,
  });
  const seedQuery = useExploreAudioToVideoSeed({
    enabled: seedPlan === "ingest",
  });
  const createGeneration = useCreateAudioToVideo();

  // Duration comes off the accepted (already trimmed) audio asset: it decides
  // the snapped generation duration, so Generate stays disabled until it loads.
  const audioAssetId = useFeatureFormStore(
    (state) => state.getValues("audio-to-video")?.audio?.assetId ?? null,
  );
  const selectedOffering = useFeatureFormStore(
    (state) => state.getValues("audio-to-video")?.model,
  );
  const audioAssetQuery = useAsset(audioAssetId);
  const audioDuration = audioAssetQuery.data
    ? audioDurationSeconds(audioAssetQuery.data)
    : null;
  const audioDurationPending = audioAssetId != null && audioAssetQuery.isPending;

  const context = useMemo<AudioToVideoContext>(
    () => ({
      specs: specsQuery.data,
      seed: seedQuery.data ?? null,
      audioDurationSeconds: audioDuration,
      audioDurationPending,
    }),
    [audioDuration, audioDurationPending, seedQuery.data, specsQuery.data],
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
      definition={audioToVideoDefinition}
      context={context}
      contextReady={contextReady}
      resourceError={resourceError}
      createGeneration={createGeneration}
      ResultFrameAdapter={ResultFrame}
      EmptyStateAdapter={AudioToVideoExampleEmpty}
      presentation={audioToVideoFormPresentation}
      audioTrimCapSeconds={a2vTrimCapSeconds(specsQuery.data, selectedOffering)}
      audioTrimMinSeconds={A2V_TRIM_MIN_SECONDS}
    />
  );
}
