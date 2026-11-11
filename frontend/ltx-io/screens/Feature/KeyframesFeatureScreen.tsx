import { useMemo } from "react";
import { Navigate } from "react-router";

import { paths } from "@/paths";

import { KEYFRAMES_SEED_KINDS } from "../../../../shared/explore-seed-filenames";
import { useImageAssetSize } from "../../hooks/useAsset";
import { useCreateKeyframes } from "../../hooks/useCreateKeyframes";
import { useFeatureFlags } from "../../hooks/useFeatureFlags";
import { useGenerations } from "../../hooks/useGenerations";
import { planPackagedSeed } from "../../hooks/packagedSeed.ts";
import { usePackagedSeedAssets } from "../../hooks/usePackagedSeedAsset";
import { useVideoGenerationModelSpecs } from "../../hooks/useVideoGenerationModelSpecs";
import { useExploreRuntime } from "../../runtime/ExploreRuntime";
import { useFeatureFormStore } from "../../stores/featureFormStore";
import {
  earliestKeyframe,
  fromGeneration,
  keyframesDefinition,
  type KeyframesContext,
} from "./definitions/keyframes";
import type { KeyframesExampleSeed } from "./definitions/keyframesExample";
import { FeatureFormScreen } from "./FeatureFormScreen";
import { keyframesFormPresentation } from "./KeyframesFormPresentation";
import { KeyframesExampleEmpty } from "./results/KeyframesExampleEmpty";
import { ResultFrame } from "./results/ResultFrame";

export function KeyframesFeatureScreen() {
  const { api } = useExploreRuntime();
  const flagsQuery = useFeatureFlags(api);
  const specsQuery = useVideoGenerationModelSpecs();
  const createGeneration = useCreateKeyframes();
  // The earliest still sets the ratio that Auto runs at.
  const startFrameId = useFeatureFormStore(
    (state) =>
      earliestKeyframe(state.getValues("keyframes")?.keyframes ?? [])?.asset.assetId ?? null,
  );
  const startFrameSize = useImageAssetSize(startFrameId);
  // A first open starts from the packaged example. A stored form or an earlier run wins.
  const hasStoredValues = useFeatureFormStore((state) => state.hasValues("keyframes"));
  const generationsQuery = useGenerations("keyframes");
  const seedPlan = planPackagedSeed({
    hasStoredValues,
    generationsReady: generationsQuery.isSuccess,
    generationsFailed: generationsQuery.isError,
    lastGenerationSpec: generationsQuery.data?.[0]?.spec,
    fromGeneration,
  });
  const seedQuery = usePackagedSeedAssets(KEYFRAMES_SEED_KINDS, {
    media: "image",
    // No ingest for a user who cannot open the page.
    enabled: seedPlan === "ingest" && flagsQuery.data?.keyframes === true,
  });
  const seedAssets = seedQuery.data;
  const seed = useMemo<KeyframesExampleSeed | null>(
    () =>
      seedAssets
        ? (Object.fromEntries(
            KEYFRAMES_SEED_KINDS.map((kind, index) => [kind, seedAssets[index]]),
          ) as KeyframesExampleSeed)
        : null,
    [seedAssets],
  );
  const context = useMemo<KeyframesContext>(
    () => ({ specs: specsQuery.data, startFrameSize, seed }),
    [seed, specsQuery.data, startFrameSize],
  );
  const seedReady = seedPlan === "skip" || (seedPlan === "ingest" && seedQuery.isSuccess);
  const resourceError =
    specsQuery.error ??
    flagsQuery.error ??
    (seedPlan === "ingest" ? seedQuery.error : null) ??
    (hasStoredValues ? null : generationsQuery.error);

  // The feature has no page without its flag. The backend refuses the request too.
  if (flagsQuery.isSuccess && flagsQuery.data.keyframes !== true) {
    return <Navigate to={paths.home} replace />;
  }

  if (!seedReady && resourceError == null) return null;

  return (
    <FeatureFormScreen
      definition={keyframesDefinition}
      context={context}
      contextReady={specsQuery.isSuccess && flagsQuery.isSuccess && seedReady}
      generateEnabled={specsQuery.budgetSettingsReady}
      resourceError={resourceError}
      createGeneration={createGeneration}
      ResultFrameAdapter={ResultFrame}
      EmptyStateAdapter={KeyframesExampleEmpty}
      presentation={keyframesFormPresentation}
    />
  );
}
