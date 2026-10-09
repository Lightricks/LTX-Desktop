import {
  RETAKE_SEED_REVISION,
  isPackagedRetakeSeedName,
  planRetakeSeed,
  readAppliedRetakeSeedRevision,
  writeAppliedRetakeSeedRevision,
} from "../../hooks/retakeSeed.ts";
import { useCreateRetake } from "../../hooks/useCreateRetake";
import { useRetakeSeed } from "../../hooks/useRetakeSeed";
import {
  RETAKE_VIDEO_RANGE,
  retakeDefinition,
  seedRetakeValues,
} from "./definitions/retake";
import { retakeFormPresentation } from "./RetakeFormPresentation";
import { RetakeExampleEmpty } from "./results/RetakeExampleEmpty";
import { VideoToVideoFeatureScreen } from "./VideoToVideoFeatureScreen";

export function RetakeFeatureScreen() {
  const createGeneration = useCreateRetake();
  return (
    <VideoToVideoFeatureScreen
      featureId="retake"
      definition={retakeDefinition}
      presentation={retakeFormPresentation}
      EmptyStateAdapter={RetakeExampleEmpty}
      createGeneration={createGeneration}
      useSeed={useRetakeSeed}
      planSeed={(input) =>
        planRetakeSeed({
          ...input,
          appliedSeedRevision: readAppliedRetakeSeedRevision(),
          packagedSeedRevision: RETAKE_SEED_REVISION,
        })
      }
      isPackagedSeedName={isPackagedRetakeSeedName}
      writeAppliedRevision={() =>
        writeAppliedRetakeSeedRevision(RETAKE_SEED_REVISION)
      }
      seedValues={seedRetakeValues}
      videoRange={RETAKE_VIDEO_RANGE}
    />
  );
}
