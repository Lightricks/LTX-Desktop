import {
  RETAKE_SEED_REVISION,
  isPackagedRetakeSeedName,
  planRetakeSeed,
  readAppliedRetakeSeedRevision,
  writeAppliedRetakeSeedRevision,
} from "../../hooks/retakeSeed.ts";
import { useCreateRetake } from "../../hooks/useCreateRetake";
import { useRetakeSeed } from "../../hooks/useRetakeSeed";
import { roundPreflightSeconds } from "../../lib/editPreflight";
import { roundDurationSeconds } from "../../lib/roundDuration";
import {
  RETAKE_DEFAULTS,
  RETAKE_MIN_DURATION_SECONDS,
  RETAKE_VIDEO_RANGE,
  retakeDefinition,
  seedRetakeValues,
  type RetakeValues,
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
      selectPreflightInput={(values: RetakeValues | undefined) => {
        // Wire duration, not the 0.5 grid: the advisory must verdict the
        // duration create sends, or boundary cells disagree with the gate.
        const duration = roundDurationSeconds(
          values?.duration ?? RETAKE_DEFAULTS.duration,
        );
        return {
          startTime: roundPreflightSeconds(
            values?.startTime ?? RETAKE_DEFAULTS.startTime,
          ),
          mode: "end",
          durations:
            duration === RETAKE_MIN_DURATION_SECONDS
              ? [RETAKE_MIN_DURATION_SECONDS]
              : [RETAKE_MIN_DURATION_SECONDS, duration],
        };
      }}
    />
  );
}
