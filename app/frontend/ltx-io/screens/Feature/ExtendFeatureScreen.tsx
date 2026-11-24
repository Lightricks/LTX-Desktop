import {
  EXTEND_SEED_REVISION,
  isPackagedExtendSeedName,
  planExtendSeed,
  readAppliedExtendSeedRevision,
  writeAppliedExtendSeedRevision,
} from "../../hooks/extendSeed.ts";
import { useCreateExtend } from "../../hooks/useCreateExtend";
import { useExtendSeed } from "../../hooks/useExtendSeed";
import {
  EXTEND_DEFAULTS,
  EXTEND_DURATION_PRESETS,
  extendDefinition,
  seedExtendValues,
  type ExtendValues,
} from "./definitions/extend";
import { extendFormPresentation } from "./ExtendFormPresentation";
import { ExtendExampleEmpty } from "./results/ExtendExampleEmpty";
import { VideoToVideoFeatureScreen } from "./VideoToVideoFeatureScreen";

export function ExtendFeatureScreen() {
  const createGeneration = useCreateExtend();
  return (
    <VideoToVideoFeatureScreen
      featureId="extend"
      definition={extendDefinition}
      presentation={extendFormPresentation}
      EmptyStateAdapter={ExtendExampleEmpty}
      createGeneration={createGeneration}
      useSeed={useExtendSeed}
      planSeed={(input) =>
        planExtendSeed({
          ...input,
          appliedSeedRevision: readAppliedExtendSeedRevision(),
          packagedSeedRevision: EXTEND_SEED_REVISION,
        })
      }
      isPackagedSeedName={isPackagedExtendSeedName}
      writeAppliedRevision={() =>
        writeAppliedExtendSeedRevision(EXTEND_SEED_REVISION)
      }
      seedValues={seedExtendValues}
      selectPreflightInput={(values: ExtendValues | undefined) => ({
        startTime: 0,
        mode: values?.mode ?? EXTEND_DEFAULTS.mode,
        durations: [...EXTEND_DURATION_PRESETS],
      })}
    />
  );
}
