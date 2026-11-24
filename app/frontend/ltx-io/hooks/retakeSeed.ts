import { fromGeneration } from "../screens/Feature/definitions/retake.ts";
import {
  RETAKE_SEED_FILENAME,
  RETAKE_SEED_PREVIOUS_FILENAMES,
  RETAKE_SEED_REVISION,
} from "../../../shared/explore-seed-filenames.ts";
import {
  PACKAGED_SEED_CACHE,
  isPackagedSeedName,
  planPackagedSeed,
  readAppliedSeedRevision,
  writeAppliedSeedRevision,
  type PackagedSeedPlan,
} from "./packagedSeed.ts";

export type RetakeSeedPlan = PackagedSeedPlan;

export const RETAKE_SEED_CACHE = PACKAGED_SEED_CACHE;

export { RETAKE_SEED_REVISION };

export const RETAKE_SEED_REVISION_STORAGE_KEY = "ltx.retake-seed-revision";

const RETAKE_SEED_NAMES = [
  RETAKE_SEED_FILENAME,
  ...RETAKE_SEED_PREVIOUS_FILENAMES,
] as const;

export function isPackagedRetakeSeedName(
  name: string | null | undefined,
): boolean {
  return isPackagedSeedName(name, RETAKE_SEED_NAMES);
}

export function readAppliedRetakeSeedRevision(): number {
  return readAppliedSeedRevision(RETAKE_SEED_REVISION_STORAGE_KEY);
}

export function writeAppliedRetakeSeedRevision(revision: number): void {
  writeAppliedSeedRevision(RETAKE_SEED_REVISION_STORAGE_KEY, revision);
}

export function planRetakeSeed(input: {
  hasStoredValues: boolean;
  generationsReady: boolean;
  generationsFailed: boolean;
  lastGenerationSpec: unknown;
  hydratedVideoName: string | null;
  hydratedVideoPending: boolean;
  appliedSeedRevision: number;
  packagedSeedRevision: number;
}): RetakeSeedPlan {
  return planPackagedSeed({
    ...input,
    fromGeneration,
    hydratedName: input.hydratedVideoName,
    hydratedPending: input.hydratedVideoPending,
    packagedFilenames: RETAKE_SEED_NAMES,
  });
}
