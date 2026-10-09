import { fromGeneration } from "../screens/Feature/definitions/extend.ts";
import {
  EXTEND_SEED_FILENAME,
  EXTEND_SEED_PREVIOUS_FILENAMES,
  EXTEND_SEED_REVISION,
} from "../../../shared/explore-seed-filenames.ts";
import {
  PACKAGED_SEED_CACHE,
  isPackagedSeedName,
  planPackagedSeed,
  readAppliedSeedRevision,
  writeAppliedSeedRevision,
  type PackagedSeedPlan,
} from "./packagedSeed.ts";

export type ExtendSeedPlan = PackagedSeedPlan;

export const EXTEND_SEED_CACHE = PACKAGED_SEED_CACHE;

export { EXTEND_SEED_REVISION };

export const EXTEND_SEED_REVISION_STORAGE_KEY = "ltx.extend-seed-revision";

const EXTEND_SEED_NAMES = [
  EXTEND_SEED_FILENAME,
  ...EXTEND_SEED_PREVIOUS_FILENAMES,
] as const;

export function isPackagedExtendSeedName(
  name: string | null | undefined,
): boolean {
  return isPackagedSeedName(name, EXTEND_SEED_NAMES);
}

export function readAppliedExtendSeedRevision(): number {
  return readAppliedSeedRevision(EXTEND_SEED_REVISION_STORAGE_KEY);
}

export function writeAppliedExtendSeedRevision(revision: number): void {
  writeAppliedSeedRevision(EXTEND_SEED_REVISION_STORAGE_KEY, revision);
}

export function planExtendSeed(input: {
  hasStoredValues: boolean;
  generationsReady: boolean;
  generationsFailed: boolean;
  lastGenerationSpec: unknown;
  hydratedVideoName?: string | null;
  hydratedVideoPending?: boolean;
  appliedSeedRevision?: number;
  packagedSeedRevision?: number;
}): ExtendSeedPlan {
  return planPackagedSeed({
    hasStoredValues: input.hasStoredValues,
    generationsReady: input.generationsReady,
    generationsFailed: input.generationsFailed,
    lastGenerationSpec: input.lastGenerationSpec,
    fromGeneration,
    hydratedName: input.hydratedVideoName ?? null,
    hydratedPending: input.hydratedVideoPending ?? false,
    appliedSeedRevision: input.appliedSeedRevision ?? EXTEND_SEED_REVISION,
    packagedSeedRevision: input.packagedSeedRevision ?? EXTEND_SEED_REVISION,
    packagedFilenames: EXTEND_SEED_NAMES,
  });
}
