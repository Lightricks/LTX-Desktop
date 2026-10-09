export type PackagedSeedPlan = "wait" | "skip" | "ingest";

export const PACKAGED_SEED_CACHE = {
  staleTime: Number.POSITIVE_INFINITY,
  gcTime: Number.POSITIVE_INFINITY,
} as const;

export function isPackagedSeedName(
  name: string | null | undefined,
  filenames: readonly string[],
): boolean {
  if (name == null || name.length === 0) return false;
  return filenames.includes(name);
}

export function readAppliedSeedRevision(storageKey: string): number {
  try {
    const raw = window.localStorage.getItem(storageKey);
    const parsed = Number(raw);
    return Number.isInteger(parsed) ? parsed : 0;
  } catch {
    return 0;
  }
}

export function writeAppliedSeedRevision(storageKey: string, revision: number): void {
  try {
    window.localStorage.setItem(storageKey, String(revision));
  } catch {
    // Private mode / blocked storage: the next visit may re-ingest.
  }
}

export function planPackagedSeed(input: {
  hasStoredValues: boolean;
  generationsReady: boolean;
  generationsFailed: boolean;
  lastGenerationSpec: unknown;
  /** Non-null when the last generation holds the seeded input (a video, or a look image). */
  fromGeneration: (spec: unknown) => unknown;
  hydratedName?: string | null;
  hydratedPending?: boolean;
  appliedSeedRevision?: number;
  packagedSeedRevision?: number;
  packagedFilenames?: readonly string[];
}): PackagedSeedPlan {
  if (input.generationsFailed) {
    return "skip";
  }
  const hasDurableSeedInput =
    input.hasStoredValues || input.fromGeneration(input.lastGenerationSpec) != null;
  if (!hasDurableSeedInput) {
    if (!input.generationsReady) {
      return "wait";
    }
    return "ingest";
  }
  const tracksRevision =
    input.appliedSeedRevision != null && input.packagedSeedRevision != null;
  if (!tracksRevision) {
    return "skip";
  }
  if (input.hydratedPending) {
    if (input.appliedSeedRevision === input.packagedSeedRevision) {
      return "skip";
    }
    return "wait";
  }
  const seedIsStale =
    isPackagedSeedName(input.hydratedName, input.packagedFilenames ?? []) &&
    input.appliedSeedRevision !== input.packagedSeedRevision;
  if (seedIsStale) {
    return "ingest";
  }
  return "skip";
}
