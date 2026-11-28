import path from "node:path";

import {
  EXPLORE_LORA_RECIPE_SEED_DIR,
  EXPLORE_LORA_RECIPE_SEED_FILES,
  EXPLORE_LORA_RECIPE_SEED_IDS,
  type ExploreLoraRecipeSeedId,
} from "../shared/explore-seed-filenames.ts";

export {
  EXPLORE_LORA_RECIPE_SEED_DIR,
  EXPLORE_LORA_RECIPE_SEED_FILES,
  EXPLORE_LORA_RECIPE_SEED_IDS,
};

export function resolveExploreLoraRecipeSeedPath(input: {
  seedId: ExploreLoraRecipeSeedId;
  isPackaged: boolean;
  projectRoot: string;
  resourcesPath: string;
}): string {
  const filename = EXPLORE_LORA_RECIPE_SEED_FILES[input.seedId];
  const segments = [EXPLORE_LORA_RECIPE_SEED_DIR, filename] as const;
  if (input.isPackaged) {
    return path.join(input.resourcesPath, ...segments);
  }
  return path.join(input.projectRoot, "resources", ...segments);
}
