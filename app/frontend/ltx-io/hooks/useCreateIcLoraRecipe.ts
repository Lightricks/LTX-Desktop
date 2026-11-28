import type { components } from "@/generated/backend-openapi";
import type { IcLoraRecipeId } from "@/lib/ic-lora-recipes";

import { useCreateFeatureGeneration } from "./useCreateFeatureGeneration";

type CreateIcLoraRecipeBody =
  | components["schemas"]["CreateIcLoraRecipeRequest"]
  | components["schemas"]["CreateIcLoraSheetRequest"]
  | components["schemas"]["CreateIcLoraTrackRequest"];

export function useCreateIcLoraRecipe(recipeId: IcLoraRecipeId) {
  return useCreateFeatureGeneration((api, body: CreateIcLoraRecipeBody) =>
    api.createIcLoraRecipe(recipeId, body),
  );
}
