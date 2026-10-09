import type { components } from "@/generated/backend-openapi";
import type { IcLoraRecipeId } from "@/lib/ic-lora-recipes";

import { useCreateFeatureGeneration } from "./useCreateFeatureGeneration";

type CreateIcLoraRecipeRequest = components["schemas"]["CreateIcLoraRecipeRequest"];

export function useCreateIcLoraRecipe(recipeId: IcLoraRecipeId) {
  return useCreateFeatureGeneration((api, body: CreateIcLoraRecipeRequest) =>
    api.createIcLoraRecipe(recipeId, body),
  );
}
