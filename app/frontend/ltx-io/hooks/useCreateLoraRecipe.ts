import type { components } from "@/generated/backend-openapi";
import { useCreateFeatureGeneration } from "./useCreateFeatureGeneration";

type CreateLoraRecipeRequest = components["schemas"]["CreateLoraRecipeRequest"];

export function useCreateLoraRecipe(recipeId: string) {
  return useCreateFeatureGeneration((api, body: CreateLoraRecipeRequest) =>
    api.createLoraRecipe(recipeId, body),
  );
}
