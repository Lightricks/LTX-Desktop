// eslint-disable-next-line no-restricted-imports
import { LoraFeatureScreen } from "@/ltx-io/screens/Feature/LoraFeatureScreen";
import type { LoraRecipeId } from "@/lib/lora-recipes";

export function LoraRecipePage({ recipeId }: { recipeId: LoraRecipeId }) {
  return <LoraFeatureScreen recipeId={recipeId} />;
}

export function makeLoraRecipeScreen(recipeId: LoraRecipeId) {
  function LoraRecipeFeatureScreen() {
    return <LoraRecipePage recipeId={recipeId} />;
  }
  LoraRecipeFeatureScreen.displayName = `${recipeId}FeatureScreen`;
  return LoraRecipeFeatureScreen;
}
