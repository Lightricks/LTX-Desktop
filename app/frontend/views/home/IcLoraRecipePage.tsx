// eslint-disable-next-line no-restricted-imports
import { IcLoraRecipeFeatureScreen } from "@/ltx-io/screens/Feature/IcLoraRecipeFeatureScreen";
import type { IcLoraRecipeId } from "@/lib/ic-lora-recipes";

export function IcLoraRecipePage({ recipeId }: { recipeId: IcLoraRecipeId }) {
  return <IcLoraRecipeFeatureScreen recipeId={recipeId} />;
}

export function makeIcLoraRecipeScreen(recipeId: IcLoraRecipeId) {
  function IcLoraRecipeScreen() {
    return <IcLoraRecipePage recipeId={recipeId} />;
  }
  IcLoraRecipeScreen.displayName = `${recipeId}FeatureScreen`;
  return IcLoraRecipeScreen;
}
