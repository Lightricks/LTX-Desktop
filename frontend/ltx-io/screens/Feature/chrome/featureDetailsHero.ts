import { HOME_FEATURE_MEDIA } from "@/components/home/home-media";
import type { HomeFeatureId } from "@/lib/home-features";
import { isLoraRecipeId } from "@/lib/lora-recipes";

import { LORA_RECIPE_EXAMPLE_MEDIA } from "../loraRecipeExampleMedia";

import type { FeatureDetailsHero } from "./featureChromeModel";
import { resolveFeatureCompare } from "./featureCompare";

/**
 * Compact Home tiles may omit recipe `previewUrl`. FeatureDetails still needs
 * a clip, so recipes use the empty-state example media rather than Home media.
 * A feature with a compare view shows its example next to its input.
 */
export function resolveFeatureDetailsHero(
  featureId: HomeFeatureId,
): FeatureDetailsHero {
  const compare = resolveFeatureCompare(featureId);
  if (compare) {
    return {
      posterUrl: compare.result.posterUrl,
      videoUrl: compare.result.videoUrl,
      compare,
    };
  }

  if (isLoraRecipeId(featureId)) {
    const example = LORA_RECIPE_EXAMPLE_MEDIA[featureId];
    return {
      posterUrl: example.posterUrl,
      videoUrl: example.videoUrl,
    };
  }

  const media = HOME_FEATURE_MEDIA[featureId];
  return {
    posterUrl: media.posterUrl,
    videoUrl: media.previewUrl,
  };
}
