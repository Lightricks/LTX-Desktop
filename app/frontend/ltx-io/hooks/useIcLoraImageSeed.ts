import { useEffect } from "react";

import type { IcLoraRecipe } from "@/lib/ic-lora-recipes";

import { readGenerationInputs } from "../screens/Feature/definitions/videoFeature";
import { useFeatureFormStore } from "../stores/featureFormStore";
import {
  planIcLoraImageSeed,
  readAppliedIcLoraImageRevision,
  writeAppliedIcLoraImageRevision,
} from "./icLoraRecipeSeed";
import { useAsset } from "./useAsset";
import { useGenerations } from "./useGenerations";
import { usePackagedSeedAsset } from "./usePackagedSeedAsset";

/**
 * Loads the packaged look image of a recipe when the planner asks for it. The
 * plan follows the video seed: an applied revision is kept in local storage, and
 * the stored image is replaced only when it is an older packaged copy. A recipe
 * with no packaged image keeps the query disabled, so the hook call stays
 * unconditional.
 */
export function useIcLoraImageSeed(recipe: IcLoraRecipe) {
  const spec = recipe.referenceImage?.seed ?? null;
  const hasStoredValues = useFeatureFormStore((state) => state.hasValues(recipe.id));
  const getValues = useFeatureFormStore((state) => state.getValues);
  const setValues = useFeatureFormStore((state) => state.setValues);
  const storedImageId = useFeatureFormStore(
    (state) => state.getValues(recipe.id)?.image?.assetId ?? null,
  );
  const generationsQuery = useGenerations(recipe.id);
  const lastGenerationSpec = generationsQuery.data?.[0]?.spec;
  const generationImageId = hasStoredValues
    ? null
    : (readGenerationInputs(lastGenerationSpec, "image")?.image?.assetId ?? null);
  const imageAssetId = storedImageId ?? generationImageId;
  const imageAsset = useAsset(imageAssetId);
  const plan = planIcLoraImageSeed(recipe, {
    hasStoredValues,
    storedImageId,
    generationsReady: generationsQuery.isSuccess,
    generationsFailed: generationsQuery.isError,
    lastGenerationSpec,
    hydratedImageName: imageAsset.data?.name ?? null,
    hydratedImagePending: imageAssetId != null && imageAsset.isPending,
    appliedSeedRevision: readAppliedIcLoraImageRevision(recipe),
  });
  const query = usePackagedSeedAsset(spec?.kind ?? recipe.seed.kind, {
    enabled: plan === "ingest",
    revision: spec?.revision,
    media: "image",
  });
  const seed = spec == null ? null : (query.data ?? null);

  useEffect(() => {
    if (plan !== "ingest" || seed == null) return;
    const current = getValues(recipe.id);
    if (current != null) {
      const currentImageId = current.image?.assetId ?? null;
      // The store changed after this render. The next render plans again.
      if (currentImageId !== storedImageId) return;
      if (currentImageId !== seed.assetId) {
        setValues(recipe.id, { ...current, image: seed });
      }
    }
    writeAppliedIcLoraImageRevision(recipe);
  }, [getValues, plan, recipe, seed, setValues, storedImageId]);

  return {
    data: seed,
    // The error matters only while the field is empty. A chosen image ends it.
    error: spec == null || storedImageId != null ? null : query.error,
    /** True once the first stored values can include the image. */
    settled:
      plan === "skip" ||
      (plan === "ingest" && (query.isSuccess || query.isError)) ||
      (plan === "wait" && hasStoredValues),
  };
}
