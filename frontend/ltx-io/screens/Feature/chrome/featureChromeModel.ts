import {
  getHomeFeature,
  type HomeFeatureDefinition,
  type HomeFeatureId,
  type HomeFeatureListing,
} from "../../../../lib/home-features.ts";
import {
  getLoraRecipe,
  isLoraRecipeId,
  type LoraRecipeListing,
} from "../../../../lib/lora-recipes.ts";
import { paths } from "../../../../paths.ts";

import type { FeatureCompare } from "./featureCompare.ts";

/** Exact LTX.io community-LoRA disclaimer from WorkflowDetail.tsx. */
export const COMMUNITY_FEATURE_DISCLAIMER =
  "This LoRA was created by a community member, not LTX. LTX does not endorse, control, or take responsibility for community LoRAs. Your use of this LoRA is subject to any applicable IP rights of the creator, and you are solely responsible for obtaining any required licenses or permissions. Any claims should be directed to the LoRA's creator.";

export type FeatureCrumb = {
  readonly id: string;
  readonly label: string;
  readonly to: string | null;
};

export type FeatureCredit = {
  readonly name: string;
  readonly affiliation: "ltx" | "community";
};

export type FeatureDetailsHero = {
  readonly posterUrl: string;
  /** Missing when Home media is poster-only (LoRA compact tiles). */
  readonly videoUrl: string | undefined;
  /** When set, the hero shows the example next to its input. */
  readonly compare?: FeatureCompare;
};

export type FeatureDetailsRow = {
  readonly id: string;
  readonly label: string;
  readonly value: string;
  readonly href: string | null;
};

export type FeatureDetailsExternalLink = {
  readonly label: string;
  readonly url: string;
};

export type FeatureDetails = {
  readonly featureId: HomeFeatureId;
  readonly title: string;
  readonly blurb: string;
  readonly credit: FeatureCredit;
  readonly rows: readonly FeatureDetailsRow[];
  readonly externalLink: FeatureDetailsExternalLink | null;
};

export function featureCrumbTrail(
  feature: HomeFeatureDefinition,
): readonly FeatureCrumb[] {
  return [
    { id: "home", label: "Home", to: paths.home },
    { id: feature.id, label: feature.title, to: null },
  ];
}

function categoriesRow(value: string): FeatureDetailsRow {
  return { id: "categories", label: "Categories", value, href: null };
}

function typeRow(value: string): FeatureDetailsRow {
  return { id: "type", label: "Type", value, href: null };
}

function firstClassListing(
  feature: HomeFeatureDefinition,
): HomeFeatureListing {
  if (!("listing" in feature)) {
    throw new Error(`Missing listing for Home feature: ${feature.id}`);
  }
  return feature.listing;
}

function recipeRows(listing: LoraRecipeListing): readonly FeatureDetailsRow[] {
  return [
    categoriesRow(listing.categories),
    typeRow(listing.typeLabel),
    {
      id: "base-model",
      label: "Base Model",
      value: listing.baseModel,
      href: null,
    },
    {
      id: "license",
      label: "License",
      value: listing.licenseName,
      href: listing.licenseUrl,
    },
  ];
}

export function buildFeatureDetails(featureId: HomeFeatureId): FeatureDetails {
  const feature = getHomeFeature(featureId);

  if (isLoraRecipeId(featureId)) {
    const listing = getLoraRecipe(featureId).listing;
    return {
      featureId,
      title: feature.title,
      blurb: listing.blurb,
      credit: {
        name: listing.authorName,
        affiliation: listing.affiliation,
      },
      rows: recipeRows(listing),
      externalLink: {
        label: "Hugging Face",
        url: listing.huggingfaceUrl,
      },
    };
  }

  const listing = firstClassListing(feature);
  return {
    featureId,
    title: feature.title,
    blurb: listing.blurb,
    credit: { name: "LTX Team", affiliation: "ltx" },
    rows: [categoriesRow(listing.categories), typeRow(listing.typeLabel)],
    externalLink: listing.huggingfaceUrl
      ? { label: "Hugging Face", url: listing.huggingfaceUrl }
      : null,
  };
}
