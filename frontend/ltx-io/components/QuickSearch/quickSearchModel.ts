import { CUSTOM_LORA_HOME_FEATURE } from "../../../lib/custom-lora.ts";
import {
  HOME_FEATURES,
  HOME_SIDEBAR_FEATURED_FEATURES,
  type HomeFeatureDefinition,
  type HomeFeatureFlags,
  type HomeFeatureId,
  getHomeFeature,
  isHomeFeatureOffered,
  offeredHomeFeatureSections,
} from "../../../lib/home-features.ts";
import { HOME_LORA_GALLERY_SECTIONS } from "../../../lib/home-lora-gallery.ts";
import { getLoraRecipe, isLoraRecipeId } from "../../../lib/lora-recipes.ts";
import { customLoraRunPath, paths } from "../../../paths.ts";
import type { CustomLoraItem } from "../../screens/Feature/definitions/customLora.ts";

export type QuickSearchPageDestination = {
  kind: "page";
  id: string;
  title: string;
  description: string;
  path: string;
};

export type QuickSearchFeatureRow = {
  kind: "feature";
  id: string;
  feature: HomeFeatureDefinition;
  section?: string;
  sectionId?: string;
};

export type QuickSearchPageRow = {
  kind: "page";
  id: string;
  destination: QuickSearchPageDestination;
  section?: string;
  sectionId?: string;
};

/** A LoRA the user saved. Its row opens the page of that LoRA. */
export type QuickSearchCustomLora = Pick<
  CustomLoraItem,
  "id" | "name" | "source" | "page_url" | "description" | "trigger" | "cover_url"
>;

export type QuickSearchCustomLoraRow = {
  kind: "custom-lora";
  id: string;
  lora: QuickSearchCustomLora;
  path: string;
  section?: string;
  sectionId?: string;
};

export type QuickSearchRow =
  | QuickSearchFeatureRow
  | QuickSearchPageRow
  | QuickSearchCustomLoraRow;

export type QuickSearchBrowseSection = {
  id: string;
  label: string;
  features: readonly HomeFeatureDefinition[];
  /** Saved LoRAs of the section. They follow the features. */
  customLoras?: readonly QuickSearchCustomLora[];
};

function customLoraRow(
  lora: QuickSearchCustomLora,
  section?: { id?: string; label: string },
): QuickSearchCustomLoraRow {
  return {
    kind: "custom-lora",
    id: `custom-lora-saved-${lora.id}`,
    lora,
    path: customLoraRunPath(lora.id),
    section: section?.label,
    sectionId: section?.id,
  };
}

const PAGE_DESTINATIONS: readonly QuickSearchPageDestination[] = [
  {
    kind: "page",
    id: "page-home",
    title: "Home",
    description: "Browse workflows and inspiration",
    path: paths.home,
  },
  {
    kind: "page",
    id: "page-assets",
    title: "Assets",
    description: "Browse generated videos and uploads",
    path: paths.assets,
  },
];

const FLOW_KIND_ALIASES: Record<string, readonly string[]> = {
  t2v: ["text", "prompt", "text to video"],
  i2v: ["image", "photo", "still", "image to video"],
  a2v: ["audio", "sound", "speech", "audio to video"],
  retake: ["video", "edit", "replace", "retake"],
  extend: ["video", "continue", "longer", "extend"],
};

function galleryBrowseSectionId(heading: string): string {
  return heading
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "");
}

function featureSearchAliases(feature: HomeFeatureDefinition): string {
  if (isLoraRecipeId(feature.id)) {
    const recipe = getLoraRecipe(feature.id);
    return [
      ...(FLOW_KIND_ALIASES[recipe.mode] ?? []),
      "lora",
      recipe.catalogId,
      recipe.listing.categories,
      recipe.listing.typeLabel,
    ].join(" ");
  }
  if (feature.fetcherTool != null) {
    return (FLOW_KIND_ALIASES[feature.fetcherTool] ?? []).join(" ");
  }
  return "";
}

export function tokenizeQuickSearchQuery(query: string): string[] {
  return query
    .toLowerCase()
    .split(/[^a-z0-9]+/i)
    .filter(Boolean);
}

function scoreQuickSearchMatch(
  tokens: string[],
  fields: { title: string; aliases: string; description: string; haystack: string },
  options: { requireStrongMatch?: boolean } = {},
): number | null {
  const title = fields.title.toLowerCase();
  const aliases = fields.aliases.toLowerCase();
  const description = fields.description.toLowerCase();
  const haystack = fields.haystack.toLowerCase();

  for (const token of tokens) {
    if (!haystack.includes(token)) return null;
  }

  let score = 0;
  let strongMatch = false;
  for (const token of tokens) {
    if (title.includes(token)) {
      score += 30;
      strongMatch = true;
    } else if (aliases.includes(token)) {
      score += 18;
      strongMatch = true;
    } else if (description.includes(token)) {
      score += 10;
    } else {
      score += 4;
    }
  }

  if (options.requireStrongMatch && !strongMatch) return null;
  return score;
}

function featureListingFields(feature: HomeFeatureDefinition): {
  blurb: string;
  categories: string;
  typeLabel: string;
} {
  if ("listing" in feature && feature.listing) {
    return feature.listing;
  }
  if (isLoraRecipeId(feature.id)) {
    const { blurb, categories, typeLabel } = getLoraRecipe(feature.id).listing;
    return { blurb, categories, typeLabel };
  }
  return { blurb: "", categories: "", typeLabel: "" };
}

function featureHaystack(feature: HomeFeatureDefinition): string {
  const listing = featureListingFields(feature);
  return [
    feature.id,
    feature.title,
    feature.description,
    listing.blurb,
    listing.categories,
    listing.typeLabel,
    featureSearchAliases(feature),
  ]
    .filter(Boolean)
    .join(" ");
}

export function searchQuickSearchFeatures(
  query: string,
  limit = 24,
  featureFlags: HomeFeatureFlags = {},
): HomeFeatureDefinition[] {
  const tokens = tokenizeQuickSearchQuery(query);
  if (tokens.length === 0) return [];

  return HOME_FEATURES.flatMap((feature) => {
    if (!isHomeFeatureOffered(feature.id, featureFlags)) return [];
    const listing = featureListingFields(feature);
    const aliases = featureSearchAliases(feature);
    const score = scoreQuickSearchMatch(tokens, {
      title: feature.title,
      aliases,
      description: [feature.description, listing.blurb, listing.categories]
        .filter(Boolean)
        .join(" "),
      haystack: featureHaystack(feature),
    });
    if (score == null) return [];
    return [{ feature, score }];
  })
    .sort((a, b) => {
      if (b.score !== a.score) return b.score - a.score;
      return a.feature.title.localeCompare(b.feature.title, undefined, {
        sensitivity: "base",
      });
    })
    .slice(0, limit)
    .map((hit) => hit.feature);
}

export function searchQuickSearchPages(
  query: string,
  limit = 4,
): QuickSearchPageDestination[] {
  const tokens = tokenizeQuickSearchQuery(query);
  if (tokens.length === 0) return [];

  return PAGE_DESTINATIONS.flatMap((destination) => {
    const score = scoreQuickSearchMatch(
      tokens,
      {
        title: destination.title,
        aliases: "",
        description: destination.description,
        haystack: `${destination.title} ${destination.description}`,
      },
      { requireStrongMatch: true },
    );
    if (score == null) return [];
    return [{ destination, score }];
  })
    .sort((a, b) => b.score - a.score)
    .slice(0, limit)
    .map((hit) => hit.destination);
}

export function buildQuickSearchBrowseSections(
  recentFeatureIds: readonly HomeFeatureId[],
  featureFlags: HomeFeatureFlags = {},
  customLoras: readonly QuickSearchCustomLora[] = [],
): QuickSearchBrowseSection[] {
  const recentIds = new Set(recentFeatureIds);
  const featured = HOME_SIDEBAR_FEATURED_FEATURES.filter(
    (feature) => !recentIds.has(feature.id),
  );
  const recent = recentFeatureIds
    // Saved LoRAs have their own rows. The add page is an entry of the sidebar only.
    .filter((id) => id !== CUSTOM_LORA_HOME_FEATURE.id)
    .map((id) => getHomeFeature(id));
  const sections: QuickSearchBrowseSection[] = [];
  if (featured.length > 0) {
    sections.push({ id: "featured", label: "Featured", features: featured });
  }
  if (customLoras.length > 0) {
    sections.push({ id: "custom-lora", label: "My LoRAs", features: [], customLoras });
  }
  if (recent.length > 0) {
    sections.push({ id: "recent", label: "Recent workflows", features: recent });
  }

  for (const section of offeredHomeFeatureSections(featureFlags)) {
    sections.push({
      id: section.id,
      label: section.heading,
      features: section.features,
    });
  }

  for (const gallerySection of HOME_LORA_GALLERY_SECTIONS) {
    sections.push({
      id: galleryBrowseSectionId(gallerySection.heading),
      label: gallerySection.heading,
      features: gallerySection.featureIds.map((featureId) => getHomeFeature(featureId)),
    });
  }

  return sections;
}

export function browseSectionsToRows(
  sections: readonly QuickSearchBrowseSection[],
): QuickSearchRow[] {
  return sections.flatMap((section): QuickSearchRow[] => [
    ...section.features.map((feature) => ({
      kind: "feature" as const,
      id: `${section.id}-${feature.id}`,
      feature,
      section: section.label,
      sectionId: section.id,
    })),
    ...(section.customLoras ?? []).map((lora) => customLoraRow(lora, section)),
  ]);
}

/** The line under the name of a saved LoRA. A saved LoRA runs with or without a start image. */
export const CUSTOM_LORA_LABEL = "Custom LoRA";

const CUSTOM_LORA_ALIASES = [
  "lora",
  "custom",
  ...(FLOW_KIND_ALIASES.t2v ?? []),
  ...(FLOW_KIND_ALIASES.i2v ?? []),
].join(" ");

function searchQuickSearchCustomLoras(
  tokens: string[],
  customLoras: readonly QuickSearchCustomLora[],
): QuickSearchCustomLoraRow[] {
  return customLoras
    .flatMap((lora) => {
      const score = scoreQuickSearchMatch(tokens, {
        title: lora.name,
        aliases: CUSTOM_LORA_ALIASES,
        description: "",
        haystack: `${lora.name} ${CUSTOM_LORA_ALIASES}`,
      });
      return score == null ? [] : [{ lora, score }];
    })
    .sort((a, b) => b.score - a.score)
    .map((hit) => customLoraRow(hit.lora, { label: "Tools" }));
}

export function searchQuickSearchRows(
  query: string,
  featureFlags: HomeFeatureFlags = {},
  customLoras: readonly QuickSearchCustomLora[] = [],
): QuickSearchRow[] {
  const trimmed = query.trim();
  if (!trimmed) return [];
  const featureRows: QuickSearchFeatureRow[] = searchQuickSearchFeatures(
    trimmed,
    undefined,
    featureFlags,
  ).map((feature) => ({
    kind: "feature",
    id: feature.id,
    feature,
    section: "Tools",
  }));
  const pageRows: QuickSearchPageRow[] = searchQuickSearchPages(trimmed).map(
    (destination) => ({
      kind: "page",
      id: destination.id,
      destination,
      section: "Pages",
    }),
  );
  const loraRows = searchQuickSearchCustomLoras(
    tokenizeQuickSearchQuery(trimmed),
    customLoras,
  );
  return [...featureRows, ...loraRows, ...pageRows];
}

export function defaultQuickSearchActiveIndex(rows: readonly QuickSearchRow[]): number {
  const featureIndex = rows.findIndex((row) => row.kind !== "page");
  if (featureIndex >= 0) return featureIndex;
  return rows.length > 0 ? 0 : -1;
}

export function quickSearchOptionId(rowId: string): string {
  return `ltxio-quick-search-option-${rowId}`;
}
