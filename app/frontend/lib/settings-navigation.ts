
export type SettingsTabId =
  | "general"
  | "models"
  | "apiKeys"
  | "legacy"
  | "about";

/** In-page anchors on General (old hash links, plus the Assistant section). */
export type SettingsLegacyScrollAnchor = "promptEnhancer" | "textEncoding" | "assistant";

export type SettingsScrollTarget = SettingsTabId | SettingsLegacyScrollAnchor;

export type SettingsInitialReason = "geminiKeyRequired" | "ltxKeyRequired";

const SETTINGS_TAB_IDS = new Set<SettingsTabId>([
  "general",
  "models",
  "apiKeys",
  "legacy",
  "about",
]);

const LEGACY_HASH_TO_TAB: Record<string, SettingsTabId> = {
  promptEnhancer: "general",
  textEncoding: "general",
};

export function parseSettingsTabId(
  value: string | null,
): SettingsTabId | undefined {
  if (value === null) return undefined;
  const legacyTab = LEGACY_HASH_TO_TAB[value];
  if (legacyTab) return legacyTab;
  return SETTINGS_TAB_IDS.has(value as SettingsTabId)
    ? (value as SettingsTabId)
    : undefined;
}

export function parseSettingsScrollAnchor(
  value: string | null,
): SettingsLegacyScrollAnchor | undefined {
  if (value === "promptEnhancer" || value === "textEncoding") return value;
  return undefined;
}

export function parseSettingsInitialReason(
  value: string | null,
): SettingsInitialReason | undefined {
  if (value === "geminiKeyRequired" || value === "ltxKeyRequired") return value;
  return undefined;
}

export type SettingsOpenDetail = {
  tab?: SettingsTabId;
  /** Row to scroll to once the tab mounts, for legacy hash links like #promptEnhancer. */
  scrollAnchor?: SettingsLegacyScrollAnchor;
  reason?: SettingsInitialReason;
};

