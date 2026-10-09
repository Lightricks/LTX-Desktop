
export type SettingsTabId =
  | "general"
  | "models"
  | "apiKeys"
  | "legacy"
  | "about";

/** In-page anchors (old hash links). */
export type SettingsLegacyScrollAnchor = "promptEnhancer" | "textEncoding";

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

/**
 * Which tabs exist for this runtime. One source of truth on purpose: the answer is needed
 * both to build the tab strip and to bounce a selection that can no longer be shown (from a
 * deep link or a stale value), and keeping those two in separate conditions is how Legacy
 * ended up listed but unreachable.
 *
 * Models is the only tab force-API mode removes — it exists purely to manage local weights.
 * Legacy stays: it owns the project assets path and the seed lock, which apply to every
 * generation, and its local-only rows gate themselves.
 */
export function isSettingsTabAvailable(
  tab: SettingsTabId,
  { forceApiGenerations }: { forceApiGenerations: boolean },
): boolean {
  if (tab === "models") return !forceApiGenerations;
  return true;
}

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

