export const DEEP_LINK_SCHEME = "ltxdesktop";
export const DEEP_LINK_FETCHER_HOST = "fetcher";
export const DEEP_LINK_INTENT_TTL_MS = 90 * 60 * 1000;

export const FETCHER_TOOL_SLUGS = [
  "t2v",
  "i2v",
  "a2v",
  "retake",
  "extend",
  "day-to-night",
  "alpha-gen",
  "deblur",
  "colorization",
  "clean-plate",
  "decompression",
  "water-simulation",
  "layout-to-render",
  "restore",
  "cozy-felt",
  "claymation",
  "fantasy-painterly",
  "paper-cut-out-style",
  "cinemagraph",
  "jib-up",
  "jib-down",
  "dolly-in",
  "dolly-out",
  "fpv-motion",
  "openwheel-t-cam",
  "transition",
  "vbvr",
] as const;
export type FetcherToolId = (typeof FETCHER_TOOL_SLUGS)[number];

/** Home-shell destinations. These are not generation tools, so they are
 * `ltxdesktop://{slug}` rather than `ltxdesktop://fetcher/{slug}`. */
export const DEEP_LINK_SHELL_SLUGS = ["assets"] as const;
export type DeepLinkShellSlug = (typeof DEEP_LINK_SHELL_SLUGS)[number];

export type DeepLinkTarget =
  | { kind: "fetcher-tool"; slug: FetcherToolId }
  | { kind: "home-shell"; slug: DeepLinkShellSlug };
export type DeepLinkArrival = "cold-start" | "open-url" | "second-instance";
export type DeepLinkIntent = {
  id: string;
  target: DeepLinkTarget;
  receivedAt: number;
  arrival: DeepLinkArrival;
};

export function isFetcherToolId(value: string | undefined): value is FetcherToolId {
  return value !== undefined && (FETCHER_TOOL_SLUGS as readonly string[]).includes(value);
}

export function isDeepLinkShellSlug(
  value: string | undefined,
): value is DeepLinkShellSlug {
  return (
    value !== undefined && (DEEP_LINK_SHELL_SLUGS as readonly string[]).includes(value)
  );
}

export function isDeepLinkIntentFresh(intent: DeepLinkIntent, now: number): boolean {
  return now - intent.receivedAt < DEEP_LINK_INTENT_TTL_MS;
}

function unwrapQuotedArg(raw: string): string {
  const trimmed = raw.trim();
  if (trimmed.length < 2) return trimmed;
  const quote = trimmed[0];
  if ((quote === '"' || quote === "'") && trimmed.endsWith(quote)) {
    return trimmed.slice(1, -1);
  }
  return trimmed;
}

function segmentsFromUrl(url: URL): string[] {
  return [url.hostname, ...url.pathname.split("/")]
    .map((part) => part.trim().toLowerCase())
    .filter((part) => part.length > 0);
}

export function parseDeepLink(raw: string): DeepLinkTarget | null {
  let url: URL;
  try {
    url = new URL(unwrapQuotedArg(raw));
  } catch {
    return null;
  }
  if (url.protocol !== `${DEEP_LINK_SCHEME}:`) return null;
  const parts = segmentsFromUrl(url);
  if (parts.length === 1 && isDeepLinkShellSlug(parts[0])) {
    return { kind: "home-shell", slug: parts[0] };
  }
  if (
    parts.length === 2 &&
    parts[0] === DEEP_LINK_FETCHER_HOST &&
    isFetcherToolId(parts[1])
  ) {
    return { kind: "fetcher-tool", slug: parts[1] };
  }
  return null;
}

export function findDeepLinkInArgv(argv: readonly string[]): DeepLinkTarget | null {
  for (const entry of argv) {
    const parsed = parseDeepLink(entry);
    if (parsed) return parsed;
  }
  return null;
}
