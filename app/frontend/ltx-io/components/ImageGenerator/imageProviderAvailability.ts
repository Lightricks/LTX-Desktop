import type { components } from "@/generated/backend-openapi";

type ImageGenerationStatus = components["schemas"]["ImageGenerationStatusResponse"];
export type ImageProvider = ImageGenerationStatus["defaultProvider"];
type ImageProviderBlocker = NonNullable<
  ImageGenerationStatus["localBlocker"] | ImageGenerationStatus["falBlocker"]
>;

export type ImageHost = "desktop" | "remote";

export type ImageProviderOption = {
  provider: ImageProvider;
  label: string;
  /** Null when this provider can run now. Otherwise the reason and the fix. */
  blockedReason: string | null;
};

const PROVIDER_LABELS: Record<ImageProvider, string> = {
  local: "Local",
  fal: "fal API",
};

// Remote cannot open Settings or start a download. Its text names Desktop as the place to fix it.
const BLOCKER_MESSAGES: Record<ImageProviderBlocker, (where: string) => string> = {
  FAL_API_KEY_NOT_CONFIGURED: (where) => `Add a fal API key ${where}.`,
  LOCAL_GENERATION_UNSUPPORTED: () => "This machine cannot run local generation.",
  IMAGE_MODEL_NOT_DOWNLOADED: (where) => `Download the Z-Image Turbo model ${where}.`,
};

function blockerMessage(blocker: ImageProviderBlocker, host: ImageHost): string {
  const where = host === "remote" ? "on Desktop" : "in Settings";
  return BLOCKER_MESSAGES[blocker](where);
}

function isImageProviderBlocker(code: string): code is ImageProviderBlocker {
  return Object.prototype.hasOwnProperty.call(BLOCKER_MESSAGES, code);
}

/** A refused create says why with its code. Show the same host-aware text as the picker. */
export function imageCreateErrorMessage(
  error: { code?: string; message: string },
  host: ImageHost,
): string {
  const { code } = error;
  return code !== undefined && isImageProviderBlocker(code)
    ? blockerMessage(code, host)
    : error.message;
}

export function describeImageProviders(
  status: ImageGenerationStatus,
  host: ImageHost,
): ImageProviderOption[] {
  const option = (provider: ImageProvider, blocker: ImageProviderBlocker | null) => ({
    provider,
    label: PROVIDER_LABELS[provider],
    blockedReason: blocker === null ? null : blockerMessage(blocker, host),
  });
  return [option("local", status.localBlocker), option("fal", status.falBlocker)];
}

function isUsableProvider(
  options: readonly ImageProviderOption[],
  provider: ImageProvider | null,
): provider is ImageProvider {
  return options.some(
    (option) => option.provider === provider && option.blockedReason === null,
  );
}

/**
 * The provider a request uses. A pick that cannot run now gives way to the backend default, so
 * a stale pick never blocks Generate when the other provider works.
 */
export function resolveImageProvider(
  options: readonly ImageProviderOption[],
  defaultProvider: ImageProvider,
  picked: ImageProvider | null,
): ImageProvider {
  if (isUsableProvider(options, picked)) return picked;
  if (isUsableProvider(options, defaultProvider)) return defaultProvider;
  return options.find((option) => option.blockedReason === null)?.provider ?? defaultProvider;
}
