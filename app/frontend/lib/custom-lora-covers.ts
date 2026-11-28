import { pathToFileUrl } from "./file-url.ts";

/**
 * The address of a LoRA cover.
 * On this computer `coverUrl` is a file path. On a paired phone it is a signed `/api/...` URL.
 */
export function customLoraPosterUrl(coverUrl: string | null | undefined): string | null {
  if (coverUrl == null || coverUrl.length === 0) return null;
  if (
    coverUrl.startsWith("/api/") ||
    coverUrl.startsWith("http://") ||
    coverUrl.startsWith("https://")
  ) {
    return coverUrl;
  }
  return pathToFileUrl(coverUrl);
}
