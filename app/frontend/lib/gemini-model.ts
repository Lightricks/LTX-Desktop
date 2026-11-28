export const DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite";

/** "gemini-3.5-flash-lite" -> "Gemini 3.5 Flash Lite". An empty id means the default model. */
export function geminiModelLabel(modelId: string): string {
  return (modelId.trim().replace(/^models\//, "") || DEFAULT_GEMINI_MODEL)
    .split("-")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}
