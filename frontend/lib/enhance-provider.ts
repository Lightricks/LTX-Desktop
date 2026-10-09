export type EnhanceProvider = "local" | "api";

/** Use a stored preference only when that provider can actually run. Do not persist the fallback. */
export function resolveUsableEnhanceProvider(input: {
  preference: EnhanceProvider | null | undefined;
  hasGeminiApiKey: boolean;
  hasLocalTextEncoder: boolean;
  fallback: EnhanceProvider;
}): EnhanceProvider {
  const preferred = input.preference ?? input.fallback;
  if (preferred === "api" && !input.hasGeminiApiKey && input.hasLocalTextEncoder) {
    return "local";
  }
  if (preferred === "local" && !input.hasLocalTextEncoder) {
    return input.hasGeminiApiKey ? "api" : input.fallback;
  }
  return preferred;
}
