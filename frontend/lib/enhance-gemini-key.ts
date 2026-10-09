export const GEMINI_KEY_REQUIRED_SETTINGS_DETAIL = {
  tab: 'apiKeys' as const,
  reason: 'geminiKeyRequired' as const,
}

export function isEnhanceBlockedByMissingGeminiKey(input: {
  enhanceAvailableForMode: boolean
  enhanceProvider: 'local' | 'api'
  hasGeminiApiKey: boolean
  hasEnhanceInput: boolean
  isGenerationInProgressForEnhance: boolean
  isOtherGenerationRunning: boolean
}): boolean {
  // True when Enhance would run via Gemini but no key is configured (API-only path).
  // When local Gemma is available, the provider hook falls back to local instead.
  return (
    input.enhanceAvailableForMode
    && input.enhanceProvider === 'api'
    && !input.hasGeminiApiKey
    && input.hasEnhanceInput
    && !input.isGenerationInProgressForEnhance
    && !input.isOtherGenerationRunning
  )
}
