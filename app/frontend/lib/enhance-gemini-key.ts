export const GEMINI_KEY_REQUIRED_SETTINGS_DETAIL = {
  tab: 'apiKeys' as const,
  reason: 'geminiKeyRequired' as const,
}

export const MISSING_GEMINI_KEY_MESSAGE =
  'Add a Gemini API key in Desktop Settings, or use local Enhance when Gemma is downloaded.'

export function isEnhanceBlockedByMissingGeminiKey(input: {
  enhanceAvailableForMode: boolean
  enhanceProvider: 'local' | 'api'
  hasGeminiApiKey: boolean
  hasEnhanceInput: boolean
}): boolean {
  // True when Enhance would run via Gemini but no key is configured (API-only path).
  // When local Gemma is available, the provider hook falls back to local instead.
  // A running generation does not matter: Gemini is an HTTP call and runs beside it.
  return (
    input.enhanceAvailableForMode
    && input.enhanceProvider === 'api'
    && !input.hasGeminiApiKey
    && input.hasEnhanceInput
  )
}

/**
 * Whether a running generation stops Enhance. Local Gemma needs the GPU slot, so it waits.
 * Gemini is an HTTP call and runs beside a generation.
 */
export function isEnhanceBlockedByGeneration(input: {
  enhanceProvider: 'local' | 'api'
  isGenerationInProgressForEnhance: boolean
  isOtherGenerationRunning: boolean
}): boolean {
  return (
    input.enhanceProvider === 'local'
    && (input.isGenerationInProgressForEnhance || input.isOtherGenerationRunning)
  )
}
