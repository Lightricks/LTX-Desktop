import { useEffect, useState } from 'react'
import { ApiClient } from '../lib/api-client'
import { useAppSettings } from '../contexts/AppSettingsContext'

export type EnhanceProvider = 'local' | 'api'

interface UsePromptEnhancerProviderResult {
  // Local requires the Gemma text-encoder checkpoint to be downloaded AND local generation to
  // actually be usable this run (e.g. not memory-constrained into API-only mode); API requires a
  // stored Gemini key (presence only — an invalid key still surfaces via the enhance call's own
  // error response, so no separate validity check is worth the extra backend round-trip).
  hasLocalTextEncoder: boolean
  hasGeminiApiKey: boolean
  // Whether at least one provider is usable.
  isAvailable: boolean
  // The provider Enhance will actually use: the persisted preference when it's currently
  // available, otherwise whichever single option works right now. A provider that's temporarily
  // unavailable (e.g. local on a memory-constrained run) falls back silently — it does NOT
  // overwrite the persisted preference, which only an explicit setProviderPreference call changes.
  provider: EnhanceProvider
  // Only true (and thus only worth showing a picker for) when the user actually has a choice
  // between two currently-usable providers.
  canToggleProvider: boolean
  setProviderPreference: (provider: EnhanceProvider) => void
}

// Single source of truth for which prompt-enhancer provider (local Gemma text encoder vs.
// Gemini's hosted API) is available and which one Enhance should use. `enabled` gates the local
// checkpoint lookup so it only fires once the enhancer could plausibly be shown for the current
// mode.
export function usePromptEnhancerProvider(enabled: boolean): UsePromptEnhancerProviderResult {
  const {
    settings: { hasGeminiApiKey, promptEnhancerProviderPreference },
    updateSettings,
    forceApiGenerations,
  } = useAppSettings()

  const [isTextEncoderDownloaded, setIsTextEncoderDownloaded] = useState(false)
  useEffect(() => {
    if (!enabled) return
    let cancelled = false
    void ApiClient.getTextEncoderRecommendation().then((result) => {
      if (!cancelled) setIsTextEncoderDownloaded(result.ok && result.data.cp_to_download === null)
    })
    return () => { cancelled = true }
  }, [enabled])

  // Downloaded isn't enough on its own — forceApiGenerations is the pure "insufficient memory
  // for local models this run" signal (deliberately NOT shouldVideoGenerateWithLtxApi, which
  // also folds in the user's own preference to use the LTX API for VIDEO specifically — that's
  // unrelated to whether the much smaller Gemma text encoder can run locally right now).
  const hasLocalTextEncoder = isTextEncoderDownloaded && !forceApiGenerations
  const canToggleProvider = hasLocalTextEncoder && hasGeminiApiKey

  // Default to local (the first available option) when the user hasn't made an explicit choice,
  // or when their choice isn't currently usable — a silent, non-persisted fallback.
  const provider: EnhanceProvider =
    promptEnhancerProviderPreference === 'api' && hasGeminiApiKey ? 'api'
    : promptEnhancerProviderPreference === 'local' && hasLocalTextEncoder ? 'local'
    : hasLocalTextEncoder ? 'local'
    : 'api'

  const setProviderPreference = (next: EnhanceProvider) => {
    updateSettings({ promptEnhancerProviderPreference: next })
  }

  return {
    hasLocalTextEncoder,
    hasGeminiApiKey,
    isAvailable: hasLocalTextEncoder || hasGeminiApiKey,
    provider,
    canToggleProvider,
    setProviderPreference,
  }
}
