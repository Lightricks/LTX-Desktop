import { useState, useEffect, useRef } from 'react'
import { X } from 'lucide-react'

const SHOW_DELAY_MS = 2500

// Module-level flag — resets on each app launch (page reload)
let dismissedThisSession = false

export function FreeApiKeyBubble({
  forceApiGenerations,
  hasLtxApiKey,
  isGenerating,
}: {
  forceApiGenerations: boolean
  hasLtxApiKey: boolean
  isGenerating: boolean
}) {
  const [dismissed, setDismissed] = useState(() => dismissedThisSession)
  const [visible, setVisible] = useState(false)
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => {
    if (isGenerating && !dismissed && !forceApiGenerations && !hasLtxApiKey) {
      timerRef.current = setTimeout(() => setVisible(true), SHOW_DELAY_MS)
    } else {
      if (timerRef.current) clearTimeout(timerRef.current)
      timerRef.current = null
      setVisible(false)
    }
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current)
    }
  }, [isGenerating, dismissed, forceApiGenerations, hasLtxApiKey])

  if (!visible) return null

  const handleDismiss = () => {
    dismissedThisSession = true
    setDismissed(true)
  }

  const handleGoToSettings = () => {
    window.dispatchEvent(
      new CustomEvent('open-settings', { detail: { tab: 'apiKeys' } }),
    )
  }

  return (
    <div
      className="mb-2 rounded-xl bg-zinc-900/90 backdrop-blur-sm border border-zinc-700/50 px-4 py-3 text-sm text-zinc-200"
      style={{ animation: 'fadeInUp 0.3s ease-out' }}
    >
      <div className="flex items-start gap-3">
        <div className="flex-1 min-w-0">
          <p>
            Speed up inference and save memory with free cloud text encoding and
            prompt enhancement.{' '}
            <button
              onClick={handleGoToSettings}
              className="text-blue-400 hover:text-blue-300 underline underline-offset-2 transition-colors"
            >
              Get a free LTX API key
            </button>{' '}
            to enable it.
          </p>
          <p className="mt-1 text-xs text-zinc-400">
            The free API key covers text encoding and prompt enhancement only.
            Video generation via API requires a paid plan.
          </p>
        </div>
        <button
          onClick={handleDismiss}
          className="shrink-0 p-0.5 rounded hover:bg-zinc-700/60 text-zinc-400 hover:text-zinc-200 transition-colors"
          aria-label="Dismiss"
        >
          <X className="h-4 w-4" />
        </button>
      </div>
    </div>
  )
}
