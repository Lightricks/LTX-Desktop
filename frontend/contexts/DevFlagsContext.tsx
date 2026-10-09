import React, { createContext, useContext, useState, useCallback } from 'react'
import { useDevPanelShortcut } from '../hooks/use-dev-panel-shortcut'
import { ApiClient } from '../lib/api-client'
import { UNLOADED_FLAGS, type FeatureFlags } from '../lib/dev-flags'
// eslint-disable-next-line no-restricted-imports
import { DevPanel } from '@/ltx-io/components/DevPanel/DevPanel'
// eslint-disable-next-line no-restricted-imports
import { useFeatureFlags } from '@/ltx-io/hooks/useFeatureFlags'

// Desktop host for the feature flags, which live in the backend's feature_flags.json
// (shared with the phone remote). Consumers only read them; toggling happens in the
// Dev Panel (Ctrl/Cmd+Shift+D), which this provider owns.
// Must render inside the QueryClientProvider.

const DevFlagsContext = createContext<{ flags: FeatureFlags } | null>(null)

export function DevFlagsProvider({ children }: { children: React.ReactNode }) {
  const { data } = useFeatureFlags(ApiClient)
  const [isPanelOpen, setPanelOpen] = useState(false)

  useDevPanelShortcut(useCallback(() => setPanelOpen(prev => !prev), []))

  return (
    <DevFlagsContext.Provider value={{ flags: data ?? UNLOADED_FLAGS }}>
      {children}
      <DevPanel api={ApiClient} open={isPanelOpen} onClose={() => setPanelOpen(false)} />
    </DevFlagsContext.Provider>
  )
}

export function useDevFlags(): { flags: FeatureFlags } {
  const ctx = useContext(DevFlagsContext)
  if (!ctx) throw new Error('useDevFlags must be used within DevFlagsProvider')
  return ctx
}
