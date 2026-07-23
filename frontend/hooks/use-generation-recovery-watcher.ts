import { useEffect, useRef } from 'react'
import { useProjects } from '../contexts/ProjectContext'
import { useAppSettings } from '../contexts/AppSettingsContext'
import { checkAndConsumeRecovery } from '../lib/generation-recovery'
import { subscribeWhileGenerationMayBeActive } from '../lib/generation-progress-poll'

// Always-mounted safety net for a generation that finishes while its own project's GenSpace
// isn't open (a different project, the Video Editor tab, or Home): periodically checks whether
// the pending recovery marker's generation has completed and, if so, persists it directly into
// that project's assets — independent of whichever view/project is currently on screen. Mount
// this once, near the app root (outside whatever view-switching renders GenSpace), so it survives
// navigation. GenSpace's own live polling/import effects are unchanged and take priority — see
// setActiveGenerationOwner in lib/generation-recovery.ts for how the two avoid double-importing.
export function useGenerationRecoveryWatcher(): void {
  const { addAsset } = useProjects()
  const { settings } = useAppSettings()
  const modelsDirRef = useRef(settings.modelsDir)
  modelsDirRef.current = settings.modelsDir
  const isCheckingRef = useRef(false)

  useEffect(() => subscribeWhileGenerationMayBeActive(progress => {
    if (isCheckingRef.current) return
    isCheckingRef.current = true
    void checkAndConsumeRecovery(progress, { addAsset, modelsDir: modelsDirRef.current })
      .finally(() => { isCheckingRef.current = false })
  }), [addAsset])
}
