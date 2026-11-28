import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { ApiSuccessOf } from '../lib/api-client'
import type { LoraSelection } from '../components/SettingsPanel'
import { useLoraCatalog } from './use-catalog'
import {
  mergeLoraLibrary,
  resolveInstalledPath,
  variantDisplayName,
  type LibraryEntry,
} from '../lib/lora-library'

// Owns the plain-LoRA library wiring: the catalog hook, the catalog ∪ on-disk merge, the
// modal open state, "use this LoRA" → selectedLoras, and re-listing installed files when a
// download finishes. Kept out of GenSpace so the view just consumes the result.
export function useLoraLibrary(
  enabled: boolean,
  installed: ApiSuccessOf<'listModels'>['models'],
  selected: LoraSelection[],
  onSelectedChange: (loras: LoraSelection[]) => void,
  refreshInstalled: () => void,
) {
  const { loras, downloadLora, downloadingKey, progress, downloadError } = useLoraCatalog(enabled)
  const [modalOpen, setModalOpen] = useState(false)
  const [useError, setUseError] = useState<string | null>(null)
  const items = useMemo(() => mergeLoraLibrary(loras, installed), [loras, installed])

  // Re-list on-disk files once a download finishes so the new file appears + is usable.
  const prevDownloadingKey = useRef<string | null>(null)
  useEffect(() => {
    if (prevDownloadingKey.current && !downloadingKey) refreshInstalled()
    prevDownloadingKey.current = downloadingKey
  }, [downloadingKey, refreshInstalled])

  // Returns whether the LoRA was actually added, so the modal knows whether it's safe to
  // close. resolveInstalledPath never substitutes a different variant's path — if the catalog
  // says this variant is downloaded but its path hasn't landed yet (listModels refresh race),
  // re-trigger the refresh and surface a retry message instead of silently using the wrong file.
  const useEntry = useCallback((e: LibraryEntry, variantId?: string) => {
    const path = resolveInstalledPath(e, variantId)
    if (!path) {
      if (variantId && e.downloadedVariantIds?.includes(variantId)) {
        refreshInstalled()
        setUseError('Still syncing installed files — try again in a moment.')
      }
      return false
    }
    setUseError(null)
    if (selected.some(s => s.ref === path)) return true
    const variantLabel = variantId ? e.variants?.find(v => v.id === variantId)?.label : undefined
    onSelectedChange([
      ...selected,
      {
        ref: path,
        name: variantDisplayName(e.name, variantLabel, e.variants?.length),
        scale: e.recommendedStrength ?? 1.0,
        catalogId: e.id,
      },
    ])
    return true
  }, [selected, onSelectedChange, refreshInstalled])

  return {
    items, downloadLora, downloadingKey, progress, downloadError,
    modalOpen, setModalOpen, useEntry, useError,
  }
}
