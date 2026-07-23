import { LibraryModal } from './library/LibraryModal'
import { LibraryItemCard } from './library/LibraryItemCard'
import { LoraInfoPopover } from './LoraInfoPopover'
import { useHfAuth } from '../hooks/use-hf-auth'
import type { LibraryEntry } from '../lib/lora-library'

export type LibraryKind = 'lora' | 'ic-lora'

interface LoraLibraryModalProps {
  open: boolean
  onClose: () => void
  kind: LibraryKind
  items: LibraryEntry[]
  selectedId: string | null
  // Which variant of selectedId is actually active — lets the card badge the exact
  // installed checkpoint as "Selected" rather than the whole catalog entry.
  selectedVariantId?: string | null
  downloadingKey: string | null
  progress: number
  downloadError: { key: string; message: string } | null
  // Transient message (e.g. "still syncing installed files") shown when onSelect returns false.
  syncError?: string | null
  onDownload: (id: string, variantId?: string) => void
  // Returns whether the entry was actually selected — the modal only closes on success.
  onSelect: (entry: LibraryEntry, variantId?: string) => boolean
}

// Generic catalog library modal — works for both plain LoRAs and IC-LoRAs. The consumer
// maps its API/on-disk types into LibraryEntry[]; LibraryModal/LibraryItemCard stay
// domain-agnostic.
export function LoraLibraryModal({
  open, onClose, kind, items, selectedId, selectedVariantId, downloadingKey, progress, downloadError,
  syncError, onDownload, onSelect,
}: LoraLibraryModalProps) {
  // A catalog entry with no download variants has nothing to show a card for.
  const visibleItems = items.filter(e => (e.variants?.length ?? 0) > 0)
  // HF "Connect" CTA: shown only when a gated item is visible and we're not signed in.
  const anyGated = visibleItems.some(e => e.requiresHfLogin)
  const { hfAuthStatus } = useHfAuth(open && anyGated)
  const showConnectCta = anyGated && hfAuthStatus !== 'authenticated'
  const handleConnect = () => {
    window.dispatchEvent(new CustomEvent('open-settings', { detail: { tab: 'apiKeys' } }))
    onClose()
  }

  return (
    <LibraryModal
      open={open}
      onClose={onClose}
      title={kind === 'ic-lora' ? 'IC-LoRAs' : 'LoRAs'}
      headerSlot={showConnectCta ? (
        <button
          onClick={handleConnect}
          className="inline-flex items-center gap-1.5 rounded-lg bg-indigo-600 px-3 py-1.5 text-xs font-medium text-white transition-colors hover:bg-indigo-500"
        >
          Connect HuggingFace
        </button>
      ) : undefined}
    >
      {syncError && (
        <div className="mb-3 rounded-lg border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-xs text-amber-300">
          {syncError}
        </div>
      )}
      {visibleItems.length === 0 ? (
        <div className="px-2 py-8 text-center text-xs text-zinc-500">
          {kind === 'ic-lora' ? 'No IC-LoRAs available.' : 'No LoRAs found. Downloaded and installed LoRAs will appear here.'}
        </div>
      ) : (
        <div className="grid grid-cols-2 gap-3 lg:grid-cols-3">
          {visibleItems.map(e => {
            const cardError = !downloadError ? null
              : (e.requiresHfLogin && hfAuthStatus !== 'authenticated')
                ? { key: downloadError.key, message: 'Sign in to HuggingFace (top right), then retry.', gated: false }
                : { ...downloadError, gated: Boolean(e.requiresHfLogin) }
            return (
              <LibraryItemCard
                key={e.id}
                selected={selectedId === e.id}
                selectedVariantId={selectedId === e.id ? selectedVariantId : undefined}
                downloadingKey={downloadingKey}
                progress={progress}
                downloadError={cardError}
                item={{
                  id: e.id,
                  title: e.name,
                  description: e.description || undefined,
                  sizeBytes: e.sizeBytes,
                  downloaded: e.downloaded,
                  thumbnailUrl: e.thumbnailUrl,
                  demoVideoUrl: e.demoVideoUrl,
                  author: e.author,
                  license: e.license,
                  requiresHfLogin: e.requiresHfLogin,
                  variants: e.variants?.map(v => ({
                    id: v.id,
                    label: v.label,
                    sizeBytes: v.sizeBytes,
                    downloaded: e.downloadedVariantIds?.includes(v.id),
                  })),
                  defaultVariantId: e.defaultVariantId,
                }}
                infoSlot={(
                  <LoraInfoPopover
                    sections={e.instructions ?? []}
                    name={e.name}
                    repoId={e.repoId}
                    isCommunity={e.author?.affiliation !== 'ltx'}
                    maxHeight="45vh"
                  />
                )}
                onDownload={onDownload}
                onRetry={onDownload}
                onRequestAccess={() => { if (e.repoId) void window.electronAPI.openHuggingFaceRepo({ repoId: e.repoId }) }}
                onUse={(_id, variantId) => { if (onSelect(e, variantId)) onClose() }}
              />
            )
          })}
        </div>
      )}
    </LibraryModal>
  )
}
