import { useEffect, useRef, type ReactNode } from 'react'
import { X } from 'lucide-react'

interface LibraryModalProps {
  open: boolean
  onClose: () => void
  title: string
  headerSlot?: ReactNode
  children: ReactNode
}

// Generic library dialog shell — no asset/recipe knowledge. A future regular-LoRA
// library reuses it.
export function LibraryModal({ open, onClose, title, headerSlot, children }: LibraryModalProps) {
  const ref = useRef<HTMLDialogElement>(null)

  useEffect(() => {
    const el = ref.current
    if (!el) return
    if (open && !el.open) el.showModal()
    else if (!open && el.open) el.close()
  }, [open])

  return (
    <dialog
      ref={ref}
      onClose={onClose}
      // Clicking the backdrop targets the <dialog> element itself (content is the inner div).
      onClick={(e) => { if (e.target === ref.current) onClose() }}
      className="m-auto w-[min(960px,calc(100%-2rem))] max-h-[80vh] overflow-hidden rounded-2xl border border-zinc-800 bg-zinc-900 p-0 text-white backdrop:bg-black/60"
    >
      {/* Flex column so only the body scrolls — no hardcoded header-height calc. */}
      <div className="flex max-h-[80vh] flex-col">
        <div className="flex shrink-0 items-center justify-between gap-3 border-b border-zinc-800 px-4 py-3">
          <h2 className="text-sm font-semibold">{title}</h2>
          <div className="flex items-center gap-2">
            {headerSlot}
            <button
              onClick={onClose}
              aria-label="Close"
              className="rounded-md p-1.5 text-zinc-400 transition-colors hover:bg-zinc-800 hover:text-white"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
        </div>
        <div className="min-h-0 flex-1 overflow-y-auto p-3">{children}</div>
      </div>
    </dialog>
  )
}
