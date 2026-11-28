import { useEffect, useRef, type ReactNode } from 'react'
import { X } from 'lucide-react'
import { Text } from '@ds/Text/Text'

interface LibraryModalProps {
  open: boolean
  onClose: () => void
  title: string
  headerSlot?: ReactNode
  children: ReactNode
}

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
      onClick={(e) => { if (e.target === ref.current) onClose() }}
      className="m-auto w-[min(960px,calc(100%-2rem))] max-h-[80vh] overflow-hidden rounded-2xl border border-separator bg-surface-primary p-0 text-fg-primary backdrop:bg-surface-overlay"
    >
      <div className="flex max-h-[80vh] flex-col">
        <div className="flex shrink-0 items-center justify-between gap-3 border-b border-separator-secondary px-4 py-3">
          <Text as="h2" variant="label" size="lg">{title}</Text>
          <div className="flex items-center gap-2">
            {headerSlot}
            <button
              onClick={onClose}
              aria-label="Close"
              className="rounded-md p-1.5 text-fg-secondary transition-colors hover:bg-action hover:text-fg-primary"
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
