import { useEffect, useRef, type ButtonHTMLAttributes, type ReactNode } from 'react'
import './ProductUpdateDialog.css'
import { cn } from '@/lib/utils'
import { Text } from '@ds/Text/Text'

const FOCUSABLE_SELECTOR =
  'a[href],button:not([disabled]),textarea:not([disabled]),input:not([disabled]),select:not([disabled]),[tabindex]:not([tabindex="-1"])'

function getFocusableElements(container: HTMLElement): HTMLElement[] {
  return Array.from(container.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR)).filter(
    (element) => !element.hasAttribute('disabled') && element.getAttribute('aria-hidden') !== 'true',
  )
}

function isProductDialogRoot(element: HTMLElement): boolean {
  return element.classList.contains('product-dialog-backdrop')
    || element.querySelector('.product-dialog-backdrop') !== null
}

function markBackgroundInert(dialogRoot: HTMLElement): () => void {
  const marked: HTMLElement[] = []
  let node: HTMLElement | null = dialogRoot
  while (node) {
    const parent: HTMLElement | null = node.parentElement
    if (!parent) break
    for (const child of Array.from(parent.children)) {
      if (!(child instanceof HTMLElement) || child === node || child.hasAttribute('inert')) continue
      // Another product dialog stays active so two prompts do not inert each other.
      // Every other dialog on the page, including ones behind this overlay, is inert.
      if (isProductDialogRoot(child)) continue
      child.setAttribute('inert', '')
      marked.push(child)
    }
    if (parent === document.body) break
    node = parent
  }
  return () => {
    for (const element of marked) element.removeAttribute('inert')
  }
}

interface AppProductDialogShellProps {
  title: string
  subtitle?: ReactNode
  headerActions?: ReactNode
  children: ReactNode
  footer?: ReactNode
  onBackdropClick?: () => void
  className?: string
  'aria-labelledby'?: string
}

export function AppProductDialogShell({
  title,
  subtitle,
  headerActions,
  children,
  footer,
  onBackdropClick,
  className,
  'aria-labelledby': ariaLabelledBy,
}: AppProductDialogShellProps) {
  const rootRef = useRef<HTMLDivElement>(null)
  const dialogRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const root = rootRef.current
    const dialog = dialogRef.current
    if (!root || !dialog) return

    const previouslyFocused = document.activeElement instanceof HTMLElement ? document.activeElement : null
    const releaseInert = markBackgroundInert(root)

    const focusFirstInsideDialog = () => {
      const focusable = getFocusableElements(dialog)
      ;(focusable[0] ?? dialog).focus()
    }
    focusFirstInsideDialog()

    const handleFocusIn = (event: FocusEvent) => {
      const target = event.target
      if (!(target instanceof Node) || root.contains(target)) return
      if (!(target instanceof Element)) {
        focusFirstInsideDialog()
        return
      }
      if (target.closest('[data-radix-popper-content-wrapper], [data-radix-portal]')) return
      const otherDialog = target.closest('.product-dialog-backdrop')
      if (
        otherDialog
        && otherDialog !== root
        && (root.compareDocumentPosition(otherDialog) & Node.DOCUMENT_POSITION_FOLLOWING) !== 0
      ) {
        return
      }
      focusFirstInsideDialog()
    }

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key !== 'Tab') return
      const focusable = getFocusableElements(dialog)
      if (focusable.length === 0) {
        event.preventDefault()
        dialog.focus()
        return
      }
      const first = focusable[0]
      const last = focusable[focusable.length - 1]
      const active = document.activeElement
      if (!event.shiftKey && active === last) {
        event.preventDefault()
        first.focus()
      } else if (event.shiftKey && active === first) {
        event.preventDefault()
        last.focus()
      }
    }

    document.addEventListener('focusin', handleFocusIn)
    document.addEventListener('keydown', handleKeyDown)
    return () => {
      document.removeEventListener('focusin', handleFocusIn)
      document.removeEventListener('keydown', handleKeyDown)
      releaseInert()
      previouslyFocused?.focus()
    }
  }, [])

  return (
    <div ref={rootRef} className="product-dialog-backdrop fixed inset-0 z-[70] flex items-center justify-center bg-surface-overlay/70 px-4 py-8">
      <div className="absolute inset-0" onClick={onBackdropClick} aria-hidden />

      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby={ariaLabelledBy}
        tabIndex={-1}
        className={cn(
          'product-dialog-card relative flex max-h-[calc(100dvh-4rem)] min-h-0 w-full max-w-[440px] flex-col overflow-hidden rounded-xl border border-separator bg-surface-primary text-fg-primary shadow-xl outline-none',
          className,
        )}
      >
        <div className="relative shrink-0 px-5 pb-1 pt-5 text-center">
          {headerActions ? (
            <div className="absolute right-3 top-3 flex items-center">{headerActions}</div>
          ) : null}
          <Text
            as="h2"
            id={ariaLabelledBy}
            variant="heading"
            size="lg"
            align="center"
            className="text-fg-primary"
          >
            {title}
          </Text>
          {subtitle ? (
            <div className="mx-auto mt-2 max-w-[34ch] text-base leading-snug text-fg-secondary">{subtitle}</div>
          ) : null}
        </div>

        <div className="min-h-0 flex-1 space-y-4 overflow-y-auto px-5 py-4">{children}</div>

        {footer ? (
          <div className="flex w-full shrink-0 flex-col items-center gap-2 px-5 pb-5 pt-2">
            {footer}
          </div>
        ) : null}
      </div>
    </div>
  )
}

/** Plain content block — no nested card chrome. */
export function ProductDialogBlock({
  title,
  children,
  className,
  tone = 'default',
}: {
  title?: string
  children: ReactNode
  className?: string
  tone?: 'default' | 'highlight'
}) {
  const body = (
    <div className={cn(tone === 'highlight' && 'product-dialog-highlight-body pr-0.5')}>
      {children}
    </div>
  )

  return (
    <div
      className={cn(
        'space-y-2',
        tone === 'highlight' && 'flex min-h-0 flex-col rounded-lg bg-action px-4 py-3',
        className,
      )}
    >
      {title ? (
        <Text as="h3" variant="label" size="sm" className="shrink-0 text-fg-primary">
          {title}
        </Text>
      ) : null}
      {tone === 'highlight' ? body : children}
    </div>
  )
}

export function ProductDialogIconButton({
  className,
  children,
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      type="button"
      className={cn(
        '-mr-1 rounded-md p-1.5 text-fg-tertiary transition-colors hover:bg-action hover:text-fg-primary',
        className,
      )}
      {...props}
    >
      {children}
    </button>
  )
}

export function ProductDialogTextButton({
  children,
  className,
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      type="button"
      className={cn(
        'text-xs text-fg-tertiary transition-colors hover:text-fg-primary',
        className,
      )}
      {...props}
    >
      {children}
    </button>
  )
}

export function ProductDialogCheckboxRow({
  checked,
  onChange,
  disabled,
  children,
  className,
}: {
  checked: boolean
  onChange: (checked: boolean) => void
  disabled?: boolean
  children: ReactNode
  className?: string
}) {
  return (
    <label className={cn('flex cursor-pointer items-start gap-2.5 py-1 text-sm text-fg-secondary', className)}>
      <input
        type="checkbox"
        checked={checked}
        onChange={(event) => onChange(event.target.checked)}
        disabled={disabled}
        className="mt-0.5 h-4 w-4 shrink-0 rounded border-separator text-primary focus:ring-ring"
      />
      <span>{children}</span>
    </label>
  )
}
