import { useEffect, useRef, useState } from 'react'
import { Text } from '@ds/Text/Text'

export type GallerySize = 'small' | 'medium' | 'large'

export const gallerySizeClasses: Record<GallerySize, string> = {
  small: 'grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 2xl:grid-cols-7',
  medium: 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 2xl:grid-cols-5',
  large: 'grid-cols-1 sm:grid-cols-1 md:grid-cols-2 lg:grid-cols-2 xl:grid-cols-3',
}

// All three glyphs sit on the same 18×18 content box (3→21) that Lucide uses,
// so a filled grid reads at the same optical size as the stroked Heart and
// Sparkles beside it. Whole cells only — partial cells blur at 16px.
function GridCells({ className, cells, gap, radius }: { className?: string; cells: number; gap: number; radius: number }) {
  const size = (18 - gap * (cells - 1)) / cells
  const offsets = Array.from({ length: cells }, (_, i) => 3 + i * (size + gap))
  return (
    <svg className={className} viewBox="0 0 24 24" fill="currentColor">
      {offsets.map(y => offsets.map(x => (
        <rect key={`${x}-${y}`} x={x} y={y} width={size} height={size} rx={radius} />
      )))}
    </svg>
  )
}

function GridSmallIcon({ className }: { className?: string }) {
  return <GridCells className={className} cells={4} gap={2} radius={0.5} />
}

function GridMediumIcon({ className }: { className?: string }) {
  return <GridCells className={className} cells={3} gap={2} radius={1} />
}

function GridLargeIcon({ className }: { className?: string }) {
  return <GridCells className={className} cells={2} gap={2} radius={1.5} />
}

export function GenSpaceGallerySizeMenu({
  gallerySize,
  onGallerySizeChange,
}: {
  gallerySize: GallerySize
  onGallerySizeChange: (size: GallerySize) => void
}) {
  const [open, setOpen] = useState(false)
  const menuRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setOpen(false)
      }
    }
    if (open) {
      document.addEventListener('mousedown', handleClickOutside)
    }
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [open])

  const CurrentIcon = gallerySize === 'small'
    ? GridSmallIcon
    : gallerySize === 'medium'
      ? GridMediumIcon
      : GridLargeIcon

  return (
    <div ref={menuRef} className="relative">
      <button
        type="button"
        onClick={() => setOpen(value => !value)}
        aria-label={`Gallery size: ${gallerySize}`}
        aria-haspopup="menu"
        aria-expanded={open}
        className={`p-2 rounded-md transition-colors ${
          open ? 'bg-action text-fg-primary' : 'text-fg-secondary hover:text-fg-primary hover:bg-action'
        }`}
      >
        <CurrentIcon className="h-4 w-4" />
      </button>

      {open && (
        <div className="absolute top-full mt-2 right-0 bg-action border border-separator rounded-md p-2 min-w-[160px] shadow-xl z-50">
          {([
            { value: 'small' as const, label: 'Small', icon: GridSmallIcon },
            { value: 'medium' as const, label: 'Medium', icon: GridMediumIcon },
            { value: 'large' as const, label: 'Large', icon: GridLargeIcon },
          ]).map(option => (
            <button
              key={option.value}
              type="button"
              onClick={() => { onGallerySizeChange(option.value); setOpen(false) }}
              className={`w-full flex items-center justify-between px-2 py-2.5 rounded-md transition-colors text-left ${gallerySize === option.value ? 'bg-[color-mix(in_srgb,var(--semantic-fg-primary)_20%,transparent)] hover:bg-[color-mix(in_srgb,var(--semantic-fg-primary)_25%,transparent)]' : 'hover:bg-action-hover'}`}
            >
              <div className="flex items-center gap-3">
                <option.icon className={`h-4 w-4 ${gallerySize === option.value ? 'text-fg-primary' : 'text-fg-tertiary'}`} />
                <Text
                  as="span"
                  variant={gallerySize === option.value ? 'label' : 'body'}
                  size="lg"
                  className={gallerySize === option.value ? 'text-fg-primary' : 'text-fg-secondary'}
                >
                  {option.label}
                </Text>
              </div>
              {gallerySize === option.value && (
                <svg className="w-4 h-4 text-fg-primary" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
              )}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}
