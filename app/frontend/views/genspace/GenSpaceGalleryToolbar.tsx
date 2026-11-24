import { Heart, Sparkles } from 'lucide-react'
import { Text } from '@ds/Text/Text'
import type { GenSpaceSortDir, GenSpaceSortKey, GenSpaceTypeFilter } from '../../lib/genspace-gallery'
import { GenSpaceGallerySizeMenu, type GallerySize } from './GenSpaceGallerySizeMenu'
import { GenSpaceSortMenu } from './GenSpaceSortMenu'
import { GenSpaceTypeFilter as TypeFilter } from './GenSpaceTypeFilter'

export function GenSpaceGalleryToolbar({
  showBrowseLoras,
  onBrowseLoras,
  typeFilter,
  onTypeFilterChange,
  sortKey,
  sortDir,
  onSortKeyChange,
  onToggleSortDir,
  showFavorites,
  favoriteCount,
  onToggleFavorites,
  gallerySize,
  onGallerySizeChange,
}: {
  showBrowseLoras: boolean
  onBrowseLoras: () => void
  typeFilter: GenSpaceTypeFilter
  onTypeFilterChange: (value: GenSpaceTypeFilter) => void
  sortKey: GenSpaceSortKey
  sortDir: GenSpaceSortDir
  onSortKeyChange: (key: GenSpaceSortKey) => void
  onToggleSortDir: () => void
  showFavorites: boolean
  favoriteCount: number
  onToggleFavorites: () => void
  gallerySize: GallerySize
  onGallerySizeChange: (size: GallerySize) => void
}) {
  return (
    <div className="flex items-center justify-between pb-2 gap-2">
      <div className="flex items-center gap-2">
        {showBrowseLoras && (
          <button
            type="button"
            onClick={onBrowseLoras}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm font-medium text-fg-secondary hover:text-fg-primary hover:bg-action transition-colors"
          >
            <Sparkles className="h-4 w-4" />
            <Text as="span" variant="label" size="lg">Browse LoRAs</Text>
          </button>
        )}
      </div>
      <div className="flex items-center gap-2">
        <TypeFilter value={typeFilter} onChange={onTypeFilterChange} />
        <GenSpaceSortMenu
          sortKey={sortKey}
          sortDir={sortDir}
          onSortKeyChange={onSortKeyChange}
          onToggleSortDir={onToggleSortDir}
        />
        <button
          type="button"
          onClick={onToggleFavorites}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
            showFavorites
              ? 'bg-danger-soft text-fg-danger border border-danger/30'
              : 'text-fg-secondary hover:text-fg-primary hover:bg-action'
          }`}
        >
          <Heart className={`h-4 w-4 ${showFavorites ? 'fill-current' : ''}`} />
          <Text as="span" variant="label" size="lg">Favorites</Text>
          {favoriteCount > 0 && (
            <span className={`text-xs px-1.5 py-0.5 rounded-full ${
              showFavorites ? 'bg-danger-soft text-fg-danger' : 'bg-action text-fg-tertiary'
            }`}>
              {favoriteCount}
            </span>
          )}
        </button>
        <GenSpaceGallerySizeMenu
          gallerySize={gallerySize}
          onGallerySizeChange={onGallerySizeChange}
        />
      </div>
    </div>
  )
}
