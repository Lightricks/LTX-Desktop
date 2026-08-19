import { Heart, Image, Video } from 'lucide-react'
import type { GenSpaceTypeFilter } from '../../lib/genspace-gallery'

export function GenSpaceFilterEmptyState({
  typeFilter,
  showFavorites,
  hasTypeMatches,
}: {
  typeFilter: GenSpaceTypeFilter
  showFavorites: boolean
  hasTypeMatches: boolean
}) {
  if (typeFilter !== 'all' && !hasTypeMatches) {
    if (typeFilter === 'video') {
      return (
        <div className="absolute inset-0 flex flex-col items-center justify-center text-center pointer-events-none">
          <Video className="h-12 w-12 text-zinc-700 mb-4" />
          <h3 className="text-lg font-semibold text-white mb-2">No videos yet</h3>
          <p className="text-zinc-500 text-sm">
            Generate a video or switch the filter to see other media.
          </p>
        </div>
      )
    }

    return (
      <div className="absolute inset-0 flex flex-col items-center justify-center text-center pointer-events-none">
        <Image className="h-12 w-12 text-zinc-700 mb-4" />
        <h3 className="text-lg font-semibold text-white mb-2">No images yet</h3>
        <p className="text-zinc-500 text-sm">
          Generate an image or switch the filter to see other media.
        </p>
      </div>
    )
  }

  if (showFavorites) {
    return (
      <div className="absolute inset-0 flex flex-col items-center justify-center text-center pointer-events-none">
        <Heart className="h-12 w-12 text-zinc-700 mb-4" />
        <h3 className="text-lg font-semibold text-white mb-2">No favorites yet</h3>
        <p className="text-zinc-500 text-sm">
          Click the heart icon on any asset to add it to your favorites.
        </p>
      </div>
    )
  }

  return null
}
