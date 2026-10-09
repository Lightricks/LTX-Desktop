import { Heart, Image, Video } from 'lucide-react'
import type { ReactNode } from 'react'
import { Text } from '@ds/Text/Text'
import type { GenSpaceTypeFilter } from '../../lib/genspace-gallery'

function FilterEmpty({
  icon,
  title,
  body,
}: {
  icon: ReactNode
  title: string
  body: string
}) {
  return (
    <div className="absolute inset-0 flex flex-col items-center justify-center text-center pointer-events-none">
      {icon}
      <Text as="h3" variant="heading" size="md" align="center" className="mb-2">
        {title}
      </Text>
      <Text as="p" variant="body" size="lg" align="center" className="text-fg-secondary">
        {body}
      </Text>
    </div>
  )
}

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
        <FilterEmpty
          icon={<Video className="h-12 w-12 text-fg-tertiary mb-4" />}
          title="No videos yet"
          body="Generate a video or switch the filter to see other media."
        />
      )
    }

    return (
      <FilterEmpty
        icon={<Image className="h-12 w-12 text-fg-tertiary mb-4" />}
        title="No images yet"
        body="Generate an image or switch the filter to see other media."
      />
    )
  }

  if (showFavorites) {
    return (
      <FilterEmpty
        icon={<Heart className="h-12 w-12 text-fg-tertiary mb-4" />}
        title="No favorites yet"
        body="Click the heart icon on any asset to add it to your favorites."
      />
    )
  }

  return null
}
