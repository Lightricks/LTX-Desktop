import { useCallback, useRef, useState } from 'react'
import { Text } from '@/ds/Text/Text'
import {
  getHomeFeature,
  HOME_FEATURE_SECTIONS,
  type HomeFeatureDefinition,
} from '../../lib/home-features.ts'
import { HOME_LORA_GALLERY_SECTIONS } from '../../lib/home-lora-gallery.ts'
import { HOME_FEATURE_MEDIA, type HomeFeatureMedia } from './home-media'
import {
  browserFeaturePreviewCanPlayType,
  canAttachFeaturePreviewSrc,
  silenceFeaturePreview,
} from './homeFeaturePreviewMedia'
import { usePrefersReducedMotion } from './usePrefersReducedMotion'
import styles from './HomeGallery.module.scss'

type HomeGalleryProps = {
  onOpen: (path: string) => void
}

function gallerySectionHeadingId(heading: string): string {
  const slug = heading
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-|-$/g, '')
  return `home-gallery-${slug}`
}

export function HomeGallery({ onOpen }: HomeGalleryProps) {
  return (
    <div className={styles.gallery}>
      {HOME_FEATURE_SECTIONS.map((section) => (
        <GallerySection
          key={section.id}
          heading={section.heading}
          features={section.features}
          onOpen={onOpen}
        />
      ))}
      {HOME_LORA_GALLERY_SECTIONS.map((section) => (
        <GallerySection
          key={section.heading}
          heading={section.heading}
          features={section.featureIds.map((featureId) => getHomeFeature(featureId))}
          onOpen={onOpen}
        />
      ))}
    </div>
  )
}

function GallerySection({
  heading,
  features,
  onOpen,
}: {
  heading: string
  features: readonly HomeFeatureDefinition[]
  onOpen: (path: string) => void
}) {
  const headingId = gallerySectionHeadingId(heading)
  return (
    <section className={styles.gallerySection} aria-labelledby={headingId}>
      <Text
        as="h2"
        id={headingId}
        variant="heading"
        size="md"
        className={styles.sectionHeading}
      >
        {heading}
      </Text>
      <ul className={`${styles.cards} ${styles.wideCards}`}>
        {features.map((feature) => (
          <li key={feature.id}>
            <HomeSpotlightCard
              feature={feature}
              media={HOME_FEATURE_MEDIA[feature.id]}
              onOpen={() => onOpen(feature.path)}
            />
          </li>
        ))}
      </ul>
    </section>
  )
}

function HomeSpotlightCard({
  feature,
  media,
  onOpen,
}: {
  feature: HomeFeatureDefinition
  media: HomeFeatureMedia
  onOpen: () => void
}) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const [previewFailed, setPreviewFailed] = useState(false)
  const [isActive, setIsActive] = useState(false)
  const reducedMotion = usePrefersReducedMotion()
  const previewUrl = media.previewUrl
  const canPreview =
    previewUrl != null &&
    !previewFailed &&
    !reducedMotion &&
    canAttachFeaturePreviewSrc(previewUrl, browserFeaturePreviewCanPlayType)
  const showVideo = canPreview && isActive

  const keepSilent = useCallback(() => {
    const video = videoRef.current
    if (!video) return
    if (video.muted && video.volume === 0) return
    silenceFeaturePreview(video)
  }, [])

  return (
    <button
      type="button"
      className={styles.card}
      onClick={onOpen}
      onMouseEnter={() => setIsActive(true)}
      onMouseLeave={() => setIsActive(false)}
      onFocus={() => setIsActive(true)}
      onBlur={() => setIsActive(false)}
    >
      <span className={styles.media}>
        <img src={media.posterUrl} alt="" loading="lazy" decoding="async" />
        {showVideo && previewUrl ? (
          <video
            ref={videoRef}
            src={previewUrl}
            autoPlay
            loop
            muted
            playsInline
            aria-hidden
            onError={() => setPreviewFailed(true)}
            onLoadedData={keepSilent}
            onPlay={keepSilent}
            onVolumeChange={keepSilent}
          />
        ) : null}
      </span>
      <span className={styles.copy}>
        <Text as="span" variant="heading" size="md">
          {feature.title}
        </Text>
        <Text
          as="span"
          variant="body"
          size="lg"
          className={styles.description}
          shouldTruncate
          truncateLines={3}
        >
          {feature.description}
        </Text>
      </span>
    </button>
  )
}
