import { useCallback, useRef, useState } from "react";

import { Text } from "@/ds/Text/Text";
import CustomLoraIcon from "@/ds/assets/Icons/CustomLora.svg?react";
import { customLoraPosterUrl } from "@/lib/custom-lora-covers";

import {
  type HomeFeatureDefinition,
  getHomeFeature,
  offeredHomeFeatureSections,
} from "../../lib/home-features.ts";
import { HOME_LORA_GALLERY_SECTIONS } from "../../lib/home-lora-gallery.ts";
import { CUSTOM_LORA_LABEL } from "../../ltx-io/components/QuickSearch/quickSearchModel.ts";
import { useCustomLoras } from "../../ltx-io/hooks/useCustomLoras.ts";
import { useHomeFeatureFlags } from "../../ltx-io/hooks/useHomeFeatureFlags.ts";
import { customLoraRunPath } from "../../paths.ts";

import styles from "./HomeGallery.module.scss";
import {
  type GalleryHomeFeatureId,
  HOME_FEATURE_MEDIA,
  type HomeFeatureMedia,
} from "./home-media";
import {
  browserFeaturePreviewCanPlayType,
  canAttachFeaturePreviewSrc,
  silenceFeaturePreview,
} from "./homeFeaturePreviewMedia";
import { usePrefersReducedMotion } from "./usePrefersReducedMotion";

type HomeGalleryProps = {
  onOpen: (path: string) => void;
};

/** One tile of a gallery section: a Home feature, or a LoRA the user saved. */
type GalleryCard = {
  key: string;
  title: string;
  description: string;
  path: string;
  media: HomeFeatureMedia | null;
};

function featureCard(
  feature: HomeFeatureDefinition & { id: GalleryHomeFeatureId },
): GalleryCard {
  return {
    key: feature.id,
    title: feature.title,
    description: feature.description,
    path: feature.path,
    media: HOME_FEATURE_MEDIA[feature.id],
  };
}

function gallerySectionHeadingId(heading: string): string {
  const slug = heading
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "");
  return `home-gallery-${slug}`;
}

export function HomeGallery({ onOpen }: HomeGalleryProps) {
  const featureFlags = useHomeFeatureFlags();
  const savedLoras = useCustomLoras().data;
  // A chosen cover is a still image. No picture means a blank cover with the brush icon.
  const loraCards: GalleryCard[] = (savedLoras ?? []).map((lora) => {
    const posterUrl = customLoraPosterUrl(lora.cover_url);
    return {
      key: lora.id,
      title: lora.name,
      description: CUSTOM_LORA_LABEL,
      path: customLoraRunPath(lora.id),
      media: posterUrl == null ? null : { posterUrl },
    };
  });
  return (
    <div className={styles.gallery}>
      {offeredHomeFeatureSections(featureFlags).map((section) => (
        <GallerySection
          key={section.id}
          heading={section.heading}
          cards={section.features.map(featureCard)}
          onOpen={onOpen}
        />
      ))}
      {HOME_LORA_GALLERY_SECTIONS.map((section) => (
        <GallerySection
          key={section.heading}
          heading={section.heading}
          cards={section.featureIds.map((featureId) =>
            featureCard(
              getHomeFeature(featureId) as HomeFeatureDefinition & {
                id: GalleryHomeFeatureId;
              },
            ),
          )}
          onOpen={onOpen}
        />
      ))}
      {loraCards.length > 0 ? (
        <GallerySection heading="My LoRAs" cards={loraCards} onOpen={onOpen} />
      ) : null}
    </div>
  );
}

function GallerySection({
  heading,
  cards,
  onOpen,
}: {
  heading: string;
  cards: readonly GalleryCard[];
  onOpen: (path: string) => void;
}) {
  const headingId = gallerySectionHeadingId(heading);
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
        {cards.map((card) => (
          <li key={card.key}>
            <HomeSpotlightCard card={card} onOpen={() => onOpen(card.path)} />
          </li>
        ))}
      </ul>
    </section>
  );
}

function HomeSpotlightCard({ card, onOpen }: { card: GalleryCard; onOpen: () => void }) {
  const { media } = card;
  const videoRef = useRef<HTMLVideoElement>(null);
  const [previewFailed, setPreviewFailed] = useState(false);
  const [failedUrl, setFailedUrl] = useState<string | null>(null);
  const [isActive, setIsActive] = useState(false);
  const posterUrl = media?.posterUrl;
  const showPoster = posterUrl != null && failedUrl !== posterUrl;
  const reducedMotion = usePrefersReducedMotion();
  const previewUrl = media?.previewUrl;
  const canPreview =
    previewUrl != null &&
    !previewFailed &&
    !reducedMotion &&
    canAttachFeaturePreviewSrc(previewUrl, browserFeaturePreviewCanPlayType);
  const showVideo = canPreview && isActive;

  const keepSilent = useCallback(() => {
    const video = videoRef.current;
    if (!video) return;
    if (video.muted && video.volume === 0) return;
    silenceFeaturePreview(video);
  }, []);

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
      <span className={`${styles.media} ${showPoster ? "" : styles.blankCover}`}>
        {showPoster ? (
          <>
            <img
              src={posterUrl}
              alt=""
              loading="lazy"
              decoding="async"
              onError={() => setFailedUrl(posterUrl ?? null)}
            />
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
          </>
        ) : (
          <CustomLoraIcon className={styles.blankCoverIcon} aria-hidden />
        )}
      </span>
      <span className={styles.copy}>
        <Text as="span" variant="heading" size="md">
          {card.title}
        </Text>
        <Text
          as="span"
          variant="body"
          size="lg"
          className={styles.description}
          shouldTruncate
          truncateLines={3}
        >
          {card.description}
        </Text>
      </span>
    </button>
  );
}
