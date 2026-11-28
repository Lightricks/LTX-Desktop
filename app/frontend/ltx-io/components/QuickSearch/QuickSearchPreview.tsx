import { Text } from "@ds/Text/Text";
import CustomLoraIcon from "@ds/assets/Icons/CustomLora.svg?react";
import ExternalLinkIcon from "@ds/assets/Icons/ExternalLink.svg?react";
import { useEffect, useRef, useState } from "react";

import { openExternalBrowserUrl } from "@/components/home/home-external-links";
import { HOME_FEATURE_MEDIA, type HomeFeatureMedia } from "@/components/home/home-media";
import {
  browserFeaturePreviewCanPlayType,
  canAttachFeaturePreviewSrc,
  silenceFeaturePreview,
} from "@/components/home/homeFeaturePreviewMedia";
import { usePrefersReducedMotion } from "@/components/home/usePrefersReducedMotion";
import { Button } from "@/ds/Button/Button";
import { customLoraPosterUrl } from "@/lib/custom-lora-covers";
import type { HomeFeatureDefinition, HomeFeatureId } from "@/lib/home-features";
import { getLoraRecipe, isLoraRecipeId } from "@/lib/lora-recipes";

import { FeatureCompareView } from "../../screens/Feature/chrome/FeatureCompareView";
import { buildFeatureDetails } from "../../screens/Feature/chrome/featureChromeModel";
import { resolveFeatureCompare } from "../../screens/Feature/chrome/featureCompare";

import styles from "./LtxioQuickSearch.module.scss";
import {
  CUSTOM_LORA_LABEL,
  type QuickSearchCustomLora,
  type QuickSearchPageDestination,
} from "./quickSearchModel.ts";

function PreviewMedia({ media }: { media: HomeFeatureMedia }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [previewFailed, setPreviewFailed] = useState(false);
  const reducedMotion = usePrefersReducedMotion();
  const previewUrl = media.previewUrl;
  const canPreview =
    previewUrl != null &&
    !previewFailed &&
    !reducedMotion &&
    canAttachFeaturePreviewSrc(previewUrl, browserFeaturePreviewCanPlayType);

  useEffect(() => {
    const video = videoRef.current;
    if (!video || !canPreview) return;
    void video.play().catch(() => undefined);
    const keepSilent = () => silenceFeaturePreview(video);
    video.addEventListener("loadeddata", keepSilent);
    video.addEventListener("play", keepSilent);
    video.addEventListener("volumechange", keepSilent);
    return () => {
      video.removeEventListener("loadeddata", keepSilent);
      video.removeEventListener("play", keepSilent);
      video.removeEventListener("volumechange", keepSilent);
    };
  }, [canPreview, previewUrl]);

  return (
    <div className={styles.previewMediaFrame}>
      <img
        className={styles.previewMedia}
        src={media.posterUrl}
        alt=""
        draggable={false}
      />
      {canPreview && previewUrl ? (
        <video
          ref={videoRef}
          className={styles.previewMediaVideo}
          src={previewUrl}
          poster={media.posterUrl}
          muted
          loop
          playsInline
          onError={() => setPreviewFailed(true)}
        />
      ) : null}
    </div>
  );
}

export function QuickSearchFeaturePreview({
  feature,
  onOpen,
  onPointerInteractionEnd,
}: {
  feature: HomeFeatureDefinition;
  onOpen: () => void;
  /** Called after a click in the preview, so the search field can take focus back. */
  onPointerInteractionEnd?: () => void;
}) {
  const featureId = feature.id as HomeFeatureId;
  const media =
    featureId === "custom-lora" ? undefined : HOME_FEATURE_MEDIA[featureId];
  const compare = resolveFeatureCompare(featureId);
  const huggingFace =
    featureId === "custom-lora" ? null : buildFeatureDetails(featureId).externalLink;
  const listing =
    "listing" in feature && feature.listing
      ? feature.listing
      : isLoraRecipeId(feature.id)
        ? getLoraRecipe(feature.id).listing
        : null;
  const description = listing?.blurb ?? feature.description;
  const categories =
    listing?.categories
      ?.split(",")
      .map((part) => part.trim())
      .filter(Boolean) ?? [];

  return (
    <aside className={styles.preview} onPointerUp={onPointerInteractionEnd}>
      <div className={styles.previewScroll}>
        {compare != null ? (
          <div className={styles.previewMediaFrame}>
            <FeatureCompareView compare={compare} title={feature.title} />
          </div>
        ) : media != null ? (
          <PreviewMedia media={media} />
        ) : null}
        <div className={styles.previewHeader}>
          <Text as="h3" variant="heading" size="sm" className={styles.previewTitle}>
            {feature.title}
          </Text>
          {listing?.typeLabel != null ? (
            <Text as="span" variant="body" size="md" className={styles.previewAbout}>
              {listing.typeLabel}
            </Text>
          ) : null}
        </div>
        {categories.length > 0 ? (
          <div className={styles.previewChips}>
            {categories.map((label) => (
              <span key={label} className={styles.keyword}>
                <Text as="span" variant="body" size="md" className={styles.keywordLabel}>
                  {label}
                </Text>
              </span>
            ))}
          </div>
        ) : null}
        {description ? (
          <Text as="p" variant="body" size="lg" className={styles.previewAbout}>
            {description}
          </Text>
        ) : null}
      </div>
      <div className={styles.previewActions}>
        {huggingFace != null ? (
          <Button
            appearance="neutral"
            hierarchy="secondary"
            size="lg"
            label={huggingFace.label}
            leftIcon={<ExternalLinkIcon />}
            className={styles.previewAction}
            onMouseDown={(event) => event.preventDefault()}
            onClick={() => openExternalBrowserUrl(huggingFace.url)}
          />
        ) : null}
        <Button
          appearance="neutral"
          hierarchy="primary"
          size="lg"
          label="Open workflow"
          className={styles.previewAction}
          onMouseDown={(event) => event.preventDefault()}
          onClick={onOpen}
        />
      </div>
    </aside>
  );
}

export function QuickSearchPagePreview({
  destination,
  onOpen,
}: {
  destination: QuickSearchPageDestination;
  onOpen: () => void;
}) {
  return (
    <aside className={styles.preview}>
      <div className={styles.previewScroll}>
        <div className={styles.previewHeader}>
          <Text as="h3" variant="heading" size="sm" className={styles.previewTitle}>
            {destination.title}
          </Text>
        </div>
        <Text as="p" variant="body" size="lg" className={styles.previewAbout}>
          {destination.description}
        </Text>
      </div>
      <div className={styles.previewActions}>
        <Button
          appearance="neutral"
          hierarchy="primary"
          size="lg"
          label={`Open ${destination.title}`}
          className={styles.previewAction}
          onMouseDown={(event) => event.preventDefault()}
          onClick={onOpen}
        />
      </div>
    </aside>
  );
}

/**
 * Preview of a LoRA the user saved. The cover is the picture they chose. With no picture, the
 * cover is blank and shows the brush icon. There is no demo clip and no intro text. The Hugging
 * Face button shows only when the LoRA came from a Hugging Face link.
 */
export function QuickSearchCustomLoraPreview({
  lora,
  onOpen,
  onPointerInteractionEnd,
}: {
  lora: QuickSearchCustomLora;
  onOpen: () => void;
  onPointerInteractionEnd?: () => void;
}) {
  const huggingFaceUrl = lora.page_url ?? null;
  const coverUrl = customLoraPosterUrl(lora.cover_url);
  const [failedUrl, setFailedUrl] = useState<string | null>(null);
  const showCover = coverUrl != null && failedUrl !== coverUrl;
  return (
    <aside className={styles.preview} onPointerUp={onPointerInteractionEnd}>
      <div className={styles.previewScroll}>
        <div className={styles.previewMediaFrame}>
          {showCover ? (
            <img
              className={styles.previewMedia}
              src={coverUrl}
              alt=""
              draggable={false}
              onError={() => setFailedUrl(coverUrl)}
            />
          ) : (
            <span className={styles.blankCover}>
              <CustomLoraIcon className={styles.blankCoverIcon} aria-hidden />
            </span>
          )}
        </div>
        <div className={styles.previewHeader}>
          <Text as="h3" variant="heading" size="sm" className={styles.previewTitle}>
            {lora.name}
          </Text>
          <Text as="span" variant="body" size="md" className={styles.previewAbout}>
            {CUSTOM_LORA_LABEL}
          </Text>
        </div>
      </div>
      <div className={styles.previewActions}>
        {huggingFaceUrl != null ? (
          <Button
            appearance="neutral"
            hierarchy="secondary"
            size="lg"
            label="Hugging Face"
            leftIcon={<ExternalLinkIcon />}
            className={styles.previewAction}
            onMouseDown={(event) => event.preventDefault()}
            onClick={() => openExternalBrowserUrl(huggingFaceUrl)}
          />
        ) : null}
        <Button
          appearance="neutral"
          hierarchy="primary"
          size="lg"
          label="Open workflow"
          className={styles.previewAction}
          onMouseDown={(event) => event.preventDefault()}
          onClick={onOpen}
        />
      </div>
    </aside>
  );
}
