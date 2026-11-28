import { clsx } from "clsx";
import { memo, useEffect, useState } from "react";

import { Skeleton } from "@ds/Skeleton/Skeleton";
import { logMessage } from "@ds/lib/handleErrors";

import styles from "./Thumbnail.module.scss";

export const THUMBNAIL_SIZES = ["xs", "sm", "md", "lg", "xl", "xxl"] as const;
export type ThumbnailSize = (typeof THUMBNAIL_SIZES)[number];

export const THUMBNAIL_RATIOS = ["1:1", "4:3", "3:4", "16:9", "9:16"] as const;
export type ThumbnailRatio = (typeof THUMBNAIL_RATIOS)[number];

export interface ThumbnailProps {
  /** The source URL of the thumbnail image. If undefined, the thumbnail will be empty and the placeholder will be shown. */
  src?: string;
  /** The alternative text for the thumbnail image */
  alt: string;
  /** The aspect ratio of the thumbnail image. Default is "1:1" */
  ratio?: ThumbnailRatio;
  /** The size of the thumbnail image. Default is "md" */
  size?: ThumbnailSize;
  /** Additional CSS classes to apply to the thumbnail */
  className?: string;
  /** Whether the thumbnail is disabled */
  isDisabled?: boolean;
  /** The URL of the video to play on hover, if applicable */
  videoUrlOnHover?: string;
}

/**
 * The Thumbnail component is used to display images in a consistent way across the application.
 * It maintains specific aspect ratios and sizes while providing a consistent visual style.
 *
 * @see {@link ThumbnailProps} for available props
 *
 * ### Usage
 * ```tsx
 * <Thumbnail
 *   src={imageUrl}
 *   alt="Description of image"
 *   ratio="1:1"
 *   size="md"
 * />
 * ```
 *
 * ### Sizes
 * - `xs`: 16px height
 * - `sm`: 24px height
 * - `md`: 32px height (default)
 * - `lg`: 44px height
 * - `xl`: 56px height
 *
 * ### Aspect Ratios
 * - `1:1`: Square
 * - `16:9`: Landscape widescreen
 * - `9:16`: Portrait widescreen
 * - `4:3`: Standard landscape
 * - `3:4`: Standard portrait
 *
 * ## Accessibility
 * - Always provide meaningful `alt` text for images
 * - Use descriptive alt text that conveys the image's purpose or content
 * - For decorative images, use an empty attribute: alt=""
 */
function ThumbnailComponent(props: ThumbnailProps) {
  const {
    src,
    alt,
    ratio = "1:1",
    size = "md",
    className,
    isDisabled,
    videoUrlOnHover,
  } = props;
  const [isLoaded, setIsLoaded] = useState(false);
  const [isError, setIsError] = useState(false);
  // Reset error when src changes
  useEffect(() => setIsError(false), [src]);

  return (
    <div
      className={clsx(styles.thumbnailContainer, className)}
      data-size={size}
      data-ratio={ratio}
      onMouseEnter={(e) => {
        if (videoUrlOnHover) {
          const videoElement = e.currentTarget.querySelector("video");
          if (videoElement) {
            videoElement.play().catch(() => {
              logMessage("Video play failure on hover", "error");
            });
            videoElement.classList.add(styles.videoPlaying);
          }
        }
      }}
      onMouseLeave={(e) => {
        e.currentTarget.classList.remove(styles.videoPlaying);
        if (videoUrlOnHover) {
          const videoElement = e.currentTarget.querySelector("video");
          if (videoElement) {
            videoElement.pause();
            videoElement.currentTime = 0;
            videoElement.classList.remove(styles.videoPlaying);
          }
        }
      }}
    >
      {!isLoaded && src && <Skeleton className={styles.skeleton} />}
      {src && !isError ? (
        <img
          data-testid={`thumbnail-${ratio}-${size}`}
          src={src}
          alt={alt}
          className={clsx(styles.thumbnail, isDisabled && styles.disabled)}
          loading="lazy"
          decoding="async"
          onLoad={() => setIsLoaded(true)}
          onError={() => {
            setIsLoaded(true);
            setIsError(true);
          }}
        />
      ) : (
        <div className={styles.placeholder} />
      )}
      {videoUrlOnHover && (
        <video className={styles.video} src={videoUrlOnHover} loop muted playsInline />
      )}
    </div>
  );
}

export const Thumbnail = memo(ThumbnailComponent);

Thumbnail.displayName = "Thumbnail";
