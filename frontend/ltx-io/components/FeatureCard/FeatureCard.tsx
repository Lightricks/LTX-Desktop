import { clsx } from "clsx";
import React, { type MouseEvent, type ReactNode, forwardRef } from "react";

import {
  type VideoConfig,
  VideoDisplay,
} from "@ds/VideoDisplay/VideoDisplay";
import { Text } from "@ds/Text/Text";
import { useKeyboardHandlers } from "@ds/hooks/useKeyboardHandlers";
import { logMessage } from "@ds/lib/handleErrors";

import { useVideoPlayerHandlers } from "@ds/VideoDisplay/useVideoPlayerHandlers";
import { Button } from "@ds/Button/Button";

import styles from "./FeatureCard.module.scss";

export const FEATURE_CARD_STATES = ["enabled", "hover", "disabled"] as const;
export type FeatureCardState = (typeof FEATURE_CARD_STATES)[number];
export const FEATURE_CARD_WIDTH = 276;
const FEATURE_CARD_MEDIA_HEIGHT = 140;

export interface FeatureCardProps extends VideoConfig {
  /** The title of the feature card */
  title: string;
  /** The description of the feature card */
  description: string;
  /** The onClick handler */
  onClick: () => void;
  /** The state of the feature card */
  state?: FeatureCardState;
  /** The call to action button text */
  ctaText?: string;
  /** Optional click handler for the call to action button */
  onCtaClick?: () => void;
  /** Additional CSS class names to be applied */
  className?: string;
  /** Optional icon to display above the title */
  icon?: ReactNode;
  /** Card width in px. Defaults to {@link FEATURE_CARD_WIDTH}. */
  width?: number;
  /** Media region height in px. Defaults to {@link FEATURE_CARD_MEDIA_HEIGHT}. */
  mediaHeight?: number;
  /** If true, the ctaText/button will only appear on hover. Defaults to false. */
  showCtaOnHover?: boolean;
  /** If true, the badges will be shown. Defaults to false. */
  showBadges?: boolean;
  /** The badges to display */
  badges?: ReactNode[];
}

/**
 * FeatureCard component displays a fixed-width card with media content above
 * title and description text. It supports different interaction states and
 * hover playback. Width and media height are configurable, defaulting to
 * {@link FEATURE_CARD_WIDTH} / {@link FEATURE_CARD_MEDIA_HEIGHT}.
 *
 * @example
 * ```tsx
 * <FeatureCard
 *   title="AI Assistant"
 *   description="Enhance your workflow with our intelligent AI assistant"
 *   fallbackImageUrl={aiDemoImage}
 *   videoUrl={aiDemoVideo}
 *   videoId="ai-demo-123"
 *   preload="autoplay"
 *   onClick={() => console.log('Feature card clicked')}
 *   state="enabled"
 *   type="tool"
 *   icon={<AIIcon />}
 * />
 * ```
 *
 * @states
 * - enabled: Default state, card is interactive and clickable
 * - hover: Visual state when user hovers over the card
 * - skeleton: Loading state showing a placeholder animation
 *
 * @types
 * - Default type: fixed-width card with the media region above the copy.
 *
 * @accessibility
 * - The arrow button includes an aria-label for screen readers
 * - Interactive elements are keyboard navigable
 * - Video content should include appropriate alt text via the MediaPlayer component
 *
 * @performance
 * - Videos are lazy loaded through the MediaPlayer component
 * - Fallback image is provided for scenarios where video cannot be played
 */
export const FeatureCard = forwardRef<HTMLDivElement, FeatureCardProps>(
  (props, ref): JSX.Element => {
    const {
      onClick,
      state = "enabled",
      title,
      description,
      videoUrl,
      videoId,
      fallbackImageUrl,
      preload = "eager",
      className,
      icon,
      ctaText,
      onCtaClick,
      width = FEATURE_CARD_WIDTH,
      mediaHeight = FEATURE_CARD_MEDIA_HEIGHT,
      showCtaOnHover = false,
      showBadges = false,
      badges,
    } = props;

    const { videoPlayerRef, handlers } = useVideoPlayerHandlers();
    const { handleButtonKeyDown } = useKeyboardHandlers();
    const isDisabled = state === "disabled";

    const withDisableHandlerCheck = (handler: () => void) => () => {
      if (isDisabled) return;
      handler();
    };
    const handleCtaClick = (event: MouseEvent<HTMLButtonElement>) => {
      event.stopPropagation();
      if (isDisabled) return;
      onCtaClick?.();
    };

    // The CTA renders a real <button>. A role="button" ancestor containing a
    // button is invalid nested-interactive content, so the card only takes the
    // button role when it has no interactive children; with a CTA the button is
    // the focusable affordance and the card click is a mouse-only convenience.
    const hasInteractiveContent = Boolean(ctaText);
    const interactionHandlers = {
      onMouseEnter: withDisableHandlerCheck(handlers.handleMouseEnter),
      onMouseLeave: withDisableHandlerCheck(
        handlers.handleMouseLeaveWithVideoReset,
      ),
      onFocus: withDisableHandlerCheck(handlers.handleFocus),
      onBlur: withDisableHandlerCheck(handlers.handleBlur),
      onClick: withDisableHandlerCheck(onClick),
      onKeyDown:
        isDisabled || hasInteractiveContent ? undefined : handleButtonKeyDown,
    };

    return (
      <div
        ref={ref}
        role={hasInteractiveContent ? undefined : "button"}
        tabIndex={isDisabled || hasInteractiveContent ? undefined : 0}
        aria-disabled={isDisabled}
        aria-label={
          hasInteractiveContent ? undefined : `${title} - ${description}`
        }
        className={clsx(styles.container, styles[state], className)}
        data-state={state}
        data-testid="feature-card"
        style={{
          width,
          flexBasis: width,
        }}
        {...interactionHandlers}
      >
        <div className={styles.mediaFrame} data-testid="feature-card-media">
          <VideoDisplay
            ref={videoPlayerRef}
            videoUrl={videoUrl}
            fallbackImageUrl={fallbackImageUrl}
            videoId={videoId}
            preload={preload}
            fullWidth
            size={{ width, height: mediaHeight }}
            className={styles.media}
            onError={() => {
              logMessage("Media playback error", "error");
            }}
          />

          {showBadges && (
            <div className={styles.badgesContainer}>
              {badges?.map((badge, index) => (
                <React.Fragment key={index}>{badge}</React.Fragment>
              ))}
            </div>
          )}

          {ctaText && (
            <div
              className={clsx(
                styles.ctaContainer,
                !showCtaOnHover && styles.alwaysShow,
              )}
            >
              <Button
                appearance="overlay"
                hierarchy="secondary"
                size="lg"
                label={ctaText}
                disabled={isDisabled}
                className={styles.ctaButton}
                onClick={handleCtaClick}
              />
            </div>
          )}
        </div>

        <div className={styles.content} data-testid="feature-card-content">
          {icon && <div className={styles.icon}>{icon}</div>}
          <div className={styles.copy}>
            <Text
              variant="label"
              size="xl"
              as="h3"
              shouldTruncate
              className={styles.title}
            >
              {title}
            </Text>
            <Text
              variant="body"
              size="lg"
              as="p"
              shouldTruncate
              truncateLines={2}
              className={styles.description}
            >
              {description}
            </Text>
          </div>
        </div>
      </div>
    );
  },
);

FeatureCard.displayName = "FeatureCard";
