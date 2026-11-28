import { useEffect, useRef } from "react";
import { Button } from "@/ds/Button/Button";
import { Text } from "@/ds/Text/Text";
import { ExternalLink } from "lucide-react";
import {
  LOCAL_GENERATION_UNSUPPORTED_BODY,
  LOCAL_GENERATION_UNSUPPORTED_REQUIREMENTS_LABEL,
  LOCAL_GENERATION_UNSUPPORTED_TITLE,
  LOCAL_GENERATION_UNSUPPORTED_URLS,
  LOCAL_GENERATION_UNSUPPORTED_VIDEO_URL,
  LOCAL_GENERATION_UNSUPPORTED_WEB_APP_LABEL,
} from "../lib/local-generation-unsupported";
import { openExternalBrowserUrl } from "./home/home-external-links";
import { useHomeMediaVisibility } from "./home/useHomeMediaVisibility";
import { usePrefersReducedMotion } from "./home/usePrefersReducedMotion";
import styles from "./LocalGenerationUnsupportedNotice.module.scss";

/**
 * Cinematic card (same language as PeakCinematicCard): muted loop fills the
 * card, copy and actions overlay bottom-left on a scrim.
 */
export function LocalGenerationUnsupportedNotice() {
  const { rootRef, hasBeenVisible, inView } = useHomeMediaVisibility();
  const reducedMotion = usePrefersReducedMotion();
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    if (inView && !reducedMotion) {
      void video.play().catch(() => {});
    } else {
      video.pause();
    }
  }, [inView, reducedMotion]);

  return (
    <article ref={rootRef} className={styles.card}>
      <div className={styles.media}>
        <video
          ref={videoRef}
          className={styles.mediaEl}
          src={hasBeenVisible ? LOCAL_GENERATION_UNSUPPORTED_VIDEO_URL : undefined}
          muted
          loop
          playsInline
          preload="auto"
          aria-hidden="true"
        />
        <div className={styles.scrim} />
      </div>
      <div className={styles.content}>
        <div className={styles.textPair}>
          <Text as="h1" variant="display" size="sm" className={styles.title}>
            {LOCAL_GENERATION_UNSUPPORTED_TITLE}
          </Text>
          <Text
            as="p"
            variant="body"
            size="lg"
            shouldWhiteSpacePreLine
            className={styles.body}
          >
            {LOCAL_GENERATION_UNSUPPORTED_BODY}
          </Text>
        </div>
        <div className={styles.actions}>
          <Button
            appearance="brand"
            hierarchy="primary"
            size="lg"
            label={LOCAL_GENERATION_UNSUPPORTED_WEB_APP_LABEL}
            rightIcon={<ExternalLink aria-hidden="true" />}
            onClick={() => openExternalBrowserUrl(LOCAL_GENERATION_UNSUPPORTED_URLS.webApp)}
          />
          <Button
            appearance="overlay"
            hierarchy="secondary"
            size="lg"
            label={LOCAL_GENERATION_UNSUPPORTED_REQUIREMENTS_LABEL}
            rightIcon={<ExternalLink aria-hidden="true" />}
            onClick={() =>
              openExternalBrowserUrl(LOCAL_GENERATION_UNSUPPORTED_URLS.systemRequirements)
            }
          />
        </div>
      </div>
    </article>
  );
}
