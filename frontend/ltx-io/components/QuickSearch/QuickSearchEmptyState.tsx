import { useEffect, useRef } from "react";

import { Text } from "@ds/Text/Text";
import { PEAK_CINEMATIC_EXAMPLES } from "@/components/home/home-media";
import { usePrefersReducedMotion } from "@/components/home/usePrefersReducedMotion";

import styles from "./LtxioQuickSearch.module.scss";

const EMPTY_STATE_CLIPS = PEAK_CINEMATIC_EXAMPLES.slice(0, 3);

export function QuickSearchEmptyState() {
  return (
    <aside className={styles.previewReelPane}>
      <div className={styles.featureReel}>
        <Text as="h3" variant="heading" size="xxl" className={styles.previewHeadline}>
          New in LTX-2.5
        </Text>
        {EMPTY_STATE_CLIPS.map((clip) => (
          <FeatureClip key={clip.id} clip={clip} />
        ))}
      </div>
    </aside>
  );
}

function FeatureClip({
  clip,
}: {
  clip: (typeof EMPTY_STATE_CLIPS)[number];
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const reducedMotion = usePrefersReducedMotion();

  useEffect(() => {
    const video = videoRef.current;
    if (!video || reducedMotion || clip.videoUrl == null) return;

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          void video.play().catch(() => undefined);
        } else {
          video.pause();
        }
      },
      { threshold: 0.5 },
    );
    observer.observe(video);
    return () => observer.disconnect();
  }, [clip.videoUrl, reducedMotion]);

  if (clip.posterUrl == null) return null;

  return (
    <div className={styles.featureClip}>
      {reducedMotion || clip.videoUrl == null ? (
        <img
          className={styles.previewMedia}
          src={clip.posterUrl}
          alt=""
          draggable={false}
        />
      ) : (
        <video
          ref={videoRef}
          className={styles.previewMedia}
          src={clip.videoUrl}
          poster={clip.posterUrl}
          muted
          loop
          playsInline
          preload="metadata"
        />
      )}
    </div>
  );
}
