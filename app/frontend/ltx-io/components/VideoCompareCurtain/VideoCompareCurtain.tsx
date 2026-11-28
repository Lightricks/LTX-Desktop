import { useEffect, useRef, useState } from "react";

import { Button } from "@ds/Button/Button";
import { usePrefersReducedMotion } from "@/components/home/usePrefersReducedMotion";

import styles from "./VideoCompareCurtain.module.scss";
import { useCurtainPairPlayback } from "./useCurtainPairPlayback";

const SHOW_BEFORE_POSITION = 100;
const SHOW_AFTER_POSITION = 0;

/** Draggable before/after video comparison. Ported from the ltx-studio curtain. */
export function VideoCompareCurtain({
  beforeUrl,
  afterUrl,
  afterPosterUrl,
  beforeLabel = "Input",
  afterLabel = "Output",
  ariaLabel,
}: {
  beforeUrl: string;
  afterUrl: string;
  afterPosterUrl?: string;
  beforeLabel?: string;
  afterLabel?: string;
  ariaLabel: string;
}) {
  const [curtainPosition, setCurtainPosition] = useState(50);
  const beforeVideoRef = useRef<HTMLVideoElement>(null);
  const afterVideoRef = useRef<HTMLVideoElement>(null);
  const prefersReducedMotion = usePrefersReducedMotion();

  const syncBeforeToAfter = useCurtainPairPlayback(beforeVideoRef, afterVideoRef, {
    beforeUrl,
    afterUrl,
    enabled: !prefersReducedMotion,
  });

  useEffect(() => {
    setCurtainPosition(50);
  }, [afterUrl, beforeUrl]);

  return (
    <div className={styles.compareMedia}>
      <video
        ref={beforeVideoRef}
        className={styles.compareVideo}
        src={beforeUrl}
        muted
        playsInline
        preload="auto"
        aria-hidden
      />
      <div
        className={styles.compareAfterLayer}
        style={{ clipPath: `inset(0 0 0 ${curtainPosition}%)` }}
        aria-hidden
      >
        <video
          ref={afterVideoRef}
          className={styles.compareVideo}
          src={afterUrl}
          poster={afterPosterUrl}
          muted
          playsInline
          preload="auto"
          onTimeUpdate={syncBeforeToAfter}
        />
      </div>
      <Button
        appearance="overlay"
        hierarchy="secondary"
        size="sm"
        label={beforeLabel}
        className={styles.compareBeforeLabel}
        isActive={curtainPosition === SHOW_BEFORE_POSITION}
        aria-label={`Show ${beforeLabel}`}
        onClick={() => setCurtainPosition(SHOW_BEFORE_POSITION)}
      />
      <Button
        appearance="overlay"
        hierarchy="secondary"
        size="sm"
        label={afterLabel}
        className={styles.compareAfterLabel}
        isActive={curtainPosition === SHOW_AFTER_POSITION}
        aria-label={`Show ${afterLabel}`}
        onClick={() => setCurtainPosition(SHOW_AFTER_POSITION)}
      />
      <span
        className={styles.compareDivider}
        style={{ left: `${curtainPosition}%` }}
        aria-hidden
      >
        <span className={styles.compareHandle} />
      </span>
      <input
        type="range"
        min="0"
        max="100"
        step="1"
        value={curtainPosition}
        className={styles.compareInput}
        aria-label={ariaLabel}
        aria-valuetext={`${100 - curtainPosition}% ${afterLabel.toLowerCase()} visible`}
        onChange={(event) => setCurtainPosition(Number(event.currentTarget.value))}
      />
    </div>
  );
}
