import { type RefObject, useCallback, useEffect, useRef, useState } from "react";

import { VideoMuteButton } from "@ds/VideoMuteButton/VideoMuteButton";
import { usePrefersReducedMotion } from "@/components/home/usePrefersReducedMotion";

import { SlidingSegmentControl } from "../SlidingSegmentControl/SlidingSegmentControl";
import {
  isAutoplayBlockedError,
  transferMediaPlayhead,
} from "../VideoCompareCurtain/curtainPairPlayback";

import styles from "./SourceResultToggle.module.scss";

type Side = "source" | "result";

export type SourceResultClip = {
  videoUrl?: string;
  /** A video shows its first frame until it plays, so its poster is optional. */
  posterUrl?: string;
};

const SEGMENTS: { value: Side; label: string }[] = [
  { value: "source", label: "Source" },
  { value: "result", label: "Result" },
];

/**
 * One side of the toggle. A video that fails to load falls back to its poster.
 * Only the active side preloads in full.
 */
function ClipLayer({
  clip,
  videoRef,
  isActive,
  isMuted,
}: {
  clip: SourceResultClip;
  videoRef: RefObject<HTMLVideoElement>;
  isActive: boolean;
  isMuted: boolean;
}) {
  const [failed, setFailed] = useState(false);

  if (clip.videoUrl == null || failed) {
    return clip.posterUrl != null ? (
      <img
        className={styles.layer}
        data-active={isActive}
        src={clip.posterUrl}
        alt=""
        draggable={false}
        aria-hidden={!isActive}
      />
    ) : null;
  }

  return (
    <video
      ref={videoRef}
      className={styles.layer}
      data-active={isActive}
      src={clip.videoUrl}
      poster={clip.posterUrl}
      muted={isMuted}
      loop
      playsInline
      preload={isActive ? "auto" : "metadata"}
      aria-hidden={!isActive}
      onError={() => setFailed(true)}
    />
  );
}

/**
 * Switches between what the user puts in and what the feature makes.
 * The source is a start frame or a video. The result is a video.
 * A video source and the result keep the same playhead when you switch.
 * Give it a `key` that changes with the clip, so a new clip starts on Result.
 */
export function SourceResultToggle({
  source,
  result,
  ariaLabel,
}: {
  source: SourceResultClip;
  result: SourceResultClip & { videoUrl: string };
  ariaLabel: string;
}) {
  const [side, setSide] = useState<Side>("result");
  const [isMuted, setIsMuted] = useState(true);
  const sourceVideoRef = useRef<HTMLVideoElement>(null);
  const resultVideoRef = useRef<HTMLVideoElement>(null);
  const prefersReducedMotion = usePrefersReducedMotion();
  const activeHasVideo = side === "source" ? source.videoUrl != null : true;

  useEffect(() => {
    if (prefersReducedMotion) return;
    const active = side === "source" ? sourceVideoRef.current : resultVideoRef.current;
    const inactive = side === "source" ? resultVideoRef.current : sourceVideoRef.current;
    inactive?.pause();
    void active?.play().catch((error: unknown) => {
      // Switching sides aborts the pending play of the other video. Ignore that.
      if (!isAutoplayBlockedError(error) || !active) return;
      active.muted = true;
      setIsMuted(true);
      void active.play().catch(() => undefined);
    });
  }, [side, prefersReducedMotion]);

  const handleSideChange = useCallback(
    (next: Side) => {
      const from = side === "source" ? sourceVideoRef.current : resultVideoRef.current;
      const to = next === "source" ? sourceVideoRef.current : resultVideoRef.current;
      transferMediaPlayhead(from, to);
      setSide(next);
    },
    [side],
  );

  return (
    <div className={styles.root}>
      <ClipLayer
        clip={source}
        videoRef={sourceVideoRef}
        isActive={side === "source"}
        isMuted={isMuted}
      />
      <ClipLayer
        clip={result}
        videoRef={resultVideoRef}
        isActive={side === "result"}
        isMuted={isMuted}
      />
      {activeHasVideo && !prefersReducedMotion ? (
        <div className={styles.mute} onMouseDown={(event) => event.preventDefault()}>
          <VideoMuteButton
            isMuted={isMuted}
            onToggle={() => setIsMuted((muted) => !muted)}
          />
        </div>
      ) : null}
      <div className={styles.toggle} onMouseDown={(event) => event.preventDefault()}>
        <SlidingSegmentControl<Side>
          size="sm"
          className={styles.toggleControl}
          aria-label={`${ariaLabel}: source or result`}
          segments={SEGMENTS}
          selectedSegment={side}
          onSegmentChanged={handleSideChange}
        />
      </div>
    </div>
  );
}
