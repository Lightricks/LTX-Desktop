import { clsx } from "clsx";
import { useEffect, useMemo, useRef, useState } from "react";
import styles from "./PeakCinematicCard.module.scss";
import type { PeakCinematicExample } from "./peakCinematic";
import {
  peakClipStates,
  peakVideoPreload,
  nextPeakPlayableIndex,
  resolvePeakActiveIndex,
  shouldRestorePeakPoster,
} from "./peakCinematicPlaylistMedia";
import { usePrefersReducedMotion } from "./usePrefersReducedMotion";

function videoRefCallback(
  refs: Map<string, HTMLVideoElement>,
  callbacks: Map<string, (node: HTMLVideoElement | null) => void>,
  id: string,
): (node: HTMLVideoElement | null) => void {
  let callback = callbacks.get(id);
  if (!callback) {
    callback = (node) => {
      if (node) refs.set(id, node);
      else refs.delete(id);
    };
    callbacks.set(id, callback);
  }
  return callback;
}

function safePlay(video: HTMLVideoElement) {
  try {
    video.muted = true;
    void video.play().catch(() => {});
  } catch {
    // Poster remains the fallback when play is rejected or throws.
  }
}

/**
 * Drop any remaining `src` and reload so the poster replaces the last decoded
 * frame. React may already have removed the attribute, so the reload is
 * unconditional.
 */
function restorePoster(video: HTMLVideoElement) {
  try {
    if (video.getAttribute("src")) video.removeAttribute("src");
    video.load();
  } catch {
    // Keep the element; the poster attribute is still the usable fallback.
  }
}

/**
 * Stacked peak playlist — owns play/pause/advance and src attach.
 * Always muted; the home showreel has no mute or progress chrome.
 */
export function PeakCinematicPlaylist({
  examples,
  hasBeenVisible,
  inView,
}: {
  examples: PeakCinematicExample[];
  hasBeenVisible: boolean;
  inView: boolean;
}) {
  const hasPlaylist = examples.length > 1;
  const hasMedia = examples.length > 0;
  const reducedMotion = usePrefersReducedMotion();

  const videoRefs = useRef<Map<string, HTMLVideoElement>>(new Map());
  const videoRefCallbacks = useRef(
    new Map<string, (node: HTMLVideoElement | null) => void>(),
  );
  /** Clip ids that carried a `src` on the previous committed render. */
  const attachedIds = useRef<Set<string>>(new Set());
  const [activeIndex, setActiveIndex] = useState(0);
  const [failedIds, setFailedIds] = useState(() => new Set<string>());

  const resolvedActiveIndex = resolvePeakActiveIndex(
    activeIndex,
    examples.length,
  );
  const activeExampleId = examples[resolvedActiveIndex]?.id;

  /** One active/attach decision per render, shared by the effect and render. */
  const clipStates = useMemo(
    () =>
      peakClipStates({
        ids: examples.map((example) => example.id),
        activeIndex,
        hasBeenVisible,
        inView,
        reducedMotion,
        failedIds,
      }),
    [examples, activeIndex, hasBeenVisible, inView, reducedMotion, failedIds],
  );

  const markFailed = (id: string) => {
    setFailedIds((current) => {
      if (current.has(id)) return current;
      const next = new Set(current);
      next.add(id);
      return next;
    });
  };

  const handleClipError = (exampleId: string) => {
    markFailed(exampleId);
    const video = videoRefs.current.get(exampleId);
    if (video) restorePoster(video);
  };

  useEffect(() => {
    if (activeIndex >= examples.length) {
      setActiveIndex(0);
    }
  }, [activeIndex, examples.length]);

  useEffect(() => {
    if (reducedMotion) {
      setActiveIndex(0);
    }
  }, [reducedMotion]);

  useEffect(() => {
    if (reducedMotion || !activeExampleId || !failedIds.has(activeExampleId)) {
      return;
    }

    const nextIndex = nextPeakPlayableIndex(
      examples.map((example) => example.id),
      resolvedActiveIndex,
      failedIds,
    );
    if (nextIndex === null || nextIndex === resolvedActiveIndex) return;
    setActiveIndex(nextIndex);
  }, [
    activeExampleId,
    examples,
    failedIds,
    reducedMotion,
    resolvedActiveIndex,
  ]);

  useEffect(() => {
    if (!hasBeenVisible || !hasMedia) return;

    const previouslyAttached = attachedIds.current;
    const nextAttached = new Set<string>();

    for (const clip of clipStates) {
      // Recorded before the ref lookup: a clip whose element is momentarily
      // missing still keeps its attachment state, so the poster restoration
      // below cannot be skipped on a later render.
      if (clip.attachSrc) nextAttached.add(clip.id);

      const video = videoRefs.current.get(clip.id);
      if (!video) continue;

      // React has already dropped the `src` attribute by the time this effect
      // runs, so the previous attachment is the only reliable signal.
      if (
        shouldRestorePeakPoster(previouslyAttached.has(clip.id), clip.attachSrc)
      ) {
        restorePoster(video);
      }

      if (reducedMotion || !inView) {
        try {
          video.pause();
        } catch {
          // Ignore pause failures; poster remains visible.
        }
        continue;
      }

      if (clip.isActive) {
        safePlay(video);
      } else {
        try {
          video.pause();
        } catch {
          // Ignore pause failures on inactive clips.
        }
      }
    }

    attachedIds.current = nextAttached;
  }, [clipStates, hasBeenVisible, hasMedia, inView, reducedMotion]);

  const handleEnded = (exampleId: string) => {
    if (reducedMotion) return;
    if (exampleId !== activeExampleId) return;
    const nextIndex = nextPeakPlayableIndex(
      examples.map((example) => example.id),
      resolvedActiveIndex,
      failedIds,
    );
    if (nextIndex === null) {
      const video = videoRefs.current.get(exampleId);
      if (video) restorePoster(video);
      return;
    }
    if (nextIndex === resolvedActiveIndex) {
      const video = videoRefs.current.get(exampleId);
      if (!video) return;
      try {
        video.currentTime = 0;
      } catch {
        // Ignore seek failures when replaying the only playable clip.
      }
      safePlay(video);
      return;
    }
    if (nextIndex === 0) {
      for (const example of examples) {
        const video = videoRefs.current.get(example.id);
        if (!video || video.readyState <= 0) continue;
        try {
          video.currentTime = 0;
        } catch {
          // Ignore seek failures when looping the playlist.
        }
      }
    }
    setActiveIndex(nextIndex);
  };

  return (
    <div className={styles.media}>
      {examples.map((example, index) => {
        const { isActive, attachSrc } = clipStates[index];
        return (
          <video
            key={example.id}
            ref={videoRefCallback(
              videoRefs.current,
              videoRefCallbacks.current,
              example.id,
            )}
            className={clsx(
              styles.mediaEl,
              styles.mediaElStacked,
              isActive ? styles.mediaElActive : styles.mediaElHidden,
            )}
            src={attachSrc ? example.videoUrl : undefined}
            poster={example.posterUrl}
            muted
            loop={!hasPlaylist}
            playsInline
            preload={peakVideoPreload(isActive, attachSrc)}
            aria-hidden
            onEnded={isActive ? () => handleEnded(example.id) : undefined}
            onError={() => handleClipError(example.id)}
            onLoadedData={
              isActive
                ? () => {
                    if (reducedMotion || !inView || !hasBeenVisible) return;
                    const video = videoRefs.current.get(example.id);
                    if (!video) return;
                    safePlay(video);
                  }
                : undefined
            }
          />
        );
      })}
      <div className={styles.scrim} aria-hidden />
    </div>
  );
}
