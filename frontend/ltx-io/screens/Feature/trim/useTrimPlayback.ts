import { useCallback, useEffect, useState, type RefObject } from "react";

import type { TrimRange } from "./trimGeometry.ts";

/**
 * Playback that stays inside `playbackWindow`, shared by the audio and video
 * trimmers: the In/Out selection for A2V audio, the full clip for video.
 * Spread `mediaHandlers` onto the `<audio>`/`<video>`.
 */
export function useTrimPlayback(
  mediaRef: RefObject<HTMLMediaElement | null>,
  playbackWindow: TrimRange,
) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentSec, setCurrentSec] = useState(0);

  const toggleIsPlaying = useCallback(() => {
    const media = mediaRef.current;
    if (!media) return;

    if (!media.paused) {
      media.pause();
      return;
    }

    // Start (or restart) inside the selected segment.
    if (media.currentTime < playbackWindow.startSec || media.currentTime >= playbackWindow.endSec) {
      media.currentTime = playbackWindow.startSec;
      setCurrentSec(playbackWindow.startSec);
    }
    void media.play().catch(() => setIsPlaying(false));
  }, [mediaRef, playbackWindow.startSec, playbackWindow.endSec]);

  // If In/Out move while playing, keep currentTime inside the new window.
  useEffect(() => {
    const media = mediaRef.current;
    if (!media) return;
    if (media.currentTime < playbackWindow.startSec) {
      media.currentTime = playbackWindow.startSec;
      setCurrentSec(playbackWindow.startSec);
    } else if (media.currentTime > playbackWindow.endSec) {
      media.currentTime = playbackWindow.endSec;
      setCurrentSec(playbackWindow.endSec);
      if (!media.paused) media.pause();
    }
  }, [mediaRef, playbackWindow.startSec, playbackWindow.endSec]);

  const mediaHandlers = {
    onPlay: () => setIsPlaying(true),
    onPause: () => setIsPlaying(false),
    onEnded: () => setIsPlaying(false),
    onTimeUpdate: () => {
      const media = mediaRef.current;
      if (!media) return;
      // Stop at Out — the playhead stays inside the selection.
      if (media.currentTime >= playbackWindow.endSec) {
        media.pause();
        media.currentTime = playbackWindow.endSec;
        setCurrentSec(playbackWindow.endSec);
        return;
      }
      setCurrentSec(media.currentTime);
    },
  };

  return { isPlaying, currentSec, setCurrentSec, toggleIsPlaying, mediaHandlers };
}
