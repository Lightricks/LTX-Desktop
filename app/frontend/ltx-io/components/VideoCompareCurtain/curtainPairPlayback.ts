const HAVE_CURRENT_DATA = 2;
const CURTAIN_RESTART_EPSILON_SEC = 0.05;
export const CURTAIN_DRIFT_THRESHOLD_SEC = 0.08;

export function seekMedia(media: HTMLMediaElement, timeSec: number): void {
  try {
    media.currentTime = timeSec;
  } catch {
    // HAVE_NOTHING: assigning currentTime throws in some browsers.
  }
}

export function transferMediaPlayhead(
  from: HTMLMediaElement | null,
  to: HTMLMediaElement | null,
): void {
  if (!from || !to) return;
  const timeSec = from.currentTime;
  const nextTimeSec =
    Number.isFinite(to.duration) && to.duration > 0
      ? Math.min(timeSec, to.duration)
      : timeSec;
  seekMedia(to, nextTimeSec);
}

/** A play() rejected by the autoplay policy. Other rejections, such as AbortError, are not. */
export function isAutoplayBlockedError(reason: unknown): boolean {
  return (
    typeof reason === "object" &&
    reason !== null &&
    "name" in reason &&
    reason.name === "NotAllowedError"
  );
}

async function playCurtainPair(
  beforeVideo: HTMLMediaElement,
  afterVideo: HTMLMediaElement,
  onAutoplayBlocked?: () => void,
  isCancelled?: () => boolean,
): Promise<void> {
  const results = await Promise.allSettled([beforeVideo.play(), afterVideo.play()]);
  if (isCancelled?.()) return;
  if (results.every((result) => result.status === "fulfilled")) return;

  // Teardown pause() rejects in-flight play() with AbortError. Only NotAllowedError
  // is an autoplay policy block; anything else must not mute or play after unbind.
  const autoplayBlocked = results.some(
    (result) => result.status === "rejected" && isAutoplayBlockedError(result.reason),
  );
  if (!autoplayBlocked) return;

  beforeVideo.muted = true;
  afterVideo.muted = true;
  onAutoplayBlocked?.();
  if (isCancelled?.()) return;
  await Promise.allSettled([beforeVideo.play(), afterVideo.play()]);
}

export function restartCurtainPair(
  beforeVideo: HTMLMediaElement,
  afterVideo: HTMLMediaElement,
  onAutoplayBlocked?: () => void,
  isCancelled?: () => boolean,
): void {
  seekMedia(beforeVideo, 0);
  seekMedia(afterVideo, 0);
  void playCurtainPair(beforeVideo, afterVideo, onAutoplayBlocked, isCancelled);
}

function hasCurrentData(media: HTMLMediaElement): boolean {
  return media.readyState >= HAVE_CURRENT_DATA;
}

/**
 * Wire a before/after pair so both clips restart together. Native `loop`
 * uses each clip's own duration, so colorization-style pairs (off by a frame)
 * drift unless they share this policy.
 */
export function bindCurtainPairPlayback(
  beforeVideo: HTMLMediaElement,
  afterVideo: HTMLMediaElement,
  onAutoplayBlocked?: () => void,
): () => void {
  let cancelled = false;
  const playPair = () => {
    if (cancelled) return;
    restartCurtainPair(beforeVideo, afterVideo, onAutoplayBlocked, () => cancelled);
  };

  beforeVideo.loop = false;
  afterVideo.loop = false;

  beforeVideo.addEventListener("ended", playPair);
  afterVideo.addEventListener("ended", playPair);

  const tryPlayPair = () => {
    if (hasCurrentData(beforeVideo) && hasCurrentData(afterVideo)) {
      playPair();
    }
  };

  beforeVideo.addEventListener("loadeddata", tryPlayPair);
  afterVideo.addEventListener("loadeddata", tryPlayPair);
  tryPlayPair();

  return () => {
    cancelled = true;
    beforeVideo.removeEventListener("loadeddata", tryPlayPair);
    afterVideo.removeEventListener("loadeddata", tryPlayPair);
    beforeVideo.removeEventListener("ended", playPair);
    afterVideo.removeEventListener("ended", playPair);
    beforeVideo.pause();
    afterVideo.pause();
  };
}

export function syncCurtainPairFromAfter(
  beforeVideo: HTMLMediaElement | null,
  afterVideo: HTMLMediaElement | null,
  onAutoplayBlocked?: () => void,
): void {
  if (!beforeVideo || !afterVideo) return;
  syncCurtainPlayheads(beforeVideo, afterVideo, () =>
    restartCurtainPair(beforeVideo, afterVideo, onAutoplayBlocked),
  );
}

export function syncCurtainPlayheads(
  beforeVideo: HTMLMediaElement,
  afterVideo: HTMLMediaElement,
  restartPair: () => void,
): void {
  if (afterVideo.seeking || beforeVideo.seeking) return;
  if (
    beforeVideo.readyState < HAVE_CURRENT_DATA ||
    afterVideo.readyState < HAVE_CURRENT_DATA
  ) {
    return;
  }

  const minDuration = Math.min(
    Number.isFinite(beforeVideo.duration)
      ? beforeVideo.duration
      : Number.POSITIVE_INFINITY,
    Number.isFinite(afterVideo.duration) ? afterVideo.duration : Number.POSITIVE_INFINITY,
  );
  if (
    Number.isFinite(minDuration) &&
    afterVideo.currentTime >= minDuration - CURTAIN_RESTART_EPSILON_SEC
  ) {
    restartPair();
    return;
  }

  if (
    Math.abs(beforeVideo.currentTime - afterVideo.currentTime) >
    CURTAIN_DRIFT_THRESHOLD_SEC
  ) {
    seekMedia(beforeVideo, afterVideo.currentTime);
  }
}
