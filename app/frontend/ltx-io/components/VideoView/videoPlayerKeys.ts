/** Chromium's media timeline is a range input. Its default step is one second. */
export const NATIVE_VIDEO_SEEK_STEP_SECONDS = 1;

export type VideoPlayerKeyAction =
  | "toggle-playback"
  | "seek-backward"
  | "seek-forward";

/** Space on a focused control button is that button's click. */
export function shouldIgnorePlayerToggle(
  action: VideoPlayerKeyAction,
  targetIsButton: boolean,
): boolean {
  return action === "toggle-playback" && targetIsButton;
}

export function videoPlayerKeyAction(key: string): VideoPlayerKeyAction | null {
  switch (key) {
    case " ":
    case "Spacebar":
      return "toggle-playback";
    case "ArrowLeft":
      return "seek-backward";
    case "ArrowRight":
      return "seek-forward";
    default:
      return null;
  }
}

export function seekByNativeVideoStep(
  currentTime: number,
  duration: number,
  direction: -1 | 1,
): number {
  const next = currentTime + direction * NATIVE_VIDEO_SEEK_STEP_SECONDS;
  if (!Number.isFinite(next)) return 0;
  const upper =
    Number.isFinite(duration) && duration > 0 ? duration : Number.POSITIVE_INFINITY;
  return Math.min(Math.max(0, next), upper);
}
