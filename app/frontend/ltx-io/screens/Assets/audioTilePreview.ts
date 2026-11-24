export function shouldPlayAudioPreview(
  isHovered: boolean,
  shouldHoverPlay: boolean,
): boolean {
  return shouldHoverPlay && isHovered;
}

export function audioPlaybackProgress(
  currentTime: number,
  duration: number,
): number {
  if (!(duration > 0) || !Number.isFinite(duration) || !Number.isFinite(currentTime)) {
    return 0;
  }
  return Math.min(Math.max(currentTime / duration, 0), 1);
}
