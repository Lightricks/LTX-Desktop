/**
 * Shared waveform bar style matching the DG (Design Guidelines) audio tile design.
 * Used across GenSpace audio tiles, Asset Library tiles, and the Lightbox player.
 */
export const AUDIO_WAVEFORM_STYLE = {
  barWidth: 2.5,
  barGap: 2,
  minBarHeight: 2,
  maxBarHeight: 28,
  lineCap: "round" as CanvasLineCap,
} as const;

export const WAVEFORM_PLAYED_COLORS = {
  dark: "#F3F4F5",
  light: "#28292C",
} as const;

export const WAVEFORM_UNPLAYED_COLORS = {
  dark: "#5C5E63",
  light: "#C0C2C5",
} as const;
