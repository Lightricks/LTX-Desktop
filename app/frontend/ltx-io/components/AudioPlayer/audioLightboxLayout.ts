export const AUDIO_LIGHTBOX_CONTROLS_HEIGHT = 56;
export const AUDIO_LIGHTBOX_WAVEFORM_HEIGHT_RATIO = 0.5;
export const AUDIO_LIGHTBOX_MIN_WAVEFORM_HEIGHT = 200;

export function audioLightboxWaveformHeight(areaHeight: number): number {
  const availableHeight = Math.max(0, areaHeight - AUDIO_LIGHTBOX_CONTROLS_HEIGHT);
  return Math.max(
    AUDIO_LIGHTBOX_MIN_WAVEFORM_HEIGHT,
    Math.floor(availableHeight * AUDIO_LIGHTBOX_WAVEFORM_HEIGHT_RATIO),
  );
}
