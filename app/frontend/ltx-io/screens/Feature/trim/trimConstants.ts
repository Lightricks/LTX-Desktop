import cap from "../../../../../shared/ic-lora-input-cap.json" with { type: "json" };

/** Ported from LTX.io `constants/trim.ts`, reduced to what the trim modals use. */

/** Video caps forgive container rounding; a2v audio stays exact because it sets frame counts. */
export const VIDEO_DURATION_CAP_TOLERANCE_SECONDS = cap.durationToleranceSeconds;

/**
 * Longest input at 24fps. A higher selected rate shortens this to
 * 240 frames / fps. Shared with create via shared/ic-lora-input-cap.json.
 */
export const MAX_IC_LORA_INPUT_VIDEO_SECONDS = cap.maxInputVideoSeconds;

/** Shortest video selection; the video features take clips down to 2s. */
export const MIN_VIDEO_TRIM_DURATION_SECONDS = 2;

/**
 * Trim selection floor. Generation now offers 2s clips; the trimmer stays at
 * 6s so a short trim is not the only way to get a short clip.
 */
export const MIN_TRIM_DURATION_SECONDS = 6;

/** Compact track height for the modal audio trimmer (waveform bars inset inside). */
export const CLIP_HEIGHT_AUDIO_COMPACT = 48;

/** Vertical inset so waveform bars don't touch the track top/bottom. */
export const WAVEFORM_VERTICAL_INSET = 8;

/** Width of each In/Out grab handle; the track reserves a gutter of this size per side. */
export const RESIZE_HANDLE_WIDTH = 12;

/** Selection outline. The needle overlaps it and paints on top. */
export const SELECTION_BORDER_PX = 2;

/**
 * Playhead grabber width (centered with a negative margin). Keep in sync with
 * `TrimSeekBar.module.scss` `.handle`.
 */
export const PLAYHEAD_HANDLE_WIDTH_PX = 13;

/** The needle's shield head sits above the track; the trimmer reserves this much room for it. */
export const PLAYHEAD_HEAD_HEIGHT_PX = 13;

export const TRIMMER_CONTAINER_WIDTH = 640;

export const TRIM_AUDIO_MODAL_NAME = "trim-audio-modal";

export const TRIM_VIDEO_MODAL_NAME = "trim-video-modal";
