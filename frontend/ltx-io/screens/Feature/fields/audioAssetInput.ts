import {
  mediaErrorCode,
  mediaIngestErrorMessage,
  mediaLookupErrorMessage,
} from "./mediaFieldMessages.ts";
import {
  VIDEO_ACCEPT,
  VIDEO_EXTENSIONS,
  droppedFileExtension,
  droppedFileMime,
  type DroppedFileLike,
} from "./videoAssetInput.ts";

/** Compat: `VIDEO_*` now lives in `videoAssetInput.ts`. */
export { VIDEO_ACCEPT, VIDEO_EXTENSIONS, isSupportedDroppedVideo } from "./videoAssetInput.ts";
export type { DroppedFileLike };

/** Mirrors the backend `ALLOWED_AUDIO_SUFFIXES` allowlist. */
export const AUDIO_EXTENSIONS = ["wav", "mp3", "m4a"] as const;

const AUDIO_MIME_TYPES = new Set([
  "audio/wav",
  "audio/x-wav",
  "audio/wave",
  "audio/mpeg",
  "audio/mp3",
  "audio/mp4",
  "audio/x-m4a",
]);

export const AUDIO_ACCEPT = [...AUDIO_MIME_TYPES].join(",");
export const AUDIO_OR_VIDEO_ACCEPT = `${AUDIO_ACCEPT},${VIDEO_ACCEPT}`;
export const AUDIO_OR_VIDEO_EXTENSIONS = [
  ...AUDIO_EXTENSIONS,
  ...VIDEO_EXTENSIONS,
] as const;

export const UNAVAILABLE_AUDIO_MESSAGE = "This audio is no longer available.";
export const UNSUPPORTED_AUDIO_MESSAGE = "This field accepts audio only.";
export const COULD_NOT_ADD_AUDIO_MESSAGE = "Could not add this audio.";
export const AUDIO_FILE_TOO_LARGE_MESSAGE = "This file is too large.";
export const VIDEO_HAS_NO_AUDIO_MESSAGE = "This video has no audio track.";
export const VIDEO_AUDIO_UNSUPPORTED_MESSAGE =
  "Extracting audio from video is only available in LTX Desktop.";
export const AUDIO_FILE_PATH_UNREADABLE_MESSAGE =
  "Could not read the dropped file path.";

export function isSupportedDroppedAudio(file: DroppedFileLike): boolean {
  const mime = droppedFileMime(file);
  if (mime.length > 0) {
    return AUDIO_MIME_TYPES.has(mime);
  }
  return AUDIO_EXTENSIONS.some((entry) => entry === droppedFileExtension(file));
}

export function audioLookupErrorMessage(error: unknown): string {
  return mediaLookupErrorMessage(
    error,
    UNAVAILABLE_AUDIO_MESSAGE,
    "Could not load this audio.",
  );
}

export function audioIngestErrorMessage(error: unknown): string {
  const code = mediaErrorCode(error);
  if (code === "FILE_TOO_LARGE") {
    return AUDIO_FILE_TOO_LARGE_MESSAGE;
  }
  if (code === "NO_AUDIO_STREAM") {
    return VIDEO_HAS_NO_AUDIO_MESSAGE;
  }
  return mediaIngestErrorMessage(error, {
    unsupported: UNSUPPORTED_AUDIO_MESSAGE,
    unavailable: UNAVAILABLE_AUDIO_MESSAGE,
    unreadable: [
      AUDIO_FILE_PATH_UNREADABLE_MESSAGE,
      VIDEO_AUDIO_UNSUPPORTED_MESSAGE,
    ],
    couldNotAdd: COULD_NOT_ADD_AUDIO_MESSAGE,
  });
}

export const COULD_NOT_TRIM_AUDIO_MESSAGE = "Could not trim this audio.";

export function audioTrimErrorMessage(error: unknown): string {
  const code = mediaErrorCode(error);
  if (code === "ASSET_NOT_FOUND") {
    return UNAVAILABLE_AUDIO_MESSAGE;
  }
  if (code === "UNSUPPORTED_MEDIA") {
    return UNSUPPORTED_AUDIO_MESSAGE;
  }
  return COULD_NOT_TRIM_AUDIO_MESSAGE;
}
