import {
  mediaErrorCode,
  mediaIngestErrorMessage,
  mediaLookupErrorMessage,
} from "./mediaFieldMessages.ts";

/** Mirrors the backend `ALLOWED_VIDEO_SUFFIXES` allowlist. */
export const VIDEO_EXTENSIONS = ["mp4", "webm", "mov"] as const;

const VIDEO_MIME_TYPES = new Set([
  "video/mp4",
  "video/webm",
  "video/quicktime",
]);

export const VIDEO_ACCEPT = [...VIDEO_MIME_TYPES].join(",");

export type DroppedFileLike = {
  name: string;
  type: string;
};

export function droppedFileMime(file: DroppedFileLike): string {
  return file.type.split(";")[0]?.trim().toLowerCase() ?? "";
}

export function droppedFileExtension(file: DroppedFileLike): string {
  const dot = file.name.lastIndexOf(".");
  if (dot < 0 || dot === file.name.length - 1) return "";
  return file.name.slice(dot + 1).toLowerCase();
}

export function isSupportedDroppedVideo(file: DroppedFileLike): boolean {
  const mime = droppedFileMime(file);
  if (mime.length > 0) {
    return VIDEO_MIME_TYPES.has(mime);
  }
  return VIDEO_EXTENSIONS.some((entry) => entry === droppedFileExtension(file));
}

export const UNAVAILABLE_VIDEO_MESSAGE = "This video is no longer available.";
export const UNSUPPORTED_VIDEO_MESSAGE = "This field accepts video only.";
export const COULD_NOT_ADD_VIDEO_MESSAGE = "Could not add this video.";
export const VIDEO_FILE_PATH_UNREADABLE_MESSAGE =
  "Could not read the dropped file path.";

export function videoLookupErrorMessage(error: unknown): string {
  return mediaLookupErrorMessage(
    error,
    UNAVAILABLE_VIDEO_MESSAGE,
    "Could not load this video.",
  );
}

export const COULD_NOT_TRIM_VIDEO_MESSAGE = "Could not trim this video.";
export const TRIMMED_VIDEO_TOO_LARGE_MESSAGE =
  "The trimmed video is too large. Select a shorter range.";

export const TRIM_BUSY_MESSAGE =
  "Another video is being trimmed. Try again in a moment.";

export function videoTrimErrorMessage(error: unknown): string {
  const code = mediaErrorCode(error);
  if (code === "ASSET_NOT_FOUND") {
    return UNAVAILABLE_VIDEO_MESSAGE;
  }
  if (code === "UNSUPPORTED_MEDIA") {
    return UNSUPPORTED_VIDEO_MESSAGE;
  }
  if (code === "FILE_TOO_LARGE") {
    return TRIMMED_VIDEO_TOO_LARGE_MESSAGE;
  }
  if (code === "TRIM_BUSY") {
    return TRIM_BUSY_MESSAGE;
  }
  return COULD_NOT_TRIM_VIDEO_MESSAGE;
}

export function videoIngestErrorMessage(error: unknown): string {
  return mediaIngestErrorMessage(error, {
    unsupported: UNSUPPORTED_VIDEO_MESSAGE,
    unavailable: UNAVAILABLE_VIDEO_MESSAGE,
    unreadable: [VIDEO_FILE_PATH_UNREADABLE_MESSAGE],
    couldNotAdd: COULD_NOT_ADD_VIDEO_MESSAGE,
  });
}
