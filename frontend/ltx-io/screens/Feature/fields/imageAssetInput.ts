import {
  mediaIngestErrorMessage,
  mediaLookupErrorMessage,
} from "./mediaFieldMessages.ts";

export const IMAGE_EXTENSIONS = ["png", "jpg", "jpeg", "webp"] as const;

const IMAGE_MIME_TYPES = new Set([
  "image/jpeg",
  "image/png",
  "image/webp",
]);

export const IMAGE_ACCEPT = [...IMAGE_MIME_TYPES].join(",");

export type DroppedFileLike = {
  name: string;
  type: string;
};

export const UNAVAILABLE_IMAGE_MESSAGE = "This image is no longer available.";
export const UNSUPPORTED_IMAGE_MESSAGE = "This field accepts images only.";
export const COULD_NOT_ADD_IMAGE_MESSAGE = "Could not add this image.";
export const DROPPED_FILE_PATH_UNREADABLE_MESSAGE =
  "Could not read the dropped file path.";

function droppedFileMime(file: DroppedFileLike): string {
  return file.type.split(";")[0]?.trim().toLowerCase() ?? "";
}

function droppedFileExtension(file: DroppedFileLike): string {
  const dot = file.name.lastIndexOf(".");
  if (dot < 0 || dot === file.name.length - 1) return "";
  return file.name.slice(dot + 1).toLowerCase();
}

export function isSupportedDroppedImage(file: DroppedFileLike): boolean {
  const mime = droppedFileMime(file);
  if (mime.length > 0) {
    return IMAGE_MIME_TYPES.has(mime);
  }
  return IMAGE_EXTENSIONS.some((entry) => entry === droppedFileExtension(file));
}

export function prepareDroppedImageIngest(
  file: DroppedFileLike,
  getPathForFile: (file: DroppedFileLike) => string | null,
): { ingestPath: string } | { error: string } {
  if (!isSupportedDroppedImage(file)) {
    return { error: UNSUPPORTED_IMAGE_MESSAGE };
  }
  const ingestPath = getPathForFile(file);
  if (!ingestPath) {
    return { error: DROPPED_FILE_PATH_UNREADABLE_MESSAGE };
  }
  return { ingestPath };
}

export function imageLookupErrorMessage(error: unknown): string {
  return mediaLookupErrorMessage(
    error,
    UNAVAILABLE_IMAGE_MESSAGE,
    "Could not load this image.",
  );
}

export function imageIngestErrorMessage(error: unknown): string {
  return mediaIngestErrorMessage(error, {
    unsupported: UNSUPPORTED_IMAGE_MESSAGE,
    unavailable: UNAVAILABLE_IMAGE_MESSAGE,
    unreadable: [DROPPED_FILE_PATH_UNREADABLE_MESSAGE],
    couldNotAdd: COULD_NOT_ADD_IMAGE_MESSAGE,
  });
}
