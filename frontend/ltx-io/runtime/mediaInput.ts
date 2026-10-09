import { AUDIO_EXTENSIONS } from "../screens/Feature/fields/audioAssetInput.ts";
import { VIDEO_EXTENSIONS } from "../screens/Feature/fields/videoAssetInput.ts";
import { IMAGE_EXTENSIONS } from "../screens/Feature/fields/imageAssetInput.ts";
import type { ExploreMediaChooseRequest } from "./ExploreRuntime.tsx";

/** Field-facing fallback when a pick matches neither the accept list nor the extensions. */
export const UNSUPPORTED_MEDIA_MESSAGE = "This file type is not supported.";

export type MediaFileLike = {
  name: string;
  type: string;
};

function requestMimeTypes(request: ExploreMediaChooseRequest): string[] {
  return request.accept
    .split(",")
    .map((entry) => entry.split(";")[0]?.trim().toLowerCase() ?? "")
    .filter((entry) => entry.length > 0);
}

function fileMimeType(file: MediaFileLike): string {
  return file.type.split(";")[0]?.trim().toLowerCase() ?? "";
}

function fileExtension(file: MediaFileLike): string {
  const dot = file.name.lastIndexOf(".");
  if (dot < 0 || dot === file.name.length - 1) return "";
  return file.name.slice(dot + 1).toLowerCase();
}

function requestHasSuffix(
  request: Pick<ExploreMediaChooseRequest, "extensions">,
  suffixes: readonly string[],
): boolean {
  return suffixes.some((suffix) => request.extensions.includes(suffix));
}

/**
 * Whether a picked file satisfies a media choose request. A reported mime
 * must be listed in `accept`; picks without one (Desktop dialog paths) fall
 * back to the request extensions. One matcher serves every field, so the
 * runtime needs no per-kind validation switch.
 */
export function isAcceptedMediaFile(
  file: MediaFileLike,
  request: ExploreMediaChooseRequest,
): boolean {
  const mime = fileMimeType(file);
  if (mime.length > 0) {
    return requestMimeTypes(request).includes(mime);
  }
  return request.extensions.includes(fileExtension(file));
}

/** Whether the request offers a video suffix. */
export function requestAcceptsVideo(
  request: Pick<ExploreMediaChooseRequest, "extensions">,
): boolean {
  return requestHasSuffix(request, VIDEO_EXTENSIONS);
}

/** Whether the request offers an audio suffix. */
export function requestAcceptsAudio(
  request: Pick<ExploreMediaChooseRequest, "extensions">,
): boolean {
  return requestHasSuffix(request, AUDIO_EXTENSIONS);
}

function requestAcceptsImage(
  request: Pick<ExploreMediaChooseRequest, "extensions">,
): boolean {
  return requestHasSuffix(request, IMAGE_EXTENSIONS);
}

/**
 * A2V import affordance: a video pick should resolve to its extracted audio
 * track only when the field wants audio and also accepts video as a source.
 * Video-only fields (Retake / Extend) ingest the clip as video.
 */
export function requestExtractsAudioFromVideo(
  request: ExploreMediaChooseRequest,
): boolean {
  return requestAcceptsAudio(request) && requestAcceptsVideo(request);
}

/** Human name for the native dialog filter, derived from the same suffixes as the predicates. */
export function describeMediaFilter(
  request: Pick<ExploreMediaChooseRequest, "extensions">,
): string {
  const hasAudio = requestAcceptsAudio(request);
  const hasVideo = requestAcceptsVideo(request);
  const hasImage = requestAcceptsImage(request);
  if (hasAudio && hasVideo) {
    return "Audio or video";
  }
  if (hasAudio) {
    return "Audio";
  }
  if (hasVideo) {
    return "Video";
  }
  if (hasImage) {
    return "Images";
  }
  return "Media";
}
