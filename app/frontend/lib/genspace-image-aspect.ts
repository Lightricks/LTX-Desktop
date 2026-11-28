/** GenSpace image picker. Video ratios outside this set stay on the video control. */
export const GENSPACE_IMAGE_ASPECT_RATIOS = ["16:9", "1:1", "9:16"] as const;

export type GenSpaceImageAspectRatio = (typeof GENSPACE_IMAGE_ASPECT_RATIOS)[number];

export function genSpaceImageAspectRatio(
  value: string | undefined,
): GenSpaceImageAspectRatio {
  if (
    value != null &&
    (GENSPACE_IMAGE_ASPECT_RATIOS as readonly string[]).includes(value)
  ) {
    return value as GenSpaceImageAspectRatio;
  }
  return "16:9";
}
