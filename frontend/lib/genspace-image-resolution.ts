/** The image resolution tiers. Each tier is the short side of the image, in pixels. */
export const IMAGE_SHORT_SIDE_BY_RESOLUTION = {
  "1080p": 1080,
  "1440p": 1440,
  "2048p": 2048,
} as const;

/** The backend rounds each side of an image down to a multiple of this (`floor_to_image_grid`). */
export const IMAGE_SIZE_GRID = 16;

export type ImageResolution = keyof typeof IMAGE_SHORT_SIDE_BY_RESOLUTION;

export const IMAGE_RESOLUTIONS = Object.keys(
  IMAGE_SHORT_SIDE_BY_RESOLUTION,
) as ImageResolution[];

export const DEFAULT_IMAGE_RESOLUTION: ImageResolution = "1080p";
