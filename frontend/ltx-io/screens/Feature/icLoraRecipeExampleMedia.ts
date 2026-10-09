import type { IcLoraRecipeId } from "@/lib/ic-lora-recipes";

import alphaGenExamplePosterUrl from "../../assets/alpha-gen/example-poster.webp";
import alphaGenExampleVideoUrl from "../../assets/alpha-gen/example.mp4";
import deblurExamplePosterUrl from "../../assets/deblur/example-poster.webp";
import deblurExampleVideoUrl from "../../assets/deblur/example.mp4";
import colorizationExamplePosterUrl from "../../assets/colorization/example-poster.webp";
import colorizationExampleVideoUrl from "../../assets/colorization/example.mp4";
import cleanPlateExamplePosterUrl from "../../assets/clean-plate/example-poster.webp";
import cleanPlateExampleVideoUrl from "../../assets/clean-plate/example.mp4";
import decompressionExamplePosterUrl from "../../assets/decompression/example-poster.webp";
import decompressionExampleVideoUrl from "../../assets/decompression/example.mp4";
import waterSimulationExamplePosterUrl from "../../assets/water-simulation/example-poster.webp";
import waterSimulationExampleVideoUrl from "../../assets/water-simulation/example.mp4";
import layoutToRenderExamplePosterUrl from "../../assets/layout-to-render/example-poster.webp";
import layoutToRenderExampleVideoUrl from "../../assets/layout-to-render/example.mp4";
import restoreExamplePosterUrl from "../../assets/restore/example-poster.webp";
import restoreExampleVideoUrl from "../../assets/restore/example.mp4";
import dayToNightExamplePosterUrl from "../../assets/day-to-night/example-poster.webp";
import dayToNightExampleVideoUrl from "../../assets/day-to-night/example.mp4";

export type IcLoraRecipeExampleMedia = {
  videoUrl: string;
  posterUrl: string;
};

export const IC_LORA_RECIPE_EXAMPLE_MEDIA: Record<
  IcLoraRecipeId,
  IcLoraRecipeExampleMedia
> = {
  "day-to-night": {
    videoUrl: dayToNightExampleVideoUrl,
    posterUrl: dayToNightExamplePosterUrl,
  },
  "alpha-gen": {
    videoUrl: alphaGenExampleVideoUrl,
    posterUrl: alphaGenExamplePosterUrl,
  },
  "deblur": {
    videoUrl: deblurExampleVideoUrl,
    posterUrl: deblurExamplePosterUrl,
  },
  "colorization": {
    videoUrl: colorizationExampleVideoUrl,
    posterUrl: colorizationExamplePosterUrl,
  },
  "clean-plate": {
    videoUrl: cleanPlateExampleVideoUrl,
    posterUrl: cleanPlateExamplePosterUrl,
  },
  "decompression": {
    videoUrl: decompressionExampleVideoUrl,
    posterUrl: decompressionExamplePosterUrl,
  },
  "water-simulation": {
    videoUrl: waterSimulationExampleVideoUrl,
    posterUrl: waterSimulationExamplePosterUrl,
  },
    "layout-to-render": {
    videoUrl: layoutToRenderExampleVideoUrl,
    posterUrl: layoutToRenderExamplePosterUrl,
  },
  "restore": {
    videoUrl: restoreExampleVideoUrl,
    posterUrl: restoreExamplePosterUrl,
  },
};
