import type { LoraRecipeId } from "@/lib/lora-recipes";

import claymationExamplePosterUrl from "../../assets/claymation/example-poster.webp";
import claymationExampleVideoUrl from "../../assets/claymation/example.webm";
import cinemagraphExamplePosterUrl from "../../assets/cinemagraph/example-poster.webp";
import cinemagraphExampleVideoUrl from "../../assets/cinemagraph/example.mp4";
import cozyFeltExamplePosterUrl from "../../assets/cozy-felt/example-poster.webp";
import cozyFeltExampleVideoUrl from "../../assets/cozy-felt/example.webm";
import dollyInExamplePosterUrl from "../../assets/dolly-in/example-poster.webp";
import dollyInExampleVideoUrl from "../../assets/dolly-in/example.webm";
import dollyOutExamplePosterUrl from "../../assets/dolly-out/example-poster.webp";
import dollyOutExampleVideoUrl from "../../assets/dolly-out/example.webm";
import fantasyPainterlyExamplePosterUrl from "../../assets/fantasy-painterly/example-poster.webp";
import fantasyPainterlyExampleVideoUrl from "../../assets/fantasy-painterly/example.webm";
import fpvMotionExamplePosterUrl from "../../assets/fpv-motion/example-poster.webp";
import fpvMotionExampleVideoUrl from "../../assets/fpv-motion/example.webm";
import jibDownExamplePosterUrl from "../../assets/jib-down/example-poster.webp";
import jibDownExampleVideoUrl from "../../assets/jib-down/example.webm";
import jibUpExamplePosterUrl from "../../assets/jib-up/example-poster.webp";
import jibUpExampleVideoUrl from "../../assets/jib-up/example.webm";
import openwheelTCamExamplePosterUrl from "../../assets/openwheel-t-cam/example-poster.webp";
import openwheelTCamExampleVideoUrl from "../../assets/openwheel-t-cam/example.webm";
import paperCutOutExamplePosterUrl from "../../assets/paper-cut-out-style/example-poster.webp";
import paperCutOutExampleVideoUrl from "../../assets/paper-cut-out-style/example.webm";
import transitionExamplePosterUrl from "../../assets/transition/example-poster.webp";
import transitionExampleVideoUrl from "../../assets/transition/example.webm";
import vbvrExamplePosterUrl from "../../assets/vbvr/example-poster.webp";
import vbvrExampleVideoUrl from "../../assets/vbvr/example.mp4";

export type LoraRecipeExampleMedia = {
  videoUrl: string;
  posterUrl: string;
};

/**
 * The empty-results example clip per recipe.
 */
export const LORA_RECIPE_EXAMPLE_MEDIA: Record<
  LoraRecipeId,
  LoraRecipeExampleMedia
> = {
  "cozy-felt": {
    videoUrl: cozyFeltExampleVideoUrl,
    posterUrl: cozyFeltExamplePosterUrl,
  },
  claymation: {
    videoUrl: claymationExampleVideoUrl,
    posterUrl: claymationExamplePosterUrl,
  },
  "fantasy-painterly": {
    videoUrl: fantasyPainterlyExampleVideoUrl,
    posterUrl: fantasyPainterlyExamplePosterUrl,
  },
  "paper-cut-out-style": {
    videoUrl: paperCutOutExampleVideoUrl,
    posterUrl: paperCutOutExamplePosterUrl,
  },
  cinemagraph: {
    videoUrl: cinemagraphExampleVideoUrl,
    posterUrl: cinemagraphExamplePosterUrl,
  },
  "jib-up": {
    videoUrl: jibUpExampleVideoUrl,
    posterUrl: jibUpExamplePosterUrl,
  },
  "jib-down": {
    videoUrl: jibDownExampleVideoUrl,
    posterUrl: jibDownExamplePosterUrl,
  },
  "dolly-in": {
    videoUrl: dollyInExampleVideoUrl,
    posterUrl: dollyInExamplePosterUrl,
  },
  "dolly-out": {
    videoUrl: dollyOutExampleVideoUrl,
    posterUrl: dollyOutExamplePosterUrl,
  },
  "fpv-motion": {
    videoUrl: fpvMotionExampleVideoUrl,
    posterUrl: fpvMotionExamplePosterUrl,
  },
  "openwheel-t-cam": {
    videoUrl: openwheelTCamExampleVideoUrl,
    posterUrl: openwheelTCamExamplePosterUrl,
  },
  transition: {
    videoUrl: transitionExampleVideoUrl,
    posterUrl: transitionExamplePosterUrl,
  },
  vbvr: {
    videoUrl: vbvrExampleVideoUrl,
    posterUrl: vbvrExamplePosterUrl,
  },
};
