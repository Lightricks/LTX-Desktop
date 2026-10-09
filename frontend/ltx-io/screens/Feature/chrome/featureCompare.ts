import { getIcLoraRecipe, isIcLoraRecipeId } from "@/lib/ic-lora-recipes";
import type { HomeFeatureId } from "@/lib/home-features";

import {
  PACKAGED_EXPLORE_EXAMPLES,
  PACKAGED_EXPLORE_SEEDS,
} from "../../../assets/packaged-explore-assets";
import { IC_LORA_RECIPE_EXAMPLE_MEDIA } from "../icLoraRecipeExampleMedia";
import { LORA_RECIPE_EXAMPLE_MEDIA } from "../loraRecipeExampleMedia";

/** The input of an example: a start frame (poster only) or a video. */
export type CompareSource = {
  readonly videoUrl?: string;
  readonly posterUrl?: string;
};

export type CompareResult = {
  readonly videoUrl: string;
  readonly posterUrl: string;
};

/**
 * How a feature shows its example against its input.
 * - curtain: a draggable slider over two clips of the same scene (IC-LoRA).
 * - toggle: a Source / Result switch. The source is a start frame or a video
 *   that the result changes (Retake, Extend).
 */
export type FeatureCompare =
  | {
      readonly mode: "curtain";
      readonly sourceVideoUrl: string;
      readonly result: CompareResult;
    }
  | {
      readonly mode: "toggle";
      readonly source: CompareSource;
      readonly result: CompareResult;
    };

function seedUrl(id: keyof typeof PACKAGED_EXPLORE_SEEDS): string {
  return PACKAGED_EXPLORE_SEEDS[id].url;
}

function exampleUrl(id: keyof typeof PACKAGED_EXPLORE_EXAMPLES): string {
  return PACKAGED_EXPLORE_EXAMPLES[id].url;
}

function startFrameToggle(
  startFrameUrl: string,
  result: CompareResult,
): FeatureCompare {
  return { mode: "toggle", source: { posterUrl: startFrameUrl }, result };
}

function packagedExampleResult(
  video: keyof typeof PACKAGED_EXPLORE_EXAMPLES,
  poster: keyof typeof PACKAGED_EXPLORE_EXAMPLES,
): CompareResult {
  return { videoUrl: exampleUrl(video), posterUrl: exampleUrl(poster) };
}

/**
 * The compare view for a feature, or null when the feature shows one clip.
 * The result is the empty-state example, so it matches the packaged input.
 */
export function resolveFeatureCompare(
  featureId: HomeFeatureId,
): FeatureCompare | null {
  if (isIcLoraRecipeId(featureId)) {
    return {
      mode: "curtain",
      sourceVideoUrl: seedUrl(getIcLoraRecipe(featureId).seed.kind),
      result: IC_LORA_RECIPE_EXAMPLE_MEDIA[featureId],
    };
  }

  switch (featureId) {
    case "retake": {
      const result = packagedExampleResult(
        "retake-example-video",
        "retake-example-poster",
      );
      return {
        mode: "toggle",
        source: { videoUrl: seedUrl("retake-video") },
        result,
      };
    }
    case "extend": {
      const result = packagedExampleResult(
        "extend-example-video",
        "extend-example-poster",
      );
      return {
        mode: "toggle",
        source: { videoUrl: seedUrl("extend-video") },
        result,
      };
    }
    case "image-to-video":
      return startFrameToggle(
        seedUrl("image-to-video-start-frame"),
        packagedExampleResult(
          "image-to-video-example-video",
          "image-to-video-example-poster",
        ),
      );
    case "audio-to-video":
      return startFrameToggle(
        seedUrl("audio-to-video-start-frame"),
        packagedExampleResult(
          "audio-to-video-example-video",
          "audio-to-video-example-poster",
        ),
      );
    case "cinemagraph":
      return startFrameToggle(
        seedUrl("cinemagraph-start-frame"),
        LORA_RECIPE_EXAMPLE_MEDIA.cinemagraph,
      );
    case "dolly-in":
      return startFrameToggle(
        seedUrl("dolly-in-start-frame"),
        LORA_RECIPE_EXAMPLE_MEDIA["dolly-in"],
      );
    case "dolly-out":
      return startFrameToggle(
        seedUrl("dolly-out-start-frame"),
        LORA_RECIPE_EXAMPLE_MEDIA["dolly-out"],
      );
    case "jib-up":
      return startFrameToggle(
        seedUrl("jib-up-start-frame"),
        LORA_RECIPE_EXAMPLE_MEDIA["jib-up"],
      );
    case "jib-down":
      return startFrameToggle(
        seedUrl("jib-down-start-frame"),
        LORA_RECIPE_EXAMPLE_MEDIA["jib-down"],
      );
    case "vbvr":
      return startFrameToggle(
        seedUrl("vbvr-start-frame"),
        LORA_RECIPE_EXAMPLE_MEDIA.vbvr,
      );
    // The result morphs the start frame into the end frame. Only the start shows.
    case "transition":
      return startFrameToggle(
        seedUrl("transition-start-frame"),
        LORA_RECIPE_EXAMPLE_MEDIA.transition,
      );
    default:
      return null;
  }
}
