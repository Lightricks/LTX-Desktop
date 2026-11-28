import { unsupportedModelReason } from "../../../lib/catalogVariantChoice.ts";
import { missingIcLoraImage, type IcLoraRecipe } from "../../../../lib/ic-lora-recipes.ts";
import type { ValidationIssue } from "../types.ts";
import { VIDEO_DURATION_CAP_TOLERANCE_SECONDS } from "../trim/trimConstants.ts";
import {
  formatIcLoraSeconds,
  icLoraFpsOptions,
  icLoraMaxInputSeconds,
  icLoraOutputFps,
} from "./icLoraFps.ts";
import type { IcLoraRecipeContext, IcLoraRecipeValues } from "./icLoraRecipe.ts";
import { videoRequiredIssue } from "./videoFeature.ts";

/** The issues that block Generate for a recipe form. `promptOptional` allows a blank prompt. */
export function validateIcLoraRecipe(
  recipe: IcLoraRecipe,
  values: IcLoraRecipeValues,
  context: IcLoraRecipeContext,
  promptOptional: boolean,
): ValidationIssue[] {
  const issues: ValidationIssue[] = [];
  if (values.video == null) issues.push(videoRequiredIssue());
  const missingImage = missingIcLoraImage(
    recipe,
    values.image,
    context.referenceImageRequired,
  );
  if (missingImage != null) {
    issues.push({
      id: "image-required",
      fieldId: "image",
      message: `${missingImage.label} is required.`,
    });
  }
  const unsupported = unsupportedModelReason(
    context.supportedModels,
    values.model,
    "IC-LoRA",
  );
  if (unsupported != null) {
    issues.push({
      id: "model-unsupported",
      fieldId: "model",
      alwaysRevealed: true,
      message: unsupported,
    });
  }
  const rate = icLoraOutputFps(values.fps, context.videoFps);
  const limit = icLoraMaxInputSeconds(rate);
  const duration = context.videoDurationSeconds;
  if (
    duration != null &&
    duration > limit + VIDEO_DURATION_CAP_TOLERANCE_SECONDS
  ) {
    const choices = context.videoFps == null ? [] : icLoraFpsOptions(context.videoFps);
    const lowest = choices[0]?.value;
    const canLower = choices.length > 1 && values.fps !== lowest;
    issues.push({
      id: "video-too-long",
      fieldId: "video",
      alwaysRevealed: true,
      message: canLower
        ? `This clip is longer than ${formatIcLoraSeconds(limit)}s at ${rate}fps. Trim it, or choose a lower frame rate.`
        : `This clip is longer than ${formatIcLoraSeconds(limit)}s. Trim it to continue.`,
    });
  }
  if (!promptOptional && values.prompt.trim().length === 0) {
    issues.push({
      id: "prompt-required",
      fieldId: "prompt",
      message: "Prompt is required.",
    });
  }
  return issues;
}
