import { readGenerationParams } from "../../../lib/resultsFeedModel.ts";
import type { FeatureFieldSchema, TextareaFieldDefForKey, TextareaFieldKey } from "../types.ts";

export const MAX_NAG_SCALE = 100;

export const NEGATIVE_PROMPT_PLACEHOLDER = "subtitles, watermark, blurry, flicker...";

export const NEGATIVE_PROMPT_HELP_TEXT =
  "List elements, styles, or artifacts to exclude from the generated video.";

export function negativePromptField<
  TSchema extends FeatureFieldSchema & { negativePrompt: { kind: "textarea" } },
>(options?: {
  disabledReason?: string | null;
}): TextareaFieldDefForKey<TSchema, "negativePrompt" & TextareaFieldKey<TSchema>> {
  return {
    kind: "textarea",
    id: "negativePrompt",
    label: "Negative prompt",
    dataKey: "negativePrompt" as "negativePrompt" & TextareaFieldKey<TSchema>,
    placeholder: NEGATIVE_PROMPT_PLACEHOLDER,
    helpText: NEGATIVE_PROMPT_HELP_TEXT,
    compact: true,
    ...(options?.disabledReason != null
      ? { disabledReason: options.disabledReason }
      : {}),
  };
}

export function readNegativePrompt(spec: unknown): string {
  const params = readGenerationParams(spec);
  return typeof params?.negativePrompt === "string" ? params.negativePrompt : "";
}

export function swapDemoNegative<V extends { negativePrompt: string }>(
  before: V,
  after: V,
  opts: {
    demoNegative?: string;
    catalogDefault?: string | null;
    changed: (before: V, after: V) => boolean;
  },
): V {
  if (
    !opts.catalogDefault ||
    before.negativePrompt !== opts.demoNegative ||
    !opts.changed(before, after)
  ) {
    return after;
  }
  return { ...after, negativePrompt: opts.catalogDefault };
}
