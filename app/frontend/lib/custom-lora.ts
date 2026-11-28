import { paths } from "../paths.ts";

/**
 * Custom LoRA is a Home feature, but not a catalog recipe: the weights are the user's own, saved
 * on the desktop from a file or a link. Its id is the generation `feature` the backend stores
 * (backend `CUSTOM_LORA_FEATURE`).
 */
export const CUSTOM_LORA_HOME_FEATURE = {
  id: "custom-lora",
  path: paths.customLora,
  title: "Add custom LoRA",
  description: "Add your own LoRA from a file or a Hugging Face link.",
  // Not a fetcher tool. Quick search reads this field on every Home feature.
  fetcherTool: undefined,
} as const;

/** What a run of a saved LoRA is called when the run holds no name. The add page has another title. */
export const CUSTOM_LORA_RUN_FALLBACK_LABEL = "Custom LoRA";

/** The longest name the backend accepts (`CustomLoraImportRequest.name`). */
export const CUSTOM_LORA_NAME_MAX_LENGTH = 60;
// The same limits as backend/api_types.py. The backend refuses a longer value.
export const CUSTOM_LORA_TRIGGER_MAX_LENGTH = 80;
export const CUSTOM_LORA_DESCRIPTION_MAX_LENGTH = 600;
export const CUSTOM_LORA_EXAMPLE_MAX_LENGTH = 1000;
export const CUSTOM_LORA_EXAMPLES_MAX = 3;

export const CUSTOM_LORA_DEFAULT_STRENGTH = 1;

/** The id of a saved LoRA in the sidebar. It cannot clash with a feature id: none has a colon. */
export type CustomLoraNavId = `${typeof CUSTOM_LORA_HOME_FEATURE.id}:${string}`;

export function customLoraNavId(customLoraId: string): CustomLoraNavId {
  return `${CUSTOM_LORA_HOME_FEATURE.id}:${customLoraId}`;
}

export function isCustomLoraNavId(id: string): id is CustomLoraNavId {
  return id.startsWith(`${CUSTOM_LORA_HOME_FEATURE.id}:`);
}
