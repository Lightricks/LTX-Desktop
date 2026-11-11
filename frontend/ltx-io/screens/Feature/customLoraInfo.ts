import { CUSTOM_LORA_EXAMPLES_MAX } from "../../../lib/custom-lora.ts";

/** What the user tells the app about a LoRA, so the prompt enhancer can write for it. */
export type CustomLoraInfo = {
  trigger: string;
  description: string;
  /** One entry for each example field. The list always has at least one entry. */
  examples: string[];
};

/** Link validity and what a Hugging Face model card suggests. Every part can be empty. */
export type CustomLoraCardSuggestions = {
  accepted?: boolean;
  trigger?: string | null;
  description?: string;
  examples?: string[];
};

export const EMPTY_CUSTOM_LORA_INFO: CustomLoraInfo = {
  trigger: "",
  description: "",
  examples: [""],
};

export type ImportErrorField = "name" | "file" | "link" | "cover" | "form";

// A named field code stays on that field. Any other code from the import request stays at the
// bottom. A background failure has no code, so it stays on the file or the link that was sent.
export function importErrorField(
  code: string | undefined,
  source: "file" | "link" | null,
): ImportErrorField {
  if (code === "CUSTOM_LORA_NAME_INVALID" || code === "CUSTOM_LORA_NAME_TAKEN")
    return "name";
  if (code === "CUSTOM_LORA_FILE_INVALID") return "file";
  if (code === "CUSTOM_LORA_URL_INVALID") return "link";
  if (code === "CUSTOM_LORA_COVER_INVALID") return "cover";
  if (code != null) return "form";
  return source ?? "form";
}

/**
 * Put card suggestions into the form. A field the user already filled keeps its text: the card
 * never replaces what the user wrote. The examples fill only when every example field is empty.
 */
export function applyCardSuggestions(
  current: CustomLoraInfo,
  suggestions: CustomLoraCardSuggestions,
): CustomLoraInfo {
  const suggestedExamples = (suggestions.examples ?? []).slice(
    0,
    CUSTOM_LORA_EXAMPLES_MAX,
  );
  const hasExamples = current.examples.some((example) => example.trim().length > 0);
  return {
    trigger:
      current.trigger.trim() === "" ? (suggestions.trigger ?? "") : current.trigger,
    description:
      current.description.trim() === ""
        ? (suggestions.description ?? "")
        : current.description,
    examples:
      hasExamples || suggestedExamples.length === 0
        ? current.examples
        : suggestedExamples,
  };
}

/** The values to send. Empty examples are dropped. An empty trigger or description is left out. */
export function customLoraInfoRequest(info: CustomLoraInfo): {
  description?: string;
  trigger?: string;
  examples: string[];
} {
  const trigger = info.trigger.trim();
  const description = info.description.trim();
  return {
    ...(description === "" ? {} : { description }),
    ...(trigger === "" ? {} : { trigger }),
    examples: info.examples
      .map((example) => example.trim())
      .filter((example) => example !== ""),
  };
}
