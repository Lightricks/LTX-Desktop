export type AssistantSelectionInput = {
  provider: "gemini" | "local";
  model: string | null | undefined;
  hasGeminiApiKey: boolean;
  models: readonly { model: string; installed: boolean }[] | undefined;
  settingsName?: string;
};

export function assistantSelectionIssue(
  input: AssistantSelectionInput,
): string | null {
  const settings = input.settingsName ?? "Settings";
  if (input.provider === "gemini") {
    return input.hasGeminiApiKey
      ? null
      : `Add a Gemini API key in ${settings} to send messages.`;
  }
  if (input.models == null) {
    return "Select a valid assistant model to send messages.";
  }
  if (
    input.model &&
    input.models.some((item) => item.model === input.model && item.installed)
  ) {
    return null;
  }
  return `Download this model in ${settings}, or select a valid model.`;
}

export function isAssistantSelectionReady(input: AssistantSelectionInput): boolean {
  return assistantSelectionIssue(input) === null;
}
