export const LORA_DOWNLOADED_TOAST_MS = 5000;

export const LORA_DOWNLOADED_TOAST_COPY =
  "LoRA downloaded. Manage your models and LoRAs in Settings.";

export const LORA_GATED_DOWNLOAD_COPY =
  "This LoRA is gated on Hugging Face. Request access, then retry.";

export const LORA_GENERIC_DOWNLOAD_COPY =
  "Couldn't download this LoRA. Check your connection and try again.";

export function isHuggingFaceGatedError(message: string): boolean {
  const lower = message.toLowerCase();
  return (
    /\bgated\b/.test(lower) ||
    lower.includes("authorized list") ||
    (lower.includes("403") && lower.includes("huggingface"))
  );
}

export function loraDownloadErrorCopy(message: string): string {
  if (isHuggingFaceGatedError(message)) return LORA_GATED_DOWNLOAD_COPY;
  if (
    message.length > 160 ||
    /status code|request id|traceback|huggingface\.co/i.test(message)
  ) {
    return LORA_GENERIC_DOWNLOAD_COPY;
  }
  return message;
}
