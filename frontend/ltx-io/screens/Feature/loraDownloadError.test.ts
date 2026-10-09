import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  LORA_GATED_DOWNLOAD_COPY,
  LORA_GENERIC_DOWNLOAD_COPY,
  isHuggingFaceGatedError,
  loraDownloadErrorCopy,
} from "./loraDownloadError.ts";

const GATED_403 = `403 Client Error. (Request ID: Root=1-abc)
Cannot access gated repo for url https://huggingface.co/Lightricks/LTX-2.3-22b-LoRA-Cinematograph/resolve/main/file.safetensors.
Access to model Lightricks/LTX-2.3-22b-LoRA-Cinematograph is restricted and you are not in the authorized list.`;

describe("loraDownloadErrorCopy", () => {
  it("detects Hugging Face gated 403s", () => {
    assert.equal(isHuggingFaceGatedError(GATED_403), true);
    assert.equal(loraDownloadErrorCopy(GATED_403), LORA_GATED_DOWNLOAD_COPY);
  });

  it("hides other long backend dumps", () => {
    const dump = `500 Server Error. (Request ID: Root=1-xyz) ${"x".repeat(200)}`;
    assert.equal(isHuggingFaceGatedError(dump), false);
    assert.equal(loraDownloadErrorCopy(dump), LORA_GENERIC_DOWNLOAD_COPY);
  });

  it("keeps short local messages", () => {
    assert.equal(loraDownloadErrorCopy("Lost contact with the download."), "Lost contact with the download.");
  });

  it("does not treat substring gated matches as Hugging Face gated errors", () => {
    assert.equal(isHuggingFaceGatedError("download navigated away"), false);
  });
});
