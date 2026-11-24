import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { resolveOpenMicrophoneSettings } from "./open-microphone-settings.ts";

describe("resolveOpenMicrophoneSettings", () => {
  it("opens Windows and macOS privacy URLs", async () => {
    const opened: string[] = [];
    const openExternal = async (url: string) => {
      opened.push(url);
    };
    assert.deepEqual(
      await resolveOpenMicrophoneSettings("win32", openExternal),
      { success: true },
    );
    assert.deepEqual(
      await resolveOpenMicrophoneSettings("darwin", openExternal),
      { success: true },
    );
    assert.deepEqual(opened, [
      "ms-settings:privacy-microphone",
      "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone",
    ]);
  });

  it("returns failure on unsupported platforms and openExternal errors", async () => {
    assert.deepEqual(
      await resolveOpenMicrophoneSettings("linux", async () => {}),
      {
        success: false,
        error: "Microphone settings are not available on this platform",
      },
    );
    assert.deepEqual(
      await resolveOpenMicrophoneSettings("win32", async () => {
        throw new Error("blocked");
      }),
      { success: false, error: "blocked" },
    );
  });
});
