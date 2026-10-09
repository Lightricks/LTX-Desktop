import { systemPreferences } from "electron";

import { resolveOpenMicrophoneSettings } from "../open-microphone-settings";
import { mapOsMicrophoneStatus, type OsMicrophoneStatus } from "../os-microphone-status";
import { handle } from "./typed-handle";

export type { OsMicrophoneStatus };

export function readOsMicrophoneStatus(): OsMicrophoneStatus {
  try {
    return mapOsMicrophoneStatus(
      systemPreferences.getMediaAccessStatus("microphone"),
    );
  } catch {
    return "unknown";
  }
}

export function registerMediaHandlers(): void {
  handle("diagnoseMicrophone", () => ({
    success: true as const,
    osMicrophone: readOsMicrophoneStatus(),
  }));

  handle("openMicrophoneSettings", async () => {
    const { shell } = await import("electron");
    return resolveOpenMicrophoneSettings(process.platform, (url) =>
      shell.openExternal(url),
    );
  });
}
