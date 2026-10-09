import type { MicBlockCause, MicRemedy, MicRemedyAction } from "./types.ts";

const COPY: Record<
  MicBlockCause,
  { title: string; body: string; action: MicRemedyAction }
> = {
  "user-denied": {
    title: "Microphone access is needed to record.",
    body: "Allow the microphone, then try again.",
    action: "retry",
  },
  "os-denied": {
    title: "Microphone is turned off in system settings.",
    body: "Turn on microphone access for this app, then try again.",
    action: "open-os-settings",
  },
  "no-device": {
    title: "No microphone was found.",
    body: "Connect a microphone, then try again.",
    action: "retry",
  },
  "device-busy": {
    title: "The microphone is in use by another app.",
    body: "Close the other app, then try again.",
    action: "retry",
  },
  dismissed: {
    title: "Microphone access is needed to record.",
    body: "Allow the microphone, then try again.",
    action: "retry",
  },
  "capture-unsupported": {
    title: "This page cannot use the microphone.",
    body: "Cancel, then import an audio file.",
    action: "none",
  },
  "unreadable-recording": {
    title: "Could not read this recording.",
    body: "Try recording again, or import an audio file.",
    action: "retry",
  },
  unknown: {
    title: "Could not start the microphone.",
    body: "Try again. If this keeps happening, import an audio file instead.",
    action: "retry",
  },
};

export function micRemedyFor(
  cause: MicBlockCause,
  canOpenSettings: boolean,
): MicRemedy {
  const copy = COPY[cause];
  if (copy.action === "open-os-settings" && !canOpenSettings) {
    return { title: copy.title, body: copy.body, action: "retry" };
  }
  return copy;
}

export function micRemedyLabel(action: MicRemedyAction): string {
  switch (action) {
    case "retry":
      return "Try again";
    case "open-os-settings":
      return "Open Settings";
    case "none":
      return "";
  }
}
