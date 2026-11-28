export type OsMicrophoneStatus =
  | "granted"
  | "denied"
  | "not-determined"
  | "unknown";

export type MicBlockCause =
  | "user-denied"
  | "os-denied"
  | "no-device"
  | "device-busy"
  | "dismissed"
  | "capture-unsupported"
  | "unreadable-recording"
  | "unknown";

export type MicRemedyAction = "retry" | "open-os-settings" | "none";

export type MicRemedy = {
  title: string;
  body: string;
  action: MicRemedyAction;
};

export type MicBlocker = {
  cause: MicBlockCause;
  remedy: MicRemedy;
};

export type MicAccess =
  | { status: "granted"; stream: MediaStream }
  | { status: "blocked"; blocker: MicBlocker };

export type LiveMicrophoneAccess = {
  kind: "live";
  requestStream: () => Promise<MicAccess>;
  openMicrophoneSettings: (() => Promise<void>) | null;
};

export type MicrophoneAccess = LiveMicrophoneAccess;

export type CaptureFailure = {
  errorName: string | null;
  errorMessage: string;
  osMicrophone: OsMicrophoneStatus;
};
