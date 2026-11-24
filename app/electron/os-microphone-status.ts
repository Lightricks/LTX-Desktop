export type OsMicrophoneStatus =
  | "granted"
  | "denied"
  | "not-determined"
  | "unknown";

export function mapOsMicrophoneStatus(status: string): OsMicrophoneStatus {
  if (
    status === "granted" ||
    status === "denied" ||
    status === "not-determined"
  ) {
    return status;
  }
  return "unknown";
}
