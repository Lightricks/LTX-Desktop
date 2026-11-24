import type { CaptureFailure, MicBlockCause } from "./types.ts";

export function classifyCaptureFailure(failure: CaptureFailure): MicBlockCause {
  if (failure.osMicrophone === "denied") {
    return "os-denied";
  }
  const name = failure.errorName;
  if (name === "NotFoundError") {
    return "no-device";
  }
  if (name === "NotReadableError") {
    return "device-busy";
  }
  if (name === "AbortError") {
    return "dismissed";
  }
  if (name === "SecurityError") {
    return "capture-unsupported";
  }
  if (name === "NotAllowedError") {
    return "user-denied";
  }
  if (name === "TypeError" || name === "NotSupportedError") {
    return "capture-unsupported";
  }
  return "unknown";
}

export function captureFailureFromUnknown(
  error: unknown,
  osMicrophone: CaptureFailure["osMicrophone"],
): CaptureFailure {
  if (error instanceof DOMException) {
    return {
      errorName: error.name,
      errorMessage: error.message.toLowerCase(),
      osMicrophone,
    };
  }
  if (error instanceof Error) {
    return {
      errorName: error.name,
      errorMessage: error.message.toLowerCase(),
      osMicrophone,
    };
  }
  return { errorName: null, errorMessage: "", osMicrophone };
}
