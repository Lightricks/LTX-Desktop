import { captureFailureFromUnknown, classifyCaptureFailure } from "./classify.ts";
import { micRemedyFor } from "./remedy.ts";
import type {
  LiveMicrophoneAccess,
  MicAccess,
  MicBlockCause,
  OsMicrophoneStatus,
} from "./types.ts";

export function createLiveMicrophoneAccess(host: {
  diagnoseOsMicrophone: () => Promise<OsMicrophoneStatus>;
  openMicrophoneSettings: (() => Promise<void>) | null;
}): LiveMicrophoneAccess {
  const canOpenSettings = host.openMicrophoneSettings != null;

  const blocked = (cause: MicBlockCause): MicAccess => ({
    status: "blocked",
    blocker: {
      cause,
      remedy: micRemedyFor(cause, canOpenSettings),
    },
  });

  const requestStream = async (): Promise<MicAccess> => {
    if (globalThis.isSecureContext === false) {
      return blocked("capture-unsupported");
    }

    const mediaDevices = globalThis.navigator?.mediaDevices;
    if (typeof mediaDevices?.getUserMedia !== "function") {
      return blocked("capture-unsupported");
    }

    try {
      const stream = await mediaDevices.getUserMedia({
        audio: true,
      });
      if (stream.getAudioTracks().length === 0) {
        stream.getTracks().forEach((track) => track.stop());
        return blocked("no-device");
      }
      return { status: "granted", stream };
    } catch (error: unknown) {
      const failure = captureFailureFromUnknown(error, "unknown");
      if (failure.errorName === "NotAllowedError") {
        let osMicrophone: OsMicrophoneStatus = "unknown";
        try {
          osMicrophone = await host.diagnoseOsMicrophone();
        } catch {
          osMicrophone = "unknown";
        }
        return blocked(
          classifyCaptureFailure(captureFailureFromUnknown(error, osMicrophone)),
        );
      }
      return blocked(classifyCaptureFailure(failure));
    }
  };

  return {
    kind: "live",
    requestStream,
    openMicrophoneSettings: host.openMicrophoneSettings,
  };
}
