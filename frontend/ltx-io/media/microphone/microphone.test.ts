import assert from "node:assert/strict";
import { afterEach, describe, it } from "node:test";

import { classifyCaptureFailure } from "./classify.ts";
import { createLiveMicrophoneAccess } from "./liveAccess.ts";
import { micRemedyFor } from "./remedy.ts";

describe("classifyCaptureFailure", () => {
  it("ranks OS deny above browser error names", () => {
    assert.equal(
      classifyCaptureFailure({
        errorName: "NotAllowedError",
        errorMessage: "permission denied",
        osMicrophone: "denied",
      }),
      "os-denied",
    );
  });

  it("maps named getUserMedia failures", () => {
    assert.equal(
      classifyCaptureFailure({
        errorName: "NotFoundError",
        errorMessage: "",
        osMicrophone: "unknown",
      }),
      "no-device",
    );
    assert.equal(
      classifyCaptureFailure({
        errorName: "NotReadableError",
        errorMessage: "",
        osMicrophone: "granted",
      }),
      "device-busy",
    );
    assert.equal(
      classifyCaptureFailure({
        errorName: "AbortError",
        errorMessage: "",
        osMicrophone: "not-determined",
      }),
      "dismissed",
    );
    assert.equal(
      classifyCaptureFailure({
        errorName: "NotAllowedError",
        errorMessage: "permission denied",
        osMicrophone: "granted",
      }),
      "user-denied",
    );
    assert.equal(
      classifyCaptureFailure({
        errorName: "SecurityError",
        errorMessage: "the request is not allowed",
        osMicrophone: "granted",
      }),
      "capture-unsupported",
    );
    assert.equal(
      classifyCaptureFailure({
        errorName: "TypeError",
        errorMessage: "undefined",
        osMicrophone: "unknown",
      }),
      "capture-unsupported",
    );
    assert.equal(
      classifyCaptureFailure({
        errorName: null,
        errorMessage: "",
        osMicrophone: "unknown",
      }),
      "unknown",
    );
  });
});

describe("micRemedyFor", () => {
  it("falls back to retry when OS settings cannot be opened", () => {
    assert.equal(micRemedyFor("os-denied", false).action, "retry");
    assert.equal(micRemedyFor("os-denied", true).action, "open-os-settings");
  });

  it("tells the user to cancel and import when capture is unsupported", () => {
    const remedy = micRemedyFor("capture-unsupported", false);
    assert.equal(remedy.action, "none");
    assert.match(remedy.body, /import an audio file/i);
  });
});

describe("createLiveMicrophoneAccess", () => {
  const previousSecure = Object.getOwnPropertyDescriptor(
    globalThis,
    "isSecureContext",
  );
  const previousNavigator = Object.getOwnPropertyDescriptor(
    globalThis,
    "navigator",
  );

  afterEach(() => {
    if (previousSecure) {
      Object.defineProperty(globalThis, "isSecureContext", previousSecure);
    } else {
      delete (globalThis as { isSecureContext?: boolean }).isSecureContext;
    }
    if (previousNavigator) {
      Object.defineProperty(globalThis, "navigator", previousNavigator);
    } else {
      delete (globalThis as { navigator?: Navigator }).navigator;
    }
  });

  function installCapture(input: {
    isSecureContext: boolean;
    getUserMedia?: () => Promise<MediaStream>;
  }): { gumCalls: number } {
    const state = { gumCalls: 0 };
    Object.defineProperty(globalThis, "isSecureContext", {
      configurable: true,
      value: input.isSecureContext,
    });
    Object.defineProperty(globalThis, "navigator", {
      configurable: true,
      value: {
        mediaDevices: {
          getUserMedia: async () => {
            state.gumCalls += 1;
            if (input.getUserMedia == null) {
              throw new Error("getUserMedia not installed");
            }
            return input.getUserMedia();
          },
        },
      },
    });
    return state;
  }

  it("fails closed without calling getUserMedia when the page is not secure", async () => {
    const capture = installCapture({
      isSecureContext: false,
      getUserMedia: async () => {
        throw new Error("should not run");
      },
    });
    const access = createLiveMicrophoneAccess({
      diagnoseOsMicrophone: async () => "granted",
      openMicrophoneSettings: null,
    });
    const result = await access.requestStream();
    assert.equal(capture.gumCalls, 0);
    assert.equal(result.status, "blocked");
    if (result.status === "blocked") {
      assert.equal(result.blocker.cause, "capture-unsupported");
    }
  });

  it("still calls getUserMedia when OS status is denied", async () => {
    const capture = installCapture({
      isSecureContext: true,
      getUserMedia: async () => {
        throw Object.assign(new Error("denied"), { name: "NotAllowedError" });
      },
    });
    let diagnosed = 0;
    const access = createLiveMicrophoneAccess({
      diagnoseOsMicrophone: async () => {
        diagnosed += 1;
        return "denied";
      },
      openMicrophoneSettings: null,
    });
    const result = await access.requestStream();
    assert.equal(capture.gumCalls, 1);
    assert.equal(diagnosed, 1);
    assert.equal(result.status, "blocked");
    if (result.status === "blocked") {
      assert.equal(result.blocker.cause, "os-denied");
    }
  });

  it("does not diagnose OS status when getUserMedia succeeds", async () => {
    const stream = {
      getAudioTracks: () => [{ stop() {} }],
      getTracks: () => [{ stop() {} }],
    } as unknown as MediaStream;
    installCapture({
      isSecureContext: true,
      getUserMedia: async () => stream,
    });
    let diagnosed = 0;
    const access = createLiveMicrophoneAccess({
      diagnoseOsMicrophone: async () => {
        diagnosed += 1;
        return "denied";
      },
      openMicrophoneSettings: null,
    });
    const result = await access.requestStream();
    assert.equal(diagnosed, 0);
    assert.equal(result.status, "granted");
  });

  it("maps SecurityError to capture-unsupported without OS diagnosis", async () => {
    installCapture({
      isSecureContext: true,
      getUserMedia: async () => {
        throw Object.assign(new Error("insecure"), { name: "SecurityError" });
      },
    });
    let diagnosed = 0;
    const access = createLiveMicrophoneAccess({
      diagnoseOsMicrophone: async () => {
        diagnosed += 1;
        return "granted";
      },
      openMicrophoneSettings: null,
    });
    const result = await access.requestStream();
    assert.equal(diagnosed, 0);
    assert.equal(result.status, "blocked");
    if (result.status === "blocked") {
      assert.equal(result.blocker.cause, "capture-unsupported");
    }
  });

  it("falls back to unknown when OS diagnosis throws after NotAllowedError", async () => {
    const capture = installCapture({
      isSecureContext: true,
      getUserMedia: async () => {
        throw Object.assign(new Error("denied"), { name: "NotAllowedError" });
      },
    });
    const access = createLiveMicrophoneAccess({
      diagnoseOsMicrophone: async () => {
        throw new Error("ipc missing");
      },
      openMicrophoneSettings: null,
    });
    const result = await access.requestStream();
    assert.equal(capture.gumCalls, 1);
    assert.equal(result.status, "blocked");
    if (result.status === "blocked") {
      assert.equal(result.blocker.cause, "user-denied");
    }
  });
});
