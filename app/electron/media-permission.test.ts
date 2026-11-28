import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  isTrustedAppOrigin,
  isTrustedWebContents,
  mediaRequestOrigin,
  readIsMainFrame,
  readMediaKinds,
  runPermissionCheckHandler,
  runPermissionRequestHandler,
  shouldAllowPermissionCheck,
  shouldGrantPermissionRequest,
  shouldTrustMediaPermissionSource,
} from "./media-permission.ts";

const DEV_ORIGIN = "http://localhost:5173";
const FILE_URL = "file:///tmp/app/dist/index.html";

describe("readMediaKinds", () => {
  it("reads a request-handler mediaTypes array", () => {
    assert.deepEqual(readMediaKinds("media", { mediaTypes: ["audio"] }), {
      audio: true,
      video: false,
      extra: false,
    });
    assert.deepEqual(readMediaKinds("media", { mediaTypes: ["video"] }), {
      audio: false,
      video: true,
      extra: false,
    });
    assert.deepEqual(
      readMediaKinds("media", { mediaTypes: ["audio", "video"] }),
      { audio: true, video: true, extra: false },
    );
    assert.deepEqual(
      readMediaKinds("media", { mediaTypes: ["audio", "unknown"] }),
      { audio: true, video: false, extra: true },
    );
  });

  it("reads a check-handler singular mediaType", () => {
    assert.deepEqual(readMediaKinds("media", { mediaType: "audio" }), {
      audio: true,
      video: false,
      extra: false,
    });
    assert.deepEqual(readMediaKinds("media", { mediaType: "video" }), {
      audio: false,
      video: true,
      extra: false,
    });
  });

  it("treats missing or empty media details as undeclared", () => {
    assert.equal(readMediaKinds("media"), "undeclared");
    assert.equal(readMediaKinds("media", {}), "undeclared");
    assert.equal(readMediaKinds("media", { mediaTypes: [] }), "undeclared");
    assert.equal(readMediaKinds("media", { mediaType: "unknown" }), "undeclared");
    assert.equal(readMediaKinds("geolocation", { mediaTypes: ["audio"] }), null);
  });
});

describe("shouldGrantPermissionRequest", () => {
  it("grants a microphone-only media request", () => {
    assert.equal(
      shouldGrantPermissionRequest("media", { mediaTypes: ["audio"] }),
      true,
    );
  });

  it("denies camera-only media requests", () => {
    assert.equal(
      shouldGrantPermissionRequest("media", { mediaTypes: ["video"] }),
      false,
    );
  });

  it("denies combined audio+video media requests in any order", () => {
    assert.equal(
      shouldGrantPermissionRequest("media", { mediaTypes: ["audio", "video"] }),
      false,
    );
    assert.equal(
      shouldGrantPermissionRequest("media", { mediaTypes: ["video", "audio"] }),
      false,
    );
  });

  it("denies media requests with extra or unknown kinds", () => {
    assert.equal(
      shouldGrantPermissionRequest("media", {
        mediaTypes: ["audio", "unknown"],
      }),
      false,
    );
    assert.equal(
      shouldGrantPermissionRequest("media", { mediaTypes: ["microphone"] }),
      false,
    );
  });

  it("denies media requests with unknown, empty, or missing media details", () => {
    assert.equal(shouldGrantPermissionRequest("media", { mediaTypes: [] }), false);
    assert.equal(shouldGrantPermissionRequest("media", {}), false);
    assert.equal(shouldGrantPermissionRequest("media"), false);
  });

  it("leaves non-media permissions allowed, including clipboard and fullscreen", () => {
    for (const permission of [
      "clipboard-sanitized-write",
      "fullscreen",
      "geolocation",
      "notifications",
      "display-capture",
      "pointerLock",
      "unknown",
    ]) {
      assert.equal(shouldGrantPermissionRequest(permission), true);
      assert.equal(
        shouldGrantPermissionRequest(permission, { mediaTypes: ["audio"] }),
        true,
      );
    }
  });
});

describe("shouldAllowPermissionCheck", () => {
  it("allows an audio-only check and denies camera, mixed, and undeclared", () => {
    assert.equal(
      shouldAllowPermissionCheck("media", { mediaType: "audio" }),
      true,
    );
    assert.equal(
      shouldAllowPermissionCheck("media", { mediaType: "video" }),
      false,
    );
    assert.equal(shouldAllowPermissionCheck("media", {}), false);
    assert.equal(shouldAllowPermissionCheck("media"), false);
    assert.equal(
      shouldAllowPermissionCheck("clipboard-sanitized-write"),
      true,
    );
    assert.equal(shouldAllowPermissionCheck("fullscreen"), true);
    assert.equal(
      shouldAllowPermissionCheck("notifications", { mediaType: "audio" }),
      true,
    );
  });
});

describe("trusted renderer origin", () => {
  it("accepts only the main webContents id", () => {
    assert.equal(isTrustedWebContents({ id: 1 }, { id: 1 }), true);
    assert.equal(isTrustedWebContents({ id: 2 }, { id: 1 }), false);
    assert.equal(isTrustedWebContents(null, { id: 1 }), false);
    assert.equal(isTrustedWebContents({ id: 1 }, null), false);
  });

  it("accepts the Vite origin in development and the loaded file URL in production", () => {
    assert.equal(isTrustedAppOrigin(DEV_ORIGIN, DEV_ORIGIN), true);
    assert.equal(isTrustedAppOrigin(`${DEV_ORIGIN}/`, DEV_ORIGIN), true);
    assert.equal(isTrustedAppOrigin("https://evil.example", DEV_ORIGIN), false);
    assert.equal(isTrustedAppOrigin("file://", FILE_URL), false);
    assert.equal(isTrustedAppOrigin(FILE_URL, FILE_URL), true);
    assert.equal(
      isTrustedAppOrigin("file:///tmp/app/dist/other.html", FILE_URL),
      false,
    );
    assert.equal(isTrustedAppOrigin(DEV_ORIGIN, FILE_URL), false);
    assert.equal(isTrustedAppOrigin("", DEV_ORIGIN), false);
    assert.equal(isTrustedAppOrigin("null", FILE_URL), false);
    assert.equal(
      isTrustedAppOrigin(`${DEV_ORIGIN}/audio-to-video#record`, DEV_ORIGIN),
      true,
    );
    assert.equal(
      isTrustedAppOrigin("javascript:alert(1)", DEV_ORIGIN),
      false,
    );
    assert.equal(isTrustedAppOrigin("about:blank", FILE_URL), false);
  });

  it("reads securityOrigin or requestingUrl from permission request details", () => {
    assert.equal(
      mediaRequestOrigin({ securityOrigin: DEV_ORIGIN }),
      DEV_ORIGIN,
    );
    assert.equal(
      mediaRequestOrigin({ requestingUrl: `${DEV_ORIGIN}/index.html` }),
      DEV_ORIGIN,
    );
    assert.equal(mediaRequestOrigin({ requestingUrl: FILE_URL }), FILE_URL);
    assert.equal(mediaRequestOrigin({ securityOrigin: "file://" }), "");
    assert.equal(mediaRequestOrigin({ securityOrigin: "null" }), "");
    assert.equal(mediaRequestOrigin({ mediaTypes: ["audio"] }), "");
    assert.equal(mediaRequestOrigin(undefined), "");
  });

  it("reads isMainFrame when Chromium provides it", () => {
    assert.equal(readIsMainFrame({ isMainFrame: true }), true);
    assert.equal(readIsMainFrame({ isMainFrame: false }), false);
    assert.equal(readIsMainFrame({ mediaTypes: ["audio"] }), undefined);
  });

  it("trusts the main renderer only when omitted origin matches the loaded URL", () => {
    assert.equal(
      shouldTrustMediaPermissionSource({
        contents: { id: 1, getURL: () => `${DEV_ORIGIN}/` },
        mainContents: { id: 1 },
        requestingOrigin: "",
        trustedOrigin: DEV_ORIGIN,
      }),
      true,
    );
    assert.equal(
      shouldTrustMediaPermissionSource({
        contents: { id: 1, getURL: () => "https://evil.example/" },
        mainContents: { id: 1 },
        requestingOrigin: "",
        trustedOrigin: DEV_ORIGIN,
      }),
      false,
    );
    assert.equal(
      shouldTrustMediaPermissionSource({
        contents: { id: 1 },
        mainContents: { id: 1 },
        requestingOrigin: "",
        trustedOrigin: DEV_ORIGIN,
      }),
      false,
    );
    assert.equal(
      shouldTrustMediaPermissionSource({
        contents: { id: 2, getURL: () => DEV_ORIGIN },
        mainContents: { id: 1 },
        requestingOrigin: DEV_ORIGIN,
        trustedOrigin: DEV_ORIGIN,
      }),
      false,
    );
  });

  it("treats an opaque null origin as omitted and uses the loaded URL", () => {
    assert.equal(
      shouldTrustMediaPermissionSource({
        contents: { id: 1, getURL: () => DEV_ORIGIN },
        mainContents: { id: 1 },
        requestingOrigin: "null",
        trustedOrigin: DEV_ORIGIN,
        isMainFrame: true,
      }),
      true,
    );
  });

  it("denies subframe requests and opaque origins without a trusted contents URL", () => {
    assert.equal(
      shouldTrustMediaPermissionSource({
        contents: { id: 1, getURL: () => DEV_ORIGIN },
        mainContents: { id: 1 },
        requestingOrigin: DEV_ORIGIN,
        trustedOrigin: DEV_ORIGIN,
        isMainFrame: false,
      }),
      false,
    );
    assert.equal(
      shouldTrustMediaPermissionSource({
        contents: null,
        mainContents: { id: 1 },
        requestingOrigin: "null",
        trustedOrigin: FILE_URL,
        allowNullContents: true,
      }),
      false,
    );
  });

  it("allows null contents only on the check handler with a concrete trusted origin", () => {
    assert.equal(
      shouldTrustMediaPermissionSource({
        contents: null,
        mainContents: { id: 1 },
        requestingOrigin: DEV_ORIGIN,
        trustedOrigin: DEV_ORIGIN,
        allowNullContents: true,
      }),
      true,
    );
    assert.equal(
      shouldTrustMediaPermissionSource({
        contents: null,
        mainContents: { id: 1 },
        requestingOrigin: DEV_ORIGIN,
        trustedOrigin: DEV_ORIGIN,
      }),
      false,
    );
  });
});

describe("permission handler settlement", () => {
  it("delivers the request callback once, denying if decide throws", () => {
    const granted: boolean[] = [];
    runPermissionRequestHandler(() => true, (value) => {
      granted.push(value);
    });
    assert.deepEqual(granted, [true]);

    granted.length = 0;
    runPermissionRequestHandler(() => {
      throw new Error("destroyed webContents");
    }, (value) => {
      granted.push(value);
    });
    assert.deepEqual(granted, [false]);
  });

  it("returns false from the check handler when decide throws", () => {
    assert.equal(runPermissionCheckHandler(() => true), true);
    assert.equal(
      runPermissionCheckHandler(() => {
        throw new Error("destroyed webContents");
      }),
      false,
    );
  });
});
