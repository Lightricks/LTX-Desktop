/**
 * Permission decision for Chromium session request + check handlers.
 *
 * Only `media` is gated: Audio to Video records via
 * `getUserMedia({ audio: true })`. Camera, mixed audio+video, unknown extra
 * types, and undeclared/empty media details are denied — never auto-grant
 * camera. Every other permission (clipboard, fullscreen, …) is left allowed
 * so installing these handlers does not change Chromium's defaults.
 */

export type MediaKinds =
  | { audio: boolean; video: boolean; extra: boolean }
  | "undeclared";

export function readMediaKinds(
  permission: string,
  details?: unknown,
): MediaKinds | null {
  if (permission !== "media") return null;
  if (typeof details !== "object" || details === null) return "undeclared";

  if ("mediaTypes" in details) {
    const mediaTypes = details.mediaTypes;
    if (!Array.isArray(mediaTypes) || mediaTypes.length === 0) {
      return "undeclared";
    }
    return {
      audio: mediaTypes.includes("audio"),
      video: mediaTypes.includes("video"),
      extra: mediaTypes.some(
        (entry) => entry !== "audio" && entry !== "video",
      ),
    };
  }

  if ("mediaType" in details) {
    const mediaType = details.mediaType;
    if (mediaType === "audio") {
      return { audio: true, video: false, extra: false };
    }
    if (mediaType === "video") {
      return { audio: false, video: true, extra: false };
    }
    return "undeclared";
  }

  return "undeclared";
}

function isAudioOnly(kinds: MediaKinds): boolean {
  return (
    kinds !== "undeclared" && kinds.audio && !kinds.video && !kinds.extra
  );
}

export function shouldGrantPermissionRequest(
  permission: string,
  details?: unknown,
): boolean {
  const kinds = readMediaKinds(permission, details);
  if (kinds === null) return true;
  return isAudioOnly(kinds);
}

export function shouldAllowPermissionCheck(
  permission: string,
  details?: unknown,
): boolean {
  const kinds = readMediaKinds(permission, details);
  if (kinds === null) return true;
  return isAudioOnly(kinds);
}

export function runPermissionRequestHandler(
  decide: () => boolean,
  callback: (granted: boolean) => void,
): void {
  let settled = false;
  const settle = (granted: boolean) => {
    if (settled) return;
    settled = true;
    callback(granted);
  };
  try {
    settle(decide());
  } catch {
    settle(false);
  } finally {
    settle(false);
  }
}

export function runPermissionCheckHandler(decide: () => boolean): boolean {
  try {
    return decide();
  } catch {
    return false;
  }
}

export function isTrustedWebContents(
  contents: { id: number } | null | undefined,
  mainContents: { id: number } | null | undefined,
): boolean {
  return (
    contents != null &&
    mainContents != null &&
    contents.id === mainContents.id
  );
}

function isConcreteOrigin(origin: string): boolean {
  if (origin.length === 0 || origin === "null") {
    return false;
  }
  try {
    const parsed = new URL(origin);
    if (parsed.protocol === "file:") {
      return parsed.pathname.length > 1;
    }
    return true;
  } catch {
    return false;
  }
}

export function isTrustedAppOrigin(
  requestingOrigin: string,
  trustedOrigin: string,
): boolean {
  if (!isConcreteOrigin(requestingOrigin) || trustedOrigin.length === 0) {
    return false;
  }
  try {
    const request = new URL(requestingOrigin);
    const trusted = new URL(trustedOrigin);
    if (trusted.protocol === "file:") {
      return (
        request.protocol === "file:" && request.pathname === trusted.pathname
      );
    }
    return request.origin === trusted.origin;
  } catch {
    return false;
  }
}

export function mediaRequestOrigin(details?: unknown): string {
  if (typeof details !== "object" || details === null) return "";
  if (
    "securityOrigin" in details &&
    typeof details.securityOrigin === "string" &&
    isConcreteOrigin(details.securityOrigin)
  ) {
    return details.securityOrigin;
  }
  if (
    "requestingUrl" in details &&
    typeof details.requestingUrl === "string" &&
    details.requestingUrl.length > 0
  ) {
    try {
      const parsed = new URL(details.requestingUrl);
      if (parsed.protocol === "file:") {
        return details.requestingUrl;
      }
      return parsed.origin;
    } catch {
      return details.requestingUrl;
    }
  }
  return "";
}

export function readIsMainFrame(details?: unknown): boolean | undefined {
  if (typeof details !== "object" || details === null) return undefined;
  if ("isMainFrame" in details && typeof details.isMainFrame === "boolean") {
    return details.isMainFrame;
  }
  return undefined;
}

export function shouldTrustMediaPermissionSource(input: {
  contents: { id: number; getURL?: () => string } | null | undefined;
  mainContents: { id: number } | null | undefined;
  requestingOrigin: string;
  trustedOrigin: string;
  allowNullContents?: boolean;
  isMainFrame?: boolean;
}): boolean {
  if (input.isMainFrame === false) {
    return false;
  }
  const contentsUrl =
    typeof input.contents?.getURL === "function" ? input.contents.getURL() : "";
  if (isTrustedWebContents(input.contents, input.mainContents)) {
    if (isConcreteOrigin(input.requestingOrigin)) {
      return isTrustedAppOrigin(input.requestingOrigin, input.trustedOrigin);
    }
    return isTrustedAppOrigin(contentsUrl, input.trustedOrigin);
  }
  if (
    input.allowNullContents === true &&
    input.contents == null &&
    isConcreteOrigin(input.requestingOrigin)
  ) {
    return isTrustedAppOrigin(input.requestingOrigin, input.trustedOrigin);
  }
  return false;
}
