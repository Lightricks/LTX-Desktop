export function mediaErrorCode(error: unknown): unknown {
  return typeof error === "object" && error !== null && "code" in error
    ? error.code
    : undefined;
}

export function mediaLookupErrorMessage(
  error: unknown,
  unavailable: string,
  fallback: string,
): string {
  if (mediaErrorCode(error) === "ASSET_NOT_FOUND") {
    return unavailable;
  }
  if (error instanceof Error && error.message.trim().length > 0) {
    return error.message;
  }
  return fallback;
}

export function mediaIngestErrorMessage(
  error: unknown,
  copy: {
    unsupported: string;
    unavailable: string;
    unreadable: readonly string[];
    couldNotAdd: string;
  },
): string {
  const code = mediaErrorCode(error);
  if (code === "UNSUPPORTED_MEDIA") {
    return copy.unsupported;
  }
  if (code === "FILE_NOT_FOUND" || code === "ASSET_NOT_FOUND") {
    return copy.unavailable;
  }
  if (error instanceof Error && copy.unreadable.includes(error.message)) {
    return error.message;
  }
  return copy.couldNotAdd;
}
