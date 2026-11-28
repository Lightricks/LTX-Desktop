export async function resolveOpenMicrophoneSettings(
  platform: string,
  openExternal: (url: string) => Promise<void>,
): Promise<{ success: true } | { success: false; error: string }> {
  try {
    if (platform === "win32") {
      await openExternal("ms-settings:privacy-microphone");
      return { success: true };
    }
    if (platform === "darwin") {
      await openExternal(
        "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone",
      );
      return { success: true };
    }
    return {
      success: false,
      error: "Microphone settings are not available on this platform",
    };
  } catch (error: unknown) {
    return {
      success: false,
      error:
        error instanceof Error
          ? error.message
          : "Could not open microphone settings",
    };
  }
}
