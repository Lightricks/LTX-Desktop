/**
 * OS-native label for the "show this file in the system file manager" action,
 * following the VS Code convention.
 */
export function revealInFolderLabel(): string {
  const platform = window.electronAPI?.platform;
  if (platform === "darwin") return "Reveal in Finder";
  if (platform === "win32") return "Reveal in File Explorer";
  return "Open Containing Folder";
}
