export function folderRevealPath(
  revealInFolder: ((filePath: string) => void) | null,
  path: string | null | undefined,
): string | null {
  if (revealInFolder === null || path == null || path.length === 0) {
    return null;
  }
  return path;
}
