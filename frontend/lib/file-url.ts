/**
 * Converts a filesystem path to a properly encoded file:// URL.
 *
 *   /Users/me/my file.mp4   → file:///Users/me/my%20file.mp4
 *   C:\Users\me\video#1.mp4 → file:///C:/Users/me/video%231.mp4
 */
export function pathToFileUrl(filePath: string): string {
  // Normalize Windows separators
  let normalized = filePath.replace(/\\/g, '/')

  // Ensure leading slash (Windows drive letters like C:/ need one prepended)
  if (!normalized.startsWith('/')) {
    normalized = '/' + normalized
  }

  // Encode each path segment individually so we don't encode the slashes
  const encoded = normalized
    .split('/')
    .map((segment) => encodeURIComponent(segment))
    .join('/')

  return 'file://' + encoded
}
