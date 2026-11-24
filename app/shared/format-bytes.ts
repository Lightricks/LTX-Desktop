// Human-readable byte size, e.g. 996_000_000 -> "949.9 MB". Binary units (1024).
// Shared by the renderer and the Electron main process.
export function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB']
  const i = Math.min(sizes.length - 1, Math.floor(Math.log(bytes) / Math.log(k)))
  return `${(bytes / Math.pow(k, i)).toFixed(1)} ${sizes[i]}`
}
