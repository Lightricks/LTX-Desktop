// A path that starts with two separators is a UNC (network) path on Windows.
// Windows starts an SMB connection when the app touches such a path. That can
// send the user's NTLM hash to a remote server. `\\?\C:\...` is a local
// long path, so it is allowed.
const LOCAL_LONG_PATH = /^[\\/]{2}\?[\\/][A-Za-z]:/
const DOUBLE_SEPARATOR_PREFIX = /^[\\/]{2}/

export function isRemoteFilePath(filePath: string): boolean {
  const trimmed = filePath.trim()
  return DOUBLE_SEPARATOR_PREFIX.test(trimmed) && !LOCAL_LONG_PATH.test(trimmed)
}
