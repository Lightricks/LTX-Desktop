/** Backend only refreshes last_seen at most once per minute. Stay a bit longer. */
export const REMOTE_DEVICE_IN_USE_WITHIN_MS = 90_000

export function isRemoteDeviceInUse(
  lastSeenAt: number,
  nowMs: number = Date.now(),
): boolean {
  return nowMs - lastSeenAt <= REMOTE_DEVICE_IN_USE_WITHIN_MS
}
