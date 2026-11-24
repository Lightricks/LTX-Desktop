type PairingStatus = {
  permitted: boolean
  serving: boolean
  reason?: string | null
}

const STARTUP_REASONS = new Set([
  "starting",
  "stopped",
  "not started",
  // Controller stores this while Off. After On, settings sync + the 2s poll still return it.
  "disabled in Settings",
])

function isPairingLoading(status: PairingStatus | null): boolean {
  if (!status?.permitted) {
    return false
  }
  if (status.serving) {
    return false
  }
  const reason = status.reason
  if (reason == null || reason === "") {
    return true
  }
  return STARTUP_REASONS.has(reason)
}

/** Laptop copy while Remote is On but not yet serving a pairing URL. */
export function pairingErrorMessage(
  on: boolean,
  serving: boolean,
  status: PairingStatus | null,
  unreachable: boolean,
): string | null {
  if (!on || serving) {
    return null
  }
  if (!status) {
    return unreachable ? "Remote is not available" : null
  }
  if (!status.permitted) {
    return status.reason ?? "Remote is not available"
  }
  if (isPairingLoading(status)) {
    return null
  }
  return status.reason ?? "Remote is not available"
}
