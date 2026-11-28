// The upgrade download lives in the backend and survives a renderer reload.
// The "remove previous checkpoint" choice does not, so it is stored for that model
// until the download finishes.

const DELETE_OLD_PREFIX = 'ltxUpgradeDeleteOld:'

function deleteOldKey(modelId: string): string {
  return `${DELETE_OLD_PREFIX}${modelId}`
}

export function hasStoredUpgradeDeleteOld(modelId: string): boolean {
  try {
    const raw = sessionStorage.getItem(deleteOldKey(modelId))
    return raw === '1' || raw === '0'
  } catch {
    return false
  }
}

export function readUpgradeDeleteOld(modelId: string, fallback: boolean): boolean {
  try {
    const raw = sessionStorage.getItem(deleteOldKey(modelId))
    if (raw === '1') return true
    if (raw === '0') return false
  } catch {
    // sessionStorage can be unavailable; the checkbox default still applies.
  }
  return fallback
}

export function writeUpgradeDeleteOld(modelId: string, deleteOld: boolean): void {
  try {
    sessionStorage.setItem(deleteOldKey(modelId), deleteOld ? '1' : '0')
  } catch {
    // Best-effort. A reload falls back to the checkbox default.
  }
}

export function clearUpgradeDeleteOld(modelId: string): void {
  try {
    sessionStorage.removeItem(deleteOldKey(modelId))
  } catch {
    // ignore
  }
}

/** Offer stays up after the user starts a download. A refreshed app keeps that download in the background. */
export function upgradePromptVisible(options: { frozen: boolean; background: boolean }): boolean {
  if (options.frozen) return true
  return !options.background
}

/** True when the single backend download is this upgrade's checkpoints. */
export function activeDownloadMatchesUpgrade(
  activeCpIds: readonly string[],
  upgradeCpIds: readonly string[],
): boolean {
  if (activeCpIds.length === 0 || upgradeCpIds.length === 0) return false
  const active = new Set(activeCpIds)
  return upgradeCpIds.some((id) => active.has(id))
}
