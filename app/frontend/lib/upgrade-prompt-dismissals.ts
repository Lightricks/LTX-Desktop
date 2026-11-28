// Model ids whose upgrade prompt the user closed. Closing hides it for that
// target; the checkpoint stays downloadable in Settings → Models.
// Best-effort: if localStorage is unavailable the prompt returns next launch.
const KEY = 'ltxDismissedUpgrades'

function read(): string[] {
  try {
    const raw = localStorage.getItem(KEY)
    if (!raw) return []
    const parsed: unknown = JSON.parse(raw)
    return Array.isArray(parsed) ? parsed.filter((v): v is string => typeof v === 'string') : []
  } catch {
    return []
  }
}

export function isUpgradeDismissed(modelId: string): boolean {
  return read().includes(modelId)
}

export function dismissUpgrade(modelId: string): void {
  try {
    const ids = read()
    if (!ids.includes(modelId)) {
      localStorage.setItem(KEY, JSON.stringify([...ids, modelId]))
    }
  } catch {
    // ignore — persistence is best-effort
  }
}
