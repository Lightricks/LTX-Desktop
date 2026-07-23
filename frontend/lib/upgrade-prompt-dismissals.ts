// Persisted "don't show the LTX upgrade prompt again" set, keyed by target model id.
// Best-effort: if localStorage is unavailable the prompt simply reappears next session.
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
