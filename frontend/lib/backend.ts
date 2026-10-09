let cached: { url: string; token: string } | null = null

export type BackendFetch = (path: string, init?: RequestInit) => Promise<Response>

export type BackendWsUrl = (path: string) => Promise<string>

export async function getBackendCredentials(): Promise<{ url: string; token: string }> {
  if (!cached) cached = await window.electronAPI.getBackend()
  return cached
}

export function resetBackendCredentials(): void {
  cached = null
}

export function createBackendWsUrl(
  getBase: () => Promise<{ url: string; token: string }>,
): BackendWsUrl {
  return async (path: string) => {
    const { url, token } = await getBase()
    const ws = url.replace('http://', 'ws://').replace('https://', 'wss://')
    const sep = path.includes('?') ? '&' : '?'
    return `${ws}${path}${sep}token=${token}`
  }
}

export async function backendFetch(path: string, init?: RequestInit): Promise<Response> {
  const { url, token } = await getBackendCredentials()
  const headers = new Headers(init?.headers)
  if (token) headers.set('Authorization', `Bearer ${token}`)
  return fetch(`${url}${path}`, { ...init, headers })
}

export const backendWsUrl: BackendWsUrl = createBackendWsUrl(getBackendCredentials)
