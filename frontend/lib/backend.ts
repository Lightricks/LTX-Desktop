export interface BackendCredentials {
  url: string
  token: string
  mode: 'managed-local' | 'external'
  connectionRevision: number
}

let cached: BackendCredentials | null = null

export async function getBackendCredentials(): Promise<BackendCredentials> {
  if (!cached) cached = await window.electronAPI.getBackend()
  return cached
}

export function resetBackendCredentials(): void {
  cached = null
}

export async function backendFetch(path: string, init?: RequestInit): Promise<Response> {
  const { url, token } = await getBackendCredentials()
  const headers = new Headers(init?.headers)
  if (token) headers.set('Authorization', `Bearer ${token}`)
  return fetch(`${url}${path}`, { ...init, headers })
}

export async function backendWsUrl(path: string): Promise<string> {
  const { url, token } = await getBackendCredentials()
  const ws = url.replace(/^http:/, 'ws:').replace(/^https:/, 'wss:')
  const sep = path.includes('?') ? '&' : '?'
  return `${ws}${path}${sep}token=${token}`
}
