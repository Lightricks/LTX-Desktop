const PAIRING_TOKEN_STORAGE_KEY = 'ltx-remote-pairing-token'

let bearerToken: string | null = null

export function apiBase(): string {
  const value = (globalThis as { __LTX_REMOTE_API_BASE__?: string }).__LTX_REMOTE_API_BASE__
  return typeof value === 'string' ? value.replace(/\/$/, '') : ''
}

export function tokenFromHref(href: string): string | null {
  const token = new URL(href).searchParams.get('t')
  return token && token.length > 0 ? token : null
}

export function tokenFromLocation(location: { search: string }): string | null {
  const token = new URLSearchParams(location.search).get('t')
  return token && token.length > 0 ? token : null
}

export function stripPairingTokenFromHref(href: string): string {
  const url = new URL(href)
  url.searchParams.delete('t')
  return `${url.pathname}${url.search}${url.hash}`
}

function persistPairingToken(token: string | null): void {
  try {
    if (token) {
      globalThis.localStorage.setItem(PAIRING_TOKEN_STORAGE_KEY, token)
    } else {
      globalThis.localStorage.removeItem(PAIRING_TOKEN_STORAGE_KEY)
    }
  } catch {
    // localStorage can throw in private mode or when storage is blocked
  }
}

function readPersistedPairingToken(): string | null {
  try {
    const value = globalThis.localStorage.getItem(PAIRING_TOKEN_STORAGE_KEY)
    return value && value.length > 0 ? value : null
  } catch {
    return null
  }
}

function throwAwayPairingToken(): void {
  const href = globalThis.window.location.href
  const next = stripPairingTokenFromHref(href)
  const url = new URL(href)
  const current = `${url.pathname}${url.search}${url.hash}`
  if (next === current) {
    return
  }
  try {
    const history = globalThis.history
    history.replaceState(history.state, '', next)
  } catch {
    // History may be missing in tests or reject replaceState
  }
}

export function getBearerToken(): string | null {
  if (bearerToken) {
    return bearerToken
  }
  const stored = readPersistedPairingToken()
  if (stored) {
    bearerToken = stored
  }
  return bearerToken
}

export function setBearerToken(token: string | null): void {
  bearerToken = token
  persistPairingToken(token)
}

async function exchangePairingToken(grant: string): Promise<string | null> {
  const headers = new Headers({ 'Content-Type': 'application/json' })
  const existing = getBearerToken()
  if (existing) {
    headers.set('Authorization', `Bearer ${existing}`)
  }
  const response = await fetch(`${apiBase()}/api/pairing/exchange`, {
    method: 'POST',
    headers,
    body: JSON.stringify({ code: grant }),
  })
  if (!response.ok) {
    return null
  }
  const payload: unknown = await response.json()
  if (
    typeof payload !== 'object' ||
    payload === null ||
    !('token' in payload) ||
    typeof payload.token !== 'string' ||
    payload.token.length === 0
  ) {
    return null
  }
  return payload.token
}

async function pairDeviceOnce(
  capturedGrant?: string | null,
): Promise<'ok' | 'missing' | 'failed'> {
  const grant =
    capturedGrant && capturedGrant.length > 0
      ? capturedGrant
      : tokenFromHref(globalThis.window.location.href)
  if (grant) {
    let session: string | null = null
    try {
      session = await exchangePairingToken(grant)
    } catch {
      session = null
    }
    if (session !== null) {
      setBearerToken(session)
      throwAwayPairingToken()
      return 'ok'
    }
    if (getBearerToken()) {
      throwAwayPairingToken()
      return 'ok'
    }
    return 'failed'
  }
  return getBearerToken() ? 'ok' : 'missing'
}

let inFlightPairing: Promise<'ok' | 'missing' | 'failed'> | null = null

/** Read the URL grant, pair the device, then throw the grant away. */
export function pairDevice(capturedGrant?: string | null): Promise<'ok' | 'missing' | 'failed'> {
  if (inFlightPairing) {
    return inFlightPairing
  }
  inFlightPairing = pairDeviceOnce(capturedGrant).finally(() => {
    inFlightPairing = null
  })
  return inFlightPairing
}

export async function api(path: string, init: RequestInit = {}): Promise<Response> {
  const headers = new Headers(init.headers)
  const token = getBearerToken()
  if (token) {
    headers.set('Authorization', `Bearer ${token}`)
  }
  const response = await fetch(`${apiBase()}${path}`, { ...init, headers })
  if (response.status === 401) {
    setBearerToken(null)
  }
  return response
}
