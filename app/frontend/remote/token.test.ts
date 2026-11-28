import assert from 'node:assert/strict'
import { afterEach, describe, it } from 'node:test'
import {
  api,
  apiBase,
  getBearerToken,
  pairDevice,
  setBearerToken,
  stripPairingTokenFromHref,
  tokenFromHref,
  tokenFromLocation,
} from './api.ts'

type MemoryStorage = {
  getItem: (key: string) => string | null
  setItem: (key: string, value: string) => void
  removeItem: (key: string) => void
  clear: () => void
}

function createMemoryStorage(store: Map<string, string>): MemoryStorage {
  return {
    getItem: (key: string) => store.get(key) ?? null,
    setItem: (key: string, value: string) => {
      store.set(key, value)
    },
    removeItem: (key: string) => {
      store.delete(key)
    },
    clear: () => {
      store.clear()
    },
  }
}

function installPairingBrowser(href: string) {
  const current = new URL(href)
  const store = new Map<string, string>()
  const localStorage = createMemoryStorage(store)
  const history = {
    state: null as unknown,
    replaceState(state: unknown, _title: string, url: string) {
      history.state = state
      current.href = new URL(url, current.origin).href
    },
  }
  const location = {
    get href() {
      return current.href
    },
    get pathname() {
      return current.pathname
    },
    get search() {
      return current.search
    },
    get hash() {
      return current.hash
    },
  }
  Object.defineProperty(globalThis, 'localStorage', {
    configurable: true,
    value: localStorage,
  })
  Object.defineProperty(globalThis, 'history', {
    configurable: true,
    value: history,
  })
  Object.defineProperty(globalThis, 'window', {
    configurable: true,
    value: { location, history, localStorage },
  })
  return { location, store, localStorage }
}

afterEach(() => {
  setBearerToken(null)
  Reflect.deleteProperty(globalThis as { __LTX_REMOTE_API_BASE__?: string }, '__LTX_REMOTE_API_BASE__')
})

describe('remote pairing token', () => {
  it('reads t from the search param', () => {
    assert.equal(tokenFromHref('http://192.168.1.9:41955/?t=secret'), 'secret')
    assert.equal(tokenFromHref('http://192.168.1.9:41955/pairing?t=secret'), 'secret')
    assert.equal(tokenFromHref('http://192.168.1.9:41955/assets?t=secret'), 'secret')
    assert.equal(
      tokenFromLocation({ search: '?t=secret' }),
      'secret',
    )
  })

  it('ignores a hash t= value', () => {
    assert.equal(tokenFromHref('http://192.168.1.9:41955/#t=grant'), null)
    assert.equal(tokenFromHref('http://192.168.1.9:41955/assets#t=grant'), null)
    assert.equal(
      tokenFromHref('http://192.168.1.9:41955/assets?t=search#t=hash'),
      'search',
    )
  })

  it('strips query t while keeping unrelated search and hash values', () => {
    assert.equal(
      stripPairingTokenFromHref('http://192.168.1.9:41955/?t=secret&keep=1#other=yes&t=not-a-grant'),
      '/?keep=1#other=yes&t=not-a-grant',
    )
    assert.equal(
      stripPairingTokenFromHref('http://192.168.1.9:41955/pairing?t=secret'),
      '/pairing',
    )
    assert.equal(
      stripPairingTokenFromHref('http://192.168.1.9:41955/text-to-video?t=secret'),
      '/text-to-video',
    )
    assert.equal(
      stripPairingTokenFromHref('http://192.168.1.9:41955/#t=not-a-grant'),
      '/#t=not-a-grant',
    )
    assert.equal(
      stripPairingTokenFromHref('http://192.168.1.9:41955/assets?keep=1#section'),
      '/assets?keep=1#section',
    )
  })

  it('defaults apiBase to same-origin and strips a trailing slash', () => {
    assert.equal(apiBase(), '')
    ;(globalThis as { __LTX_REMOTE_API_BASE__?: string }).__LTX_REMOTE_API_BASE__ =
      'https://connect.ltx.io/'
    assert.equal(apiBase(), 'https://connect.ltx.io')
  })

  it('pairs with the URL token then throws the token away', async () => {
    const { location, store } = installPairingBrowser(
      'http://192.168.1.9:41955/text-to-video?t=grant&keep=1#other=yes',
    )
    const previousFetch = globalThis.fetch
    globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
      assert.equal(String(input), '/api/pairing/exchange')
      assert.equal(init?.method, 'POST')
      assert.equal(location.search, '?t=grant&keep=1')
      return new Response(JSON.stringify({ token: 'session-token', device_id: 'dev-1', name: 'Phone' }), {
        status: 200,
      })
    }) as typeof fetch
    try {
      assert.equal(await pairDevice(), 'ok')
      assert.equal(getBearerToken(), 'session-token')
      assert.equal([...store.values()].includes('session-token'), true)
      assert.equal(location.search, '?keep=1')
      assert.equal(location.hash, '#other=yes')
      assert.equal(location.pathname, '/text-to-video')
    } finally {
      globalThis.fetch = previousFetch
    }
  })

  it('leaves the URL token in place when pairing fails', async () => {
    const { location } = installPairingBrowser('http://192.168.1.9:41955/?t=grant')
    const previousFetch = globalThis.fetch
    globalThis.fetch = (async () => new Response('', { status: 401 })) as typeof fetch
    try {
      assert.equal(await pairDevice(), 'failed')
      assert.equal(getBearerToken(), null)
      assert.equal(location.search, '?t=grant')
    } finally {
      globalThis.fetch = previousFetch
    }
  })

  it('keeps a stored session when a leftover grant fails', async () => {
    const { location } = installPairingBrowser('http://192.168.1.9:41955/?t=stale')
    setBearerToken('session-token')
    const previousFetch = globalThis.fetch
    globalThis.fetch = (async () => new Response('', { status: 401 })) as typeof fetch
    try {
      assert.equal(await pairDevice(), 'ok')
      assert.equal(getBearerToken(), 'session-token')
      assert.equal(location.search, '')
    } finally {
      globalThis.fetch = previousFetch
    }
  })

  it('keeps a stored session when exchange throws', async () => {
    const { location } = installPairingBrowser('http://192.168.1.9:41955/?t=grant')
    setBearerToken('session-token')
    const previousFetch = globalThis.fetch
    globalThis.fetch = (async () => {
      throw new TypeError('Failed to fetch')
    }) as typeof fetch
    try {
      assert.equal(await pairDevice(), 'ok')
      assert.equal(getBearerToken(), 'session-token')
      assert.equal(location.search, '')
    } finally {
      globalThis.fetch = previousFetch
    }
  })

  it('maps a thrown exchange to failed when nothing is stored', async () => {
    const { location } = installPairingBrowser('http://192.168.1.9:41955/?t=grant')
    const previousFetch = globalThis.fetch
    globalThis.fetch = (async () => {
      throw new TypeError('Failed to fetch')
    }) as typeof fetch
    try {
      assert.equal(await pairDevice(), 'failed')
      assert.equal(getBearerToken(), null)
      assert.equal(location.search, '?t=grant')
    } finally {
      globalThis.fetch = previousFetch
    }
  })

  it('is missing when the URL has no token and nothing is stored', async () => {
    installPairingBrowser('http://192.168.1.9:41955/')
    assert.equal(await pairDevice(), 'missing')
    installPairingBrowser('http://192.168.1.9:41955/#t=grant')
    assert.equal(await pairDevice(), 'missing')
  })

  it('pairs with a grant captured before the address bar was scrubbed', async () => {
    const { location } = installPairingBrowser('http://192.168.1.9:41955/pairing?t=grant')
    const grant = tokenFromHref(location.href)
    globalThis.history.replaceState(null, '', '/')
    assert.equal(location.search, '')
    assert.equal(location.hash, '')
    const previousFetch = globalThis.fetch
    globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
      assert.equal(String(input), '/api/pairing/exchange')
      assert.deepEqual(JSON.parse(String(init?.body)), { code: 'grant' })
      return new Response(JSON.stringify({ token: 'session-token', device_id: 'dev-1', name: 'Phone' }), {
        status: 200,
      })
    }) as typeof fetch
    try {
      assert.equal(await pairDevice(grant), 'ok')
      assert.equal(getBearerToken(), 'session-token')
    } finally {
      globalThis.fetch = previousFetch
    }
  })

  it('sends the stored session when exchanging a new QR grant', async () => {
    installPairingBrowser('http://192.168.1.9:41955/pairing?t=grant')
    setBearerToken('old-session')
    const previousFetch = globalThis.fetch
    globalThis.fetch = (async (_input: RequestInfo | URL, init?: RequestInit) => {
      const headers = new Headers(init?.headers)
      assert.equal(headers.get('Authorization'), 'Bearer old-session')
      return new Response(JSON.stringify({ token: 'new-session', device_id: 'dev-1', name: 'Phone' }), {
        status: 200,
      })
    }) as typeof fetch
    try {
      assert.equal(await pairDevice(), 'ok')
      assert.equal(getBearerToken(), 'new-session')
    } finally {
      globalThis.fetch = previousFetch
    }
  })

  it('shares one exchange across overlapping pairDevice calls', async () => {
    installPairingBrowser('http://192.168.1.9:41955/pairing?t=grant')
    let calls = 0
    let release: ((value: Response) => void) | undefined
    const gate = new Promise<Response>((resolve) => {
      release = resolve
    })
    const previousFetch = globalThis.fetch
    globalThis.fetch = (async () => {
      calls += 1
      return gate
    }) as typeof fetch
    try {
      const first = pairDevice()
      const second = pairDevice()
      assert.equal(calls, 1)
      release!(
        new Response(JSON.stringify({ token: 'session-token', device_id: 'dev-1', name: 'Phone' }), {
          status: 200,
        }),
      )
      assert.equal(await first, 'ok')
      assert.equal(await second, 'ok')
      assert.equal(calls, 1)
      assert.equal(getBearerToken(), 'session-token')
    } finally {
      globalThis.fetch = previousFetch
    }
  })

  it('skips exchange when a stored session already exists', async () => {
    installPairingBrowser('http://192.168.1.9:41955/')
    setBearerToken('session-token')
    const previousFetch = globalThis.fetch
    globalThis.fetch = (async () => {
      throw new Error('should not exchange')
    }) as typeof fetch
    try {
      assert.equal(await pairDevice(), 'ok')
      assert.equal(getBearerToken(), 'session-token')
    } finally {
      globalThis.fetch = previousFetch
    }
  })

  it('recovers the session token from localStorage after a reload', () => {
    const { store } = installPairingBrowser('http://192.168.1.9:41955/')
    setBearerToken('session-token')
    const snapshot = new Map(store)

    setBearerToken(null)
    assert.equal(getBearerToken(), null)
    for (const [key, value] of snapshot) {
      store.set(key, value)
    }
    assert.equal(getBearerToken(), 'session-token')
  })

  it('clears localStorage when the bearer token is removed', () => {
    const { store } = installPairingBrowser('http://192.168.1.9:41955/')
    setBearerToken('session-token')
    assert.equal(store.size > 0, true)
    setBearerToken(null)
    assert.equal(store.size, 0)
    assert.equal(getBearerToken(), null)
  })

  it('does not throw when localStorage is unavailable', () => {
    Object.defineProperty(globalThis, 'localStorage', {
      configurable: true,
      value: {
        getItem: () => {
          throw new Error('blocked')
        },
        setItem: () => {
          throw new Error('blocked')
        },
        removeItem: () => {
          throw new Error('blocked')
        },
      },
    })
    setBearerToken('still-in-memory')
    assert.equal(getBearerToken(), 'still-in-memory')
    setBearerToken(null)
    assert.equal(getBearerToken(), null)
  })

  it('sends the bearer token and clears it on a 401 response', async () => {
    const { store } = installPairingBrowser('http://192.168.1.9:41955/')
    setBearerToken('stale')
    const previousFetch = globalThis.fetch
    globalThis.fetch = (async (_input: RequestInfo | URL, init?: RequestInit) => {
      const headers = new Headers(init?.headers)
      assert.equal(headers.get('Authorization'), 'Bearer stale')
      return new Response('', { status: 401 })
    }) as typeof fetch
    try {
      const response = await api('/api/session')
      assert.equal(response.status, 401)
      assert.equal(getBearerToken(), null)
      assert.equal(store.size, 0)
    } finally {
      globalThis.fetch = previousFetch
    }
  })

  it('keeps the stored token on a 403 media response', async () => {
    installPairingBrowser('http://192.168.1.9:41955/')
    setBearerToken('session-token')
    const previousFetch = globalThis.fetch
    globalThis.fetch = (async () => new Response('', { status: 403 })) as typeof fetch
    try {
      const response = await api('/api/assets/asset-1/bytes')
      assert.equal(response.status, 403)
      assert.equal(getBearerToken(), 'session-token')
    } finally {
      globalThis.fetch = previousFetch
    }
  })

  it('keeps the stored token on a successful api response', async () => {
    installPairingBrowser('http://192.168.1.9:41955/')
    setBearerToken('session-token')
    const previousFetch = globalThis.fetch
    globalThis.fetch = (async () => new Response('', { status: 200 })) as typeof fetch
    try {
      const response = await api('/api/session')
      assert.equal(response.status, 200)
      assert.equal(getBearerToken(), 'session-token')
    } finally {
      globalThis.fetch = previousFetch
    }
  })
})
