function parseIpv4(hostname: string): [number, number, number, number] | null {
  const parts = hostname.split('.')
  if (parts.length !== 4) return null
  const octets = parts.map((part) => {
    if (!/^\d{1,3}$/.test(part)) return null
    const value = Number(part)
    return value <= 255 ? value : null
  })
  if (octets.some((value) => value === null)) return null
  return octets as [number, number, number, number]
}

function isLoopbackOrPrivateLanHost(hostname: string): boolean {
  const host = hostname.replace(/^\[|\]$/g, '').toLowerCase()
  if (host === 'localhost' || host === '127.0.0.1' || host === '::1') return true
  const ipv4 = parseIpv4(host)
  if (ipv4 === null) return false
  const [a, b] = ipv4
  if (a === 127) return true
  if (a === 10) return true
  if (a === 192 && b === 168) return true
  if (a === 172 && b >= 16 && b <= 31) return true
  return false
}

/** https anywhere; http only to loopback or RFC1918 so LAN pairing can open in a browser. */
export function isAllowedExternalUrl(url: string): boolean {
  let parsed: URL
  try {
    parsed = new URL(url)
  } catch {
    return false
  }
  if (parsed.username !== '' || parsed.password !== '') return false
  if (parsed.protocol === 'https:') return true
  if (parsed.protocol !== 'http:') return false
  return isLoopbackOrPrivateLanHost(parsed.hostname)
}
