import { session } from 'electron'
import { isDev } from './config'
import { getBackendConnectionSummary } from './app-state'

function getConfiguredBackendSources(): string[] {
  const summary = getBackendConnectionSummary()
  if (summary.mode !== 'external' || !summary.url) return []
  try {
    const origin = new URL(summary.url).origin
    const wsOrigin = origin.replace(/^http:/, 'ws:').replace(/^https:/, 'wss:')
    return [origin, wsOrigin]
  } catch {
    return []
  }
}

// Enforce Content Security Policy via response headers (tamper-proof from renderer)
export function setupCSP(): void {
  session.defaultSession.webRequest.onHeadersReceived((details, callback) => {
    const configuredBackendSources = getConfiguredBackendSources().join(' ')
    const connectSources = `'self' http://localhost:* http://127.0.0.1:* ws://localhost:* ws://127.0.0.1:*${configuredBackendSources ? ` ${configuredBackendSources}` : ''}`
    const csp = isDev
      ? [
          "default-src 'self'",
          "script-src 'self' 'unsafe-inline'",
          "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
          "font-src 'self' https://fonts.gstatic.com",
          `connect-src ${connectSources}`,
          "img-src 'self' data: blob: file:",
          "media-src 'self' blob: file:",
          "object-src 'none'",
          "base-uri 'self'",
          "form-action 'self'",
          "frame-ancestors 'none'",
        ].join('; ')
      : [
          "default-src 'self'",
          "script-src 'self'",
          "style-src 'self' https://fonts.googleapis.com",
          "font-src 'self' https://fonts.gstatic.com",
          `connect-src ${connectSources}`,
          "img-src 'self' data: blob: file:",
          "media-src 'self' blob: file:",
          "object-src 'none'",
          "base-uri 'self'",
          "form-action 'self'",
          "frame-ancestors 'none'",
        ].join('; ')

    callback({
      responseHeaders: {
        ...details.responseHeaders,
        'Content-Security-Policy': [csp],
      },
    })
  })
}
