import { useEffect, useState } from 'react'
import { Check, Loader2, Monitor, Server } from 'lucide-react'
import { Button } from './ui/button'

type ConnectionMode = 'managed-local' | 'external'

interface BackendConnectionPanelProps {
  onConfigured?: () => void
  compact?: boolean
}

export function BackendConnectionPanel({ onConfigured, compact = false }: BackendConnectionPanelProps) {
  const [mode, setMode] = useState<ConnectionMode>('managed-local')
  const [url, setUrl] = useState('http://127.0.0.1:8000')
  const [authToken, setAuthToken] = useState('')
  const [adminToken, setAdminToken] = useState('')
  const [hasSavedToken, setHasSavedToken] = useState(false)
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState<string | null>(null)
  const [success, setSuccess] = useState(false)

  useEffect(() => {
    void window.electronAPI.getBackendConnectionConfig().then((config) => {
      setMode(config.mode)
      if (config.url) setUrl(config.url)
      setHasSavedToken(config.hasAuthToken)
    }).catch((error) => setMessage(String(error)))
  }, [])

  const testConnection = async () => {
    if (!authToken.trim()) {
      setSuccess(false)
      setMessage(hasSavedToken
        ? 'Re-enter the saved authentication token to test or update this connection.'
        : 'Enter the backend authentication token.')
      return
    }
    setBusy(true)
    setMessage(null)
    setSuccess(false)
    try {
      const result = await window.electronAPI.testBackendConnection({ url, authToken })
      if (!result.success) throw new Error(result.error)
      setSuccess(true)
      setMessage(`Connected to standalone API v${result.serverInfo.api_version}.`)
    } catch (error) {
      setMessage(error instanceof Error ? error.message : String(error))
    } finally {
      setBusy(false)
    }
  }

  const saveConnection = async () => {
    setBusy(true)
    setMessage(null)
    setSuccess(false)
    try {
      if (mode === 'external' && !authToken.trim()) {
        throw new Error('Enter the backend authentication token before saving.')
      }
      const result = mode === 'managed-local'
        ? await window.electronAPI.setBackendConnection({ mode: 'managed-local' })
        : await window.electronAPI.setBackendConnection({
            mode: 'external',
            url,
            authToken,
            ...(adminToken.trim() ? { adminToken } : {}),
          })
      if (!result.success) throw new Error(result.error)
      setSuccess(true)
      setMessage('Connection saved. Reloading LTX Desktop…')
      onConfigured?.()
      window.setTimeout(() => window.location.reload(), 250)
    } catch (error) {
      setMessage(error instanceof Error ? error.message : String(error))
      setBusy(false)
    }
  }

  return (
    <div className={`space-y-4 ${compact ? '' : 'rounded-xl border border-zinc-700 bg-zinc-900/70 p-5'}`}>
      <div>
        <h3 className="text-sm font-semibold text-white">Compute backend</h3>
        <p className="mt-1 text-xs leading-relaxed text-zinc-500">
          Run inference on this computer, or connect to a standalone LTX backend through an SSH tunnel or HTTPS.
        </p>
      </div>

      <div className="grid grid-cols-2 gap-2">
        <button
          type="button"
          onClick={() => { setMode('managed-local'); setMessage(null); setSuccess(false) }}
          className={`rounded-lg border p-3 text-left transition-colors ${mode === 'managed-local' ? 'border-blue-500 bg-blue-500/10' : 'border-zinc-700 bg-zinc-800/50 hover:border-zinc-600'}`}
        >
          <Monitor className="mb-2 h-4 w-4 text-blue-400" />
          <div className="text-sm font-medium text-white">This computer</div>
          <div className="mt-1 text-[11px] text-zinc-500">Managed Python and models</div>
        </button>
        <button
          type="button"
          onClick={() => { setMode('external'); setMessage(null); setSuccess(false) }}
          className={`rounded-lg border p-3 text-left transition-colors ${mode === 'external' ? 'border-blue-500 bg-blue-500/10' : 'border-zinc-700 bg-zinc-800/50 hover:border-zinc-600'}`}
        >
          <Server className="mb-2 h-4 w-4 text-blue-400" />
          <div className="text-sm font-medium text-white">Remote machine</div>
          <div className="mt-1 text-[11px] text-zinc-500">Standalone backend and GPU</div>
        </button>
      </div>

      {mode === 'external' && (
        <div className="space-y-3">
          <label className="block">
            <span className="mb-1 block text-xs text-zinc-400">Backend URL</span>
            <input
              value={url}
              onChange={(event) => setUrl(event.target.value)}
              onKeyDown={(event) => event.stopPropagation()}
              placeholder="http://127.0.0.1:8000"
              className="w-full rounded-lg border border-zinc-700 bg-zinc-800 px-3 py-2 text-sm text-white outline-none focus:border-blue-500"
            />
          </label>
          <label className="block">
            <span className="mb-1 block text-xs text-zinc-400">Authentication token</span>
            <input
              type="password"
              value={authToken}
              onChange={(event) => setAuthToken(event.target.value)}
              onKeyDown={(event) => event.stopPropagation()}
              placeholder={hasSavedToken ? 'Saved securely — re-enter to change' : 'Required'}
              className="w-full rounded-lg border border-zinc-700 bg-zinc-800 px-3 py-2 text-sm text-white outline-none focus:border-blue-500"
            />
          </label>
          <label className="block">
            <span className="mb-1 block text-xs text-zinc-400">Admin token <span className="text-zinc-600">(optional)</span></span>
            <input
              type="password"
              value={adminToken}
              onChange={(event) => setAdminToken(event.target.value)}
              onKeyDown={(event) => event.stopPropagation()}
              placeholder="Only needed to change Atom settings"
              className="w-full rounded-lg border border-zinc-700 bg-zinc-800 px-3 py-2 text-sm text-white outline-none focus:border-blue-500"
            />
          </label>
          <p className="text-[11px] leading-relaxed text-zinc-500">
            Plain HTTP is accepted only on localhost. For an Atom bound to port 8000, use an SSH tunnel to the same Mac port.
          </p>
        </div>
      )}

      {message && (
        <div className={`flex items-start gap-2 rounded-lg border px-3 py-2 text-xs ${success ? 'border-emerald-500/30 bg-emerald-500/10 text-emerald-300' : 'border-amber-500/30 bg-amber-500/10 text-amber-200'}`}>
          {success && <Check className="mt-0.5 h-3.5 w-3.5 flex-shrink-0" />}
          <span>{message}</span>
        </div>
      )}

      <div className="flex justify-end gap-2">
        {mode === 'external' && (
          <Button variant="outline" onClick={() => { void testConnection() }} disabled={busy} className="border-zinc-700">
            {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : 'Test connection'}
          </Button>
        )}
        <Button onClick={() => { void saveConnection() }} disabled={busy}>
          {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : 'Save & reconnect'}
        </Button>
      </div>
    </div>
  )
}
