import { AlertCircle } from 'lucide-react'
import type { ApiSuccessOf } from '../lib/api-client'
import { Button } from './ui/button'

type HfAuthStatus = ApiSuccessOf<'getHuggingFaceAuthStatus'>['status']
type ModelAccessMap = ApiSuccessOf<'checkModelAccess'>['access']

interface HfModelAccessGateProps {
  accessMap: ModelAccessMap
  allAuthorized: boolean
  hfAuthStatus: HfAuthStatus
  hfAuthPolling: boolean
  startHuggingFaceLogin: () => void
  /** When the access check itself failed (network/backend), distinct from unauthorized. */
  checkError?: string | null
  onRetryCheck?: () => void
  className?: string
}

export function HfModelAccessGate({
  accessMap,
  allAuthorized,
  hfAuthStatus,
  hfAuthPolling,
  startHuggingFaceLogin,
  checkError = null,
  onRetryCheck,
  className,
}: HfModelAccessGateProps) {
  if (allAuthorized) return null

  if (checkError) {
    return (
      <div className={className ?? 'space-y-2'}>
        <div className="flex items-start gap-2 text-xs text-amber-400">
          <AlertCircle className="h-3.5 w-3.5 flex-shrink-0 mt-0.5" />
          <span>Couldn&apos;t verify Hugging Face access: {checkError}</span>
        </div>
        {onRetryCheck && (
          <Button
            size="sm"
            variant="outline"
            onClick={onRetryCheck}
            className="text-xs"
          >
            Retry
          </Button>
        )}
      </div>
    )
  }

  const unauthorizedRepos = Object.entries(accessMap).filter(([, status]) => status === 'not_authorized')
  if (unauthorizedRepos.length === 0) return null

  if (hfAuthStatus !== 'authenticated') {
    return (
      <div className={className ?? 'space-y-2'}>
        <div className="flex items-start gap-2 text-xs text-amber-400">
          <AlertCircle className="h-3.5 w-3.5 flex-shrink-0 mt-0.5" />
          <span>
            This model is gated on Hugging Face. Sign in, then accept the license to download.
          </span>
        </div>
        <Button
          size="sm"
          onClick={startHuggingFaceLogin}
          disabled={hfAuthPolling}
          className="bg-indigo-600 hover:bg-indigo-500 text-white text-xs"
        >
          {hfAuthPolling ? 'Waiting for sign in…' : 'Sign in with Hugging Face'}
        </Button>
      </div>
    )
  }

  return (
    <div className={className ?? 'space-y-1.5'}>
      <p className="text-xs text-amber-400">
        Accept the Hugging Face license for this model, then download.
      </p>
      {unauthorizedRepos.map(([repoId]) => (
        <div key={repoId} className="flex items-center justify-between gap-2 bg-zinc-900 rounded px-2 py-1.5">
          <span className="text-[10px] text-zinc-400 font-mono truncate">{repoId}</span>
          <button
            type="button"
            onClick={() => {
              void window.electronAPI.openHuggingFaceRepo({ repoId })
            }}
            className="text-[10px] text-indigo-400 hover:text-indigo-300 font-medium flex-shrink-0"
          >
            Request access
          </button>
        </div>
      ))}
    </div>
  )
}
