import { useState, useEffect, useCallback } from 'react'
import { ApiClient, type ApiSuccessOf } from '../lib/api-client'
import { logger } from '../lib/logger'

type HfAuthStatus = ApiSuccessOf<'getHuggingFaceAuthStatus'>['status']

interface UseHfAuthResult {
  hfAuthStatus: HfAuthStatus
  hfAuthPolling: boolean
  startHuggingFaceLogin: () => Promise<void>
  handleHuggingFaceLogout: () => Promise<void>
}

const NOOP = async () => {}

export type HuggingFaceAuthHandoff = (params: {
  clientId: string
  redirectUri: string
  scope: string
  state: string
  codeChallenge: string
  codeChallengeMethod: string
}) => Promise<boolean | void>

function desktopHuggingFaceAuthHandoff(
  params: Parameters<HuggingFaceAuthHandoff>[0],
): Promise<boolean> {
  return window.electronAPI.openHuggingFaceAuth(params)
}

// The subset of the API client HF auth needs. Defaulting to the desktop-bound
// `ApiClient` keeps Settings / first-run unchanged. Shared Home/Remote callers
// pass the host runtime `api` and a host `openAuth` (or `null` on Remote) so
// this hook never assumes Electron is present.
type HfAuthApi = Pick<
  typeof ApiClient,
  'getHuggingFaceAuthStatus' | 'startHuggingFaceLogin' | 'huggingFaceLogout'
>

export function useHfAuth(
  enabled: boolean,
  api: HfAuthApi = ApiClient,
  openAuth: HuggingFaceAuthHandoff | null | undefined = undefined,
): UseHfAuthResult {
  // Used for gated downloads (LTX 2.5 base models, catalog LoRAs / IC-LoRAs).
  const [hfAuthStatus, setHfAuthStatus] = useState<HfAuthStatus>('not_authenticated')
  const [hfAuthPolling, setHfAuthPolling] = useState(false)

  // One-time check when enabled becomes true
  useEffect(() => {
    if (!enabled) return
    const checkAuth = async () => {
      const result = await api.getHuggingFaceAuthStatus()
      if (!result.ok) {
        logger.error(`HF auth status check failed: ${result.error.message}`)
        return
      }
      setHfAuthStatus(result.data.status)
    }
    void checkAuth()
  }, [enabled, api])

  // Poll while waiting for user to complete auth in browser
  useEffect(() => {
    if (!hfAuthPolling) return
    const interval = setInterval(async () => {
      const result = await api.getHuggingFaceAuthStatus()
      if (!result.ok) {
        logger.error(`HF auth status check failed: ${result.error.message}`)
        return
      }
      const { status } = result.data
      setHfAuthStatus(status)
      if (status === 'authenticated') setHfAuthPolling(false)
    }, 2000)
    return () => clearInterval(interval)
  }, [hfAuthPolling, api])

  const startHuggingFaceLogin = useCallback(async () => {
    const result = await api.startHuggingFaceLogin()
    if (!result.ok) {
      logger.error(`HF login failed: ${result.error.message}`)
      return
    }

    const params = result.data
    setHfAuthPolling(true)
    const handoff = openAuth === undefined ? desktopHuggingFaceAuthHandoff : openAuth
    if (handoff == null) {
      logger.error('HF login has no host handoff on this runtime')
      setHfAuthPolling(false)
      return
    }
    await handoff({
      clientId: params.client_id,
      redirectUri: params.redirect_uri,
      scope: params.scope,
      state: params.state,
      codeChallenge: params.code_challenge,
      codeChallengeMethod: params.code_challenge_method,
    })
  }, [api, openAuth])

  const handleHuggingFaceLogout = useCallback(async () => {
    const result = await api.huggingFaceLogout()
    if (!result.ok) {
      logger.error(`HF logout failed: ${result.error.message}`)
      return
    }
    setHfAuthStatus('not_authenticated')
  }, [api])

  if (!enabled) {
    return {
      hfAuthStatus: 'authenticated',
      hfAuthPolling: false,
      startHuggingFaceLogin: NOOP,
      handleHuggingFaceLogout: NOOP,
    }
  }

  return {
    hfAuthStatus,
    hfAuthPolling,
    startHuggingFaceLogin,
    handleHuggingFaceLogout,
  }
}
