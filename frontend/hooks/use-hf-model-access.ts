import { useState, useEffect, useCallback } from 'react'
import { ApiClient, type ApiRequestBodyOf, type ApiSuccessOf } from '../lib/api-client'
import { logger } from '../lib/logger'

type HfAuthStatus = ApiSuccessOf<'getHuggingFaceAuthStatus'>['status']
type ModelAccessMap = ApiSuccessOf<'checkModelAccess'>['access']
type CheckModelAccessBody = NonNullable<ApiRequestBodyOf<'checkModelAccess'>>
type ModelCheckpointID = NonNullable<CheckModelAccessBody['cp_ids']>[number]

interface UseHfModelAccessResult {
  accessMap: ModelAccessMap
  allAuthorized: boolean
  checking: boolean
  recheckAccess: () => void
}

const NOOP = () => {}

export function useHfModelAccess(modelTypes: readonly ModelCheckpointID[], hfAuthStatus: HfAuthStatus): UseHfModelAccessResult {
  // Bundled checkpoints are public — when signed out there's nothing to verify (treated as
  // authorized below). The per-repo access check only runs when signed in, covering any future
  // gated checkpoint without forcing sign-in.
  const [accessMap, setAccessMap] = useState<ModelAccessMap>({})
  const [checking, setChecking] = useState(false)
  const [polling, setPolling] = useState(false)

  const allAuthorized = modelTypes.length > 0
    && modelTypes.every((modelType) => accessMap[modelType] === 'authorized')

  const doCheck = useCallback(async () => {
    if (modelTypes.length === 0) return
    setChecking(true)
    const result = await ApiClient.checkModelAccess({ cp_ids: [...modelTypes] })
    if (!result.ok) {
      logger.error(`Model access check failed: ${result.error.message}`)
      setChecking(false)
      return
    }

    const { access } = result.data
    setAccessMap(access)
    const allOk = Object.values(access).every((s) => s === 'authorized')
    if (allOk) setPolling(false)
    setChecking(false)
  }, [modelTypes])

  // Initial check when authenticated
  useEffect(() => {
    if (hfAuthStatus !== 'authenticated' || modelTypes.length === 0) {
      setAccessMap((current) => (Object.keys(current).length === 0 ? current : {}))
      setPolling(false)
      return
    }
    void doCheck()
    setPolling(true)
  }, [hfAuthStatus, modelTypes.length, doCheck])

  // Poll while any model is not_authorized
  useEffect(() => {
    if (!polling || hfAuthStatus !== 'authenticated') return
    const interval = setInterval(() => { void doCheck() }, 5000)
    return () => clearInterval(interval)
  }, [polling, hfAuthStatus, doCheck])

  const recheckAccess = useCallback(() => {
    void doCheck()
  }, [doCheck])

  // Signed out: bundled checkpoints are public, so nothing gates the download.
  if (hfAuthStatus !== 'authenticated') {
    return { accessMap: {}, allAuthorized: true, checking: false, recheckAccess: NOOP }
  }

  return { accessMap, allAuthorized, checking, recheckAccess }
}
