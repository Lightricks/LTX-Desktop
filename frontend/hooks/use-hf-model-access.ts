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
  /** Set when the access check itself failed — distinct from "not authorized". */
  checkError: string | null
  recheckAccess: () => void
}

export function useHfModelAccess(modelTypes: readonly ModelCheckpointID[], hfAuthStatus: HfAuthStatus): UseHfModelAccessResult {
  const [accessMap, setAccessMap] = useState<ModelAccessMap>({})
  const [checking, setChecking] = useState(false)
  const [polling, setPolling] = useState(false)
  const [checkError, setCheckError] = useState<string | null>(null)

  // Nothing to authorize, or every returned repo is authorized. Empty map with pending
  // checkpoints is NOT authorized — that covers both "still checking" and a failed check
  // (which leaves the map empty and sets checkError for the gate to show).
  const allAuthorized = modelTypes.length === 0
    || (Object.keys(accessMap).length > 0
      && Object.values(accessMap).every((status) => status === 'authorized'))

  const doCheck = useCallback(async () => {
    if (modelTypes.length === 0) return
    setChecking(true)
    const result = await ApiClient.checkModelAccess({ cp_ids: [...modelTypes] })
    if (!result.ok) {
      logger.error(`Model access check failed: ${result.error.message}`)
      setAccessMap({})
      setCheckError(result.error.message)
      setChecking(false)
      return
    }

    const { access } = result.data
    setAccessMap(access)
    setCheckError(null)
    const allOk = Object.values(access).every((s) => s === 'authorized')
    if (allOk) setPolling(false)
    setChecking(false)
  }, [modelTypes])

  // Signed out still needs a check: gated repos (LTX 2.5) can't be downloaded without a token.
  useEffect(() => {
    if (modelTypes.length === 0) {
      setAccessMap((current) => (Object.keys(current).length === 0 ? current : {}))
      setCheckError(null)
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

  return { accessMap, allAuthorized, checking, checkError, recheckAccess }
}
