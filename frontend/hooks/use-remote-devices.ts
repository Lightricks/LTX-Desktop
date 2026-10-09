import { useCallback, useEffect, useState } from 'react'
import { ApiClient, type ApiSuccessOf } from '../lib/api-client'

export type RemoteDevice = ApiSuccessOf<'listRemoteDevices'>[number]

export const REMOTE_DEVICES_POLL_MS = 5000

export function useRemoteDevices(options: {
  active: boolean
  poll: boolean
  intervalMs?: number
}): {
  devices: RemoteDevice[]
  revoke: (deviceId: string) => Promise<boolean>
  revokingId: string | null
} {
  const { active, poll, intervalMs = REMOTE_DEVICES_POLL_MS } = options
  const [devices, setDevices] = useState<RemoteDevice[]>([])
  const [revokingId, setRevokingId] = useState<string | null>(null)

  const refresh = useCallback(async () => {
    const result = await ApiClient.listRemoteDevices()
    if (result.ok) {
      setDevices(result.data)
    }
  }, [])

  useEffect(() => {
    if (!active) {
      setDevices([])
      return
    }
    let cancelled = false
    const pollOnce = async () => {
      try {
        const result = await ApiClient.listRemoteDevices()
        if (!cancelled && result.ok) {
          setDevices(result.data)
        }
      } catch {
        // Keep the last successful list; a transport failure is not "no devices".
      }
    }
    void pollOnce()
    if (!poll) {
      return () => {
        cancelled = true
      }
    }
    const interval = window.setInterval(() => {
      void pollOnce()
    }, intervalMs)
    return () => {
      cancelled = true
      window.clearInterval(interval)
    }
  }, [active, intervalMs, poll])

  const revoke = useCallback(
    async (deviceId: string) => {
      setRevokingId(deviceId)
      try {
        const result = await ApiClient.revokeRemoteDevice(deviceId)
        if (result.ok) {
          await refresh()
          return true
        }
        return false
      } finally {
        setRevokingId(null)
      }
    },
    [refresh],
  )

  return { devices, revoke, revokingId }
}
