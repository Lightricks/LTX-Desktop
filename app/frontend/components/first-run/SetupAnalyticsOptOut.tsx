import { useEffect, useRef, useState } from 'react'

import { Text } from '@ds/Text/Text'

import { Checkbox } from '../../ltx-io/components/Checkbox/Checkbox'
import styles from '../FirstRunSetup.module.scss'

import { createAnalyticsOptOutWrites } from './analytics-opt-out-write'

// Unchecked means collection stays on. Checking is an explicit opt-out and
// writes the same flag the Settings privacy toggle reads.
const analyticsOptOutWrites = createAnalyticsOptOutWrites((enabled) =>
  window.electronAPI.setAnalyticsEnabled({ enabled }),
)

export function flushAnalyticsOptOut(): Promise<void> {
  return analyticsOptOutWrites.flush()
}

export function SetupAnalyticsOptOut() {
  const [optedOut, setOptedOut] = useState(false)
  const touched = useRef(false)

  useEffect(() => {
    let cancelled = false
    void window.electronAPI
      .getAnalyticsState()
      .then((state) => {
        if (!cancelled && !touched.current) setOptedOut(!state.analyticsEnabled)
      })
      .catch(() => {})
    return () => {
      cancelled = true
    }
  }, [])

  const onChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const nextOptedOut = event.target.checked
    touched.current = true
    setOptedOut(nextOptedOut)
    analyticsOptOutWrites.enqueue(!nextOptedOut, () => {
      setOptedOut((current) => (current === nextOptedOut ? !nextOptedOut : current))
    })
  }

  return (
    <Checkbox
      checked={optedOut}
      appearance="neutral"
      className={styles.analyticsOptOut}
      onChange={onChange}
    >
      <Text as="span" variant="body" size="md">
        Opt out of basic analytics
      </Text>
    </Checkbox>
  )
}
