import type { ReactNode } from 'react'
import { Text } from '@ds/Text/Text'
import styles from './HfStatusToast.module.scss'

interface HfStatusToastProps {
  /** Primary message rendered as the toast's leading text. */
  message: string
  /** Optional secondary detail (e.g. the underlying check error) under the message. */
  detail?: string | null
  /** Action control(s) rendered trailing, e.g. a Retry button. */
  children?: ReactNode
}

export function HfStatusToast({ message, detail = null, children }: HfStatusToastProps) {
  return (
    <div className={styles.toast}>
      <div className={styles.toastText}>
        <Text as="p" variant="body" size="lg" className={styles.toastMessage}>
          {message}
        </Text>
        {detail ? (
          <Text as="p" variant="body" size="sm" className={styles.toastDetail}>
            {detail}
          </Text>
        ) : null}
      </div>
      {children}
    </div>
  )
}
