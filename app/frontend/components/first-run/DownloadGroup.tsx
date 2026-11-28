import { useState, type ReactNode } from 'react'

import ArrowSmall from '@ds/assets/Icons/Arrow/Down/Small.svg?react'
import { Text } from '@ds/Text/Text'

import { Checkbox } from '../../ltx-io/components/Checkbox/Checkbox'
import { Info } from '../../ltx-io/components/shared/Info/Info'
import styles from '../FirstRunSetup.module.scss'

interface DownloadGroupProps {
  title: string
  meta: string
  checked: boolean
  indeterminate?: boolean
  checkboxDisabled?: boolean
  onToggle?: () => void
  infoContent?: string
  children: ReactNode
}

export function DownloadGroup({
  title,
  meta,
  checked,
  indeterminate,
  checkboxDisabled,
  onToggle,
  infoContent,
  children,
}: DownloadGroupProps) {
  const [open, setOpen] = useState(false)
  return (
    <div className={styles.downloadGroup}>
      <div className={styles.downloadRow}>
        <span className={styles.downloadName}>
          <Checkbox
            checked={checked}
            indeterminate={indeterminate}
            disabled={checkboxDisabled}
            appearance="neutral"
            onChange={() => onToggle?.()}
            aria-label={title}
          />
          <button
            type="button"
            className={styles.groupTrigger}
            aria-expanded={open}
            onClick={() => setOpen((value) => !value)}
          >
            <Text as="span" variant="body" size="md">
              {title}
            </Text>
            {infoContent ? (
              <span
                className={styles.groupInfo}
                onClick={(event) => event.stopPropagation()}
                onKeyDown={(event) => event.stopPropagation()}
              >
                <Info content={infoContent} side="top" maxWidth={280} />
              </span>
            ) : null}
            <span className={styles.groupChevron} aria-hidden>
              <ArrowSmall />
            </span>
          </button>
        </span>
        <Text as="span" variant="body" size="md" className={styles.downloadMeta}>
          {meta}
        </Text>
      </div>
      {open && <div className={styles.groupItems}>{children}</div>}
    </div>
  )
}
