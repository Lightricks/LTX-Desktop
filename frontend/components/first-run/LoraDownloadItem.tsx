import { Text } from '@ds/Text/Text'

import { formatBytes } from '@/lib/format'
import type { FirstRunLoraItem } from '@/lib/first-run-downloads'

import { Checkbox } from '../../ltx-io/components/Checkbox/Checkbox'
import styles from '../FirstRunSetup.module.scss'

interface LoraDownloadItemProps {
  item: FirstRunLoraItem
  checked: boolean
  onToggle: () => void
}

export function LoraDownloadItem({ item, checked, onToggle }: LoraDownloadItemProps) {
  const skipped = !item.downloaded && !checked
  return (
    <div className={styles.downloadRow}>
      <span className={styles.downloadName}>
        <Checkbox
          checked={item.downloaded || checked}
          disabled={item.downloaded}
          appearance="neutral"
          onChange={() => onToggle()}
        >
          <Text as="span" variant="body" size="md">
            {item.name}
          </Text>
        </Checkbox>
      </span>
      <Text
        as="span"
        variant="body"
        size="md"
        className={`${styles.downloadMeta} ${item.downloaded ? styles.downloadMetaInstalled : ''}`}
      >
        {item.downloaded ? 'Installed' : skipped ? 'Skipped' : formatBytes(item.sizeBytes)}
      </Text>
    </div>
  )
}
