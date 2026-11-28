import { SearchX } from 'lucide-react'

import { Text } from '@/ds/Text/Text'

import styles from './SettingsPageLayout.module.scss'

export function ModelsCatalogSearchEmpty({ query }: { query: string }) {
  return (
    <div className={styles.modelsSearchEmpty} role="status">
      <SearchX className={styles.modelsSearchEmptyIcon} aria-hidden />
      <div className={styles.modelsSearchEmptyCopy}>
        <Text as="p" variant="heading" size="xs" align="center">
          No models match &ldquo;{query}&rdquo;
        </Text>
        <Text as="p" variant="body" size="sm" align="center" className={styles.modelsSearchEmptyHint}>
          Try a different name or clear the search field.
        </Text>
      </div>
    </div>
  )
}
