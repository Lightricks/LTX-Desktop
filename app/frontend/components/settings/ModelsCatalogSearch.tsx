import { Search } from 'lucide-react'

import styles from './SettingsPageLayout.module.scss'

export function ModelsCatalogSearch({
  value,
  onChange,
}: {
  value: string
  onChange: (value: string) => void
}) {
  return (
    <div className={styles.modalCatalogSearch}>
      <Search className={styles.modalCatalogSearchIcon} aria-hidden />
      <input
        id="models-catalog-search"
        className={styles.modalCatalogSearchInput}
        type="text"
        placeholder="Search models"
        value={value}
        onChange={(event) => onChange(event.target.value)}
        autoComplete="off"
        aria-label="Search models"
      />
    </div>
  )
}
