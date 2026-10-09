import { useState } from 'react'

import { BaseModelSection } from './BaseModelSection'
import { ImageGenModelSection } from './ImageGenModelSection'
import { LoraSettingsSection } from './LoraSettingsSection'
import { ModelsCatalogSearch } from './ModelsCatalogSearch'
import { ModelsCatalogSearchEmpty } from './ModelsCatalogSearchEmpty'
import { EncodingEnhanceModelsSection } from './EncodingEnhanceModelsSection'
import { ModelsManagementSection } from './ModelsManagementSection'
import { SettingsAnchorSection } from './SettingsAnchorSection'
import styles from './SettingsPageLayout.module.scss'

export function ModelsSettingsSection({
  sectionActive,
}: {
  sectionActive: boolean
}) {
  const [searchQuery, setSearchQuery] = useState('')
  const [baseHasMatches, setBaseHasMatches] = useState(true)
  const [loraHasMatches, setLoraHasMatches] = useState(true)
  const [imageHasMatches, setImageHasMatches] = useState(true)
  const [encodingHasMatches, setEncodingHasMatches] = useState(true)
  const normalizedSearch = searchQuery.trim().toLowerCase()
  const isSearching = normalizedSearch.length > 0
  const anySearchMatches =
    !isSearching ||
    baseHasMatches ||
    loraHasMatches ||
    imageHasMatches ||
    encodingHasMatches

  const handleSearchChange = (value: string) => {
    setSearchQuery(value)
    setBaseHasMatches(true)
    setLoraHasMatches(true)
    setImageHasMatches(true)
    setEncodingHasMatches(true)
  }

  return (
    <SettingsAnchorSection
      title="Models"
      headerTrailing={<ModelsCatalogSearch value={searchQuery} onChange={handleSearchChange} />}
    >
      {!isSearching ? <ModelsManagementSection /> : null}
      <div
        className={styles.modelsCatalogStack}
        hidden={isSearching && !anySearchMatches}
      >
        <BaseModelSection
          searchQuery={normalizedSearch}
          onSearchHasMatches={setBaseHasMatches}
        />
        <LoraSettingsSection
          searchQuery={normalizedSearch}
          onSearchHasMatches={setLoraHasMatches}
        />
        <ImageGenModelSection
          searchQuery={normalizedSearch}
          onSearchHasMatches={setImageHasMatches}
        />
        <EncodingEnhanceModelsSection
          active={sectionActive}
          searchQuery={normalizedSearch}
          onSearchHasMatches={setEncodingHasMatches}
        />
      </div>
      {isSearching && !anySearchMatches ? (
        <div className={styles.modelsSearchEmptyHost}>
          <ModelsCatalogSearchEmpty query={searchQuery.trim()} />
        </div>
      ) : null}
    </SettingsAnchorSection>
  )
}
