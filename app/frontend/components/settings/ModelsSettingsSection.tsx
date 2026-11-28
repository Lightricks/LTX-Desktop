import { useState } from 'react'

import { useAppSettings } from '../../contexts/AppSettingsContext'
import { LocalRendererSection, UnavailableH3Section } from './LocalRendererSection'
import { BaseModelSection } from './BaseModelSection'
import { ImageGenModelSection } from './ImageGenModelSection'
import { LoraSettingsSection } from './LoraSettingsSection'
import { ModelsCatalogSearch } from './ModelsCatalogSearch'
import { ModelsCatalogSearchEmpty } from './ModelsCatalogSearchEmpty'
import { EncodingEnhanceModelsSection } from './EncodingEnhanceModelsSection'
import { AssistantModelsSection } from './AssistantModelsSection'
import { ModelsManagementSection } from './ModelsManagementSection'
import { SettingsAnchorSection } from './SettingsAnchorSection'
import styles from './SettingsPageLayout.module.scss'

export function ModelsSettingsSection({
  sectionActive,
}: {
  sectionActive: boolean
}) {
  const { managedRendererAvailable, forceApiGenerations, runtimePolicyLoaded } = useAppSettings()
  const [searchQuery, setSearchQuery] = useState('')
  const [rendererMatches, setRendererMatches] = useState(true)
  const [baseHasMatches, setBaseHasMatches] = useState(true)
  const [loraHasMatches, setLoraHasMatches] = useState(true)
  const [imageHasMatches, setImageHasMatches] = useState(true)
  const [encodingHasMatches, setEncodingHasMatches] = useState(true)
  const [assistantHasMatches, setAssistantHasMatches] = useState(true)
  const normalizedSearch = searchQuery.trim().toLowerCase()
  const isSearching = normalizedSearch.length > 0
  const catalogMatches = baseHasMatches || loraHasMatches || imageHasMatches || encodingHasMatches || assistantHasMatches
  const anySearchMatches = !isSearching || rendererMatches || (!forceApiGenerations && catalogMatches)

  const handleSearchChange = (value: string) => {
    setSearchQuery(value)
    setBaseHasMatches(true)
    setLoraHasMatches(true)
    setImageHasMatches(true)
    setEncodingHasMatches(true)
    setAssistantHasMatches(true)
  }

  return (
    <SettingsAnchorSection
      title="Models"
      headerTrailing={<ModelsCatalogSearch value={searchQuery} onChange={handleSearchChange} />}
    >
      {runtimePolicyLoaded && !forceApiGenerations && !isSearching ? <ModelsManagementSection /> : null}
      <div
        className={styles.modelsCatalogStack}
        hidden={isSearching && !anySearchMatches}
      >
        {sectionActive && runtimePolicyLoaded && (managedRendererAvailable && !forceApiGenerations
          ? <LocalRendererSection query={normalizedSearch} onMatches={setRendererMatches} />
          : <UnavailableH3Section query={normalizedSearch} onMatches={setRendererMatches} />)}
        {runtimePolicyLoaded && !forceApiGenerations && <>
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
          <AssistantModelsSection
            active={sectionActive}
            searchQuery={normalizedSearch}
            onSearchHasMatches={setAssistantHasMatches}
          />
        </>}
      </div>
      {isSearching && !anySearchMatches ? (
        <div className={styles.modelsSearchEmptyHost}>
          <ModelsCatalogSearchEmpty query={searchQuery.trim()} />
        </div>
      ) : null}
    </SettingsAnchorSection>
  )
}
