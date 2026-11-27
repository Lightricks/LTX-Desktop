import { useState } from 'react'

import { useAppSettings } from '../../contexts/AppSettingsContext'
import { LocalRendererSection } from './LocalRendererSection'
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
  const { managedRendererAvailable } = useAppSettings()
  const [searchQuery, setSearchQuery] = useState('')
  const [baseHasMatches, setBaseHasMatches] = useState(true)
  const [loraHasMatches, setLoraHasMatches] = useState(true)
  const [imageHasMatches, setImageHasMatches] = useState(true)
  const [encodingHasMatches, setEncodingHasMatches] = useState(true)
  const [assistantHasMatches, setAssistantHasMatches] = useState(true)
  const normalizedSearch = searchQuery.trim().toLowerCase()
  const isSearching = normalizedSearch.length > 0
  const rendererMatches = managedRendererAvailable
    && 'local video renderer nvidia cuda amd radeon rocm ltx 2.3 ltx 2.5 fast comfyui int8 native bf16 kandinsky 6.0 video lite distill minimax h3 fl2va turbo 8-step ref2va reference keyframe'.includes(normalizedSearch)
  const anySearchMatches =
    !isSearching ||
    rendererMatches ||
    baseHasMatches ||
    loraHasMatches ||
    imageHasMatches ||
    encodingHasMatches ||
    assistantHasMatches

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
      {!isSearching ? <ModelsManagementSection /> : null}
      <div
        className={styles.modelsCatalogStack}
        hidden={isSearching && !anySearchMatches}
      >
        {rendererMatches && sectionActive && <LocalRendererSection />}
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
      </div>
      {isSearching && !anySearchMatches ? (
        <div className={styles.modelsSearchEmptyHost}>
          <ModelsCatalogSearchEmpty query={searchQuery.trim()} />
        </div>
      ) : null}
    </SettingsAnchorSection>
  )
}
