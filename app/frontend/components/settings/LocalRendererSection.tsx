import { useEffect, useState } from 'react'
import { useAppSettings, type AppSettings } from '../../contexts/AppSettingsContext'
import { useVideoGenerationModelSpecs } from '../../hooks/use-video-generation-model-specs'
import { ApiClient } from '../../lib/api-client'
import { H3_VARIANTS } from '../../lib/h3-models'
import { localModelChoices, matchesModelSearch, modelSetupLabels, type ModelSetupKind } from '../../lib/local-model-catalog'
import { ModelSetupPanel } from './ModelSetupPanel'
import { SettingsFieldRow, SettingsHint, SettingsSelect, SettingsStack, SettingsSubsection } from './SettingsContent'
import styles from './ModelSetupPanel.module.scss'

type SearchProps = { query: string; onMatches: (matches: boolean) => void }

export function UnavailableH3Section({ query, onMatches }: SearchProps) {
  const models = H3_VARIANTS.filter(kind => matchesModelSearch(kind, query))
  const hasMatches = models.length > 0
  useEffect(() => { onMatches(hasMatches) }, [onMatches, hasMatches])
  if (!hasMatches) return null
  return <SettingsSubsection showDivider title="MiniMax H3"
    description="Local H3 rendering requires a supported Apple silicon Mac running native macOS, a Windows PC with supported Radeon graphics, or an NVIDIA CUDA GPU on Windows or Linux.">
    <ul className={styles.catalogList}>{models.map(kind => <li key={kind}>{modelSetupLabels[kind]}</li>)}</ul>
    <SettingsHint>The local H3 engine is unavailable in this installation.</SettingsHint>
  </SettingsSubsection>
}

export function LocalRendererSection({ query, onMatches }: SearchProps) {
  const { settings, isLoaded, saveAmdFastEngine, notifyModelsChanged } = useAppSettings()
  const { modelSpecs, isLoading, errorMessage } = useVideoGenerationModelSpecs()
  const [selectedModel, setModel] = useState<ModelSetupKind>()
  const choices = modelSpecs ? localModelChoices(modelSpecs) : []
  const models = choices.filter(kind => matchesModelSearch(kind, query))
  const defaultModel = settings.amdFastEngine === 'comfy' && choices.includes('comfy-int8') ? 'comfy-int8' : 'ltx-2.5'
  const model = selectedModel && models.includes(selectedModel) ? selectedModel
    : models.includes(defaultModel) ? defaultModel : models[0]
  const hasMatches = isLoading || !!errorMessage || models.length > 0
  useEffect(() => { onMatches(hasMatches) }, [onMatches, hasMatches])
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string>()

  async function save(engine: AppSettings['amdFastEngine']) {
    setSaving(true)
    setError(undefined)
    try {
      await saveAmdFastEngine(engine)
      setModel(engine === 'native' ? 'ltx-2.5' : 'comfy-int8')
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setSaving(false)
    }
  }

  async function useManagedModel() {
    try {
      if (choices.includes('comfy-int8')) await saveAmdFastEngine(model === 'comfy-int8' ? 'comfy' : 'native')
      notifyModelsChanged()
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    }
  }

  if (!hasMatches) return null
  return <SettingsSubsection showDivider title="Local video renderer" alert={error || errorMessage}
    description="Manage native LTX and MiniMax H3 models for local video generation.">
    <SettingsStack>
      {isLoading && <SettingsHint>Checking available models…</SettingsHint>}
      {choices.includes('comfy-int8') && !query && <SettingsFieldRow className={styles.rendererField}>
        <label htmlFor="local-default-renderer">Default renderer</label>
        <SettingsSelect id="local-default-renderer" value={settings.amdFastEngine} disabled={!isLoaded || saving}
          onChange={(e) => void save(e.target.value === 'native' ? 'native' : 'comfy')}>
          <option value="comfy">ComfyUI INT8</option>
          <option value="native">Native LTX</option>
        </SettingsSelect>
      </SettingsFieldRow>}
      {model && <>
        <SettingsFieldRow className={styles.rendererField}>
          <label htmlFor="local-setup-model">Manage model</label>
          <SettingsSelect id="local-setup-model" value={model} onChange={(e) => setModel(e.target.value as ModelSetupKind)}>
            {models.map(kind => <option key={kind} value={kind}>{modelSetupLabels[kind]}</option>)}
          </SettingsSelect>
        </SettingsFieldRow>
        <ModelSetupPanel kind={model} api={ApiClient} onReady={model === 'comfy-int8' || model.startsWith('ltx-') ? () => void useManagedModel() : undefined} />
      </>}
    </SettingsStack>
  </SettingsSubsection>
}
