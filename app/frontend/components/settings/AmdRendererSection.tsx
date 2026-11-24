import { useState } from 'react'

import { useAppSettings, type AppSettings } from '../../contexts/AppSettingsContext'
import { ApiClient } from '../../lib/api-client'
import { ModelSetupPanel, modelSetupLabels, type ModelSetupKind } from './ModelSetupPanel'
import { SettingsFieldRow, SettingsSelect, SettingsStack, SettingsSubsection } from './SettingsContent'
import styles from './ModelSetupPanel.module.scss'

const models = Object.entries(modelSetupLabels) as [ModelSetupKind, string][]

export function AmdRendererSection() {
  const { settings, isLoaded, saveAmdFastEngine, notifyModelsChanged } = useAppSettings()
  const [model, setModel] = useState<ModelSetupKind>(() => settings.amdFastEngine === 'native' ? 'ltx-2.5' : 'comfy-int8')
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
      await saveAmdFastEngine(model === 'comfy-int8' ? 'comfy' : 'native')
      notifyModelsChanged()
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    }
  }

  return <SettingsSubsection showDivider title="Radeon video renderer" alert={error}
    description="Choose the default LTX 2.5 Fast renderer or manage a local video model.">
    <SettingsStack>
      <SettingsFieldRow className={styles.rendererField}>
        <label htmlFor="amd-default-renderer">Default renderer</label>
        <SettingsSelect id="amd-default-renderer" value={settings.amdFastEngine} disabled={!isLoaded || saving}
          onChange={(e) => void save(e.target.value === 'native' ? 'native' : 'comfy')}>
          <option value="comfy">ComfyUI INT8</option>
          <option value="native">Native BF16</option>
        </SettingsSelect>
      </SettingsFieldRow>
      <SettingsFieldRow className={styles.rendererField}>
        <label htmlFor="amd-setup-model">Manage model</label>
        <SettingsSelect id="amd-setup-model" value={model} onChange={(e) => setModel(e.target.value as ModelSetupKind)}>
          {models.map(([kind, label]) => <option key={kind} value={kind}>{label}</option>)}
        </SettingsSelect>
      </SettingsFieldRow>
      <ModelSetupPanel kind={model} api={ApiClient} onReady={model === 'comfy-int8' || model.startsWith('ltx-') ? () => void useManagedModel() : undefined} />
    </SettingsStack>
  </SettingsSubsection>
}
