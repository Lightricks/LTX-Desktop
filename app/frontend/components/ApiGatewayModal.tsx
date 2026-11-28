import { useEffect, useMemo, useState } from 'react'
import { KeyRound, X, Zap } from 'lucide-react'
import { ApiKeyHelperRow, LtxApiKeyInput } from './LtxApiKeyInput'

export type ApiKeyType = 'ltx' | 'fal'

export interface ApiGatewaySection {
  keyType: ApiKeyType
  title: string
  description: string
  required: boolean
  isConfigured: boolean
  inputLabel: string
  placeholder?: string
  onSave: (apiKey: string) => Promise<void> | void
  onGetKey?: () => void
  getKeyLabel?: string
}

export interface ApiGatewayModalProps {
  isOpen: boolean
  onClose: () => void
  title: string
  description: string
  sections: ApiGatewaySection[]
  blocking?: boolean
}

const KEY_TYPE_META: Record<ApiKeyType, { icon: typeof Zap; iconClass: string; chipClass: string }> = {
  ltx: {
    icon: Zap,
    iconClass: 'text-blue-400',
    chipClass: 'bg-amber-500/10 text-amber-300',
  },
  fal: {
    icon: KeyRound,
    iconClass: 'text-cyan-400',
    chipClass: 'bg-zinc-800 text-zinc-400',
  },
}

export function ApiGatewayModal({
  isOpen,
  onClose,
  title,
  description,
  sections,
  blocking = false,
}: ApiGatewayModalProps) {
  const [values, setValues] = useState<Record<ApiKeyType, string>>({ ltx: '', fal: '' })
  const [isSaving, setIsSaving] = useState<Record<ApiKeyType, boolean>>({ ltx: false, fal: false })
  const [errors, setErrors] = useState<Record<ApiKeyType, string | null>>({ ltx: null, fal: null })

  useEffect(() => {
    if (!isOpen) return
    setValues({ ltx: '', fal: '' })
    setIsSaving({ ltx: false, fal: false })
    setErrors({ ltx: null, fal: null })
  }, [isOpen])

  const allRequiredConfigured = useMemo(() => {
    const requiredSections = sections.filter((section) => section.required)
    if (requiredSections.length === 0) return true
    return requiredSections.every((section) => section.isConfigured)
  }, [sections])

  useEffect(() => {
    if (!isOpen) return
    if (!allRequiredConfigured) return
    if (sections.length <= 1) {
      onClose()
    }
  }, [isOpen, onClose, sections.length, allRequiredConfigured])

  useEffect(() => {
    if (!isOpen) return
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && !blocking) {
        onClose()
      }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [blocking, isOpen, onClose])

  const handleSave = async (section: ApiGatewaySection) => {
    const keyType = section.keyType
    const trimmedKey = (values[keyType] ?? '').trim()
    if (!trimmedKey) {
      setErrors((prev) => ({ ...prev, [keyType]: `Please enter a valid ${section.inputLabel}.` }))
      return
    }

    setIsSaving((prev) => ({ ...prev, [keyType]: true }))
    setErrors((prev) => ({ ...prev, [keyType]: null }))
    try {
      await section.onSave(trimmedKey)
      setValues((prev) => ({ ...prev, [keyType]: '' }))
    } catch (err) {
      if (err instanceof Error && err.message.trim()) {
        setErrors((prev) => ({ ...prev, [keyType]: err.message }))
      } else {
        setErrors((prev) => ({ ...prev, [keyType]: 'Failed to save API key.' }))
      }
    } finally {
      setIsSaving((prev) => ({ ...prev, [keyType]: false }))
    }
  }

  const requiredMissing = useMemo(() => sections.some((section) => section.required && !section.isConfigured), [sections])

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4">
      <div className="w-full max-w-[620px] rounded-xl border border-zinc-700 bg-zinc-900 shadow-2xl overflow-hidden">
        <div className="flex items-center justify-between border-b border-zinc-800 px-5 py-4">
          <div className="flex items-center gap-2.5">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-blue-500/15 text-blue-300">
              <KeyRound className="h-4 w-4" />
            </div>
            <h2 className="text-base font-semibold text-zinc-100">{title}</h2>
          </div>
          {!blocking && (
            <button
              onClick={onClose}
              className="rounded-lg p-1.5 text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-300"
              aria-label="Close API gateway modal"
            >
              <X className="h-4 w-4" />
            </button>
          )}
        </div>

        <div className="space-y-5 px-5 py-5">
          <p className="text-sm leading-relaxed text-zinc-300">{description}</p>

          <div className="space-y-4">
            {sections.map((section) => {
              const meta = KEY_TYPE_META[section.keyType]
              const Icon = meta.icon
              const configured = section.isConfigured
              const saving = isSaving[section.keyType]
              const value = values[section.keyType] ?? ''
              const error = errors[section.keyType]
              const canSubmit = value.trim().length > 0 && !saving

              return (
                <div key={section.keyType} className="rounded-lg border border-zinc-800 bg-zinc-900/60 p-4 space-y-3">
                  <div className="flex items-start justify-between gap-3">
                    <div className="flex-1">
                      <div className="flex items-center gap-2">
                        <Icon className={`h-4 w-4 ${meta.iconClass}`} />
                        <h3 className="text-sm font-semibold text-white">{section.title}</h3>
                        <span className={`text-[10px] px-1.5 py-0.5 rounded ${section.required ? meta.chipClass : 'bg-zinc-800 text-zinc-500'}`}>
                          {section.required ? 'Required' : 'Optional'}
                        </span>
                      </div>
                      <p className="text-xs text-zinc-500 mt-1">{section.description}</p>
                    </div>
                    <div className={`text-xs px-2 py-1 rounded inline-flex items-center gap-1.5 ${configured ? 'bg-green-500/10 text-green-400' : 'bg-zinc-800 text-zinc-500'}`}>
                      {configured ? 'Configured' : 'Not set'}
                    </div>
                  </div>

                  {!configured && (
                    <div className="space-y-2">
                      <label className="block text-xs text-zinc-300">{section.inputLabel}</label>
                      <div className="flex gap-2">
                        <LtxApiKeyInput
                          value={value}
                          onChange={(event) => setValues((prev) => ({ ...prev, [section.keyType]: event.target.value }))}
                          placeholder={section.placeholder ?? 'Paste your API key'}
                          className="flex-1"
                        />
                        <button
                          onClick={() => handleSave(section)}
                          disabled={!canSubmit}
                          className="px-3 py-2 bg-blue-600 text-white text-sm rounded-lg hover:bg-blue-500 disabled:bg-zinc-700 disabled:text-zinc-500 disabled:cursor-not-allowed transition-colors whitespace-nowrap"
                        >
                          {saving ? 'Saving...' : 'Save Key'}
                        </button>
                      </div>
                      <ApiKeyHelperRow
                        label={section.getKeyLabel ?? 'Get API key'}
                        onOpenKey={section.onGetKey}
                      />
                    </div>
                  )}

                  {error && (
                    <div className="rounded-lg border border-red-500/30 bg-red-500/10 px-3 py-2 text-sm text-red-300">
                      {error}
                    </div>
                  )}
                </div>
              )
            })}
          </div>

          {blocking && requiredMissing && (
            <div className="rounded-lg border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-xs text-amber-200">
              Required API keys are missing. Add them to continue.
            </div>
          )}

          {sections.length > 1 && allRequiredConfigured && (
            <div className="flex justify-end">
              <button
                onClick={onClose}
                className="px-4 py-2 bg-blue-600 text-white text-sm rounded-lg hover:bg-blue-500 transition-colors"
              >
                Done
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
