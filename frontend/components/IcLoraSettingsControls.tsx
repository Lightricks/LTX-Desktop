import { Fragment } from 'react'
import { ChevronUp } from 'lucide-react'
import { SettingsDropdown } from './SettingsDropdown'
import { Tooltip } from './ui/tooltip'
import type { ApiSuccessOf } from '../lib/api-client'
import type { components } from '../generated/backend-openapi'
import type { ICLoraConditioningType } from './ICLoraPanel'
import type { IcLoraAudioMode } from '../hooks/use-ic-lora'
import type { ResolutionOption } from '../lib/video-resolution'

type CatalogControl = components['schemas']['IcLoraControl']

// Short hover/focus help per IC-LoRA control. The catalog-driven controls are keyed by control id.
const CONTROL_TOOLTIPS: Record<string, string> = {
  condStrength: 'How strongly the conditioning (edges/depth) constrains the result. Higher follows the control signal more closely.',
  loraStrength: 'How strongly the IC-LoRA weights are applied. Lower is subtler, higher is stronger.',
  duration: 'Length of the generated clip.',
}

// A catalog control's display label for a given option: explicit value_labels win, else value + unit.
function controlOptionLabel(control: CatalogControl, option: number | string): string {
  return control.value_labels?.[String(option)] ?? `${option}${control.unit ?? ''}`
}

// The IC-LoRA control row in the prompt bar: conditioning-type selector + per-mode knobs
// (strength, duration, LoRA strength, and the advanced stage-2 / resolution / audio / fps).
export interface IcLoraControlsProps {
  icLoraCondType?: ICLoraConditioningType
  icLoraSelectorValue?: string
  icLoraSelectorOptions?: { value: string; label: string }[]
  onIcLoraSelectorChange?: (value: string) => void
  icLoraStrength?: number
  onIcLoraStrengthChange?: (strength: number) => void
  availableIcLoras?: ApiSuccessOf<'listModels'>['models']
  icLoraCustomRef?: string | null
  onIcLoraCustomRefChange?: (ref: string | null) => void
  icLoraSkipStage2?: boolean
  onIcLoraSkipStage2Change?: (skip: boolean) => void
  icLoraUseLoraInStage2?: boolean
  onIcLoraUseLoraInStage2Change?: (use: boolean) => void
  // Resolution tiers for the use_lora_in_stage_2 path (empty until source media loads).
  icLoraResolutionOptions?: ResolutionOption[]
  icLoraResolutionKey?: string
  onIcLoraResolutionKeyChange?: (key: string) => void
  icLoraResolutionFactor?: number
  onIcLoraResolutionFactorChange?: (factor: number) => void
  icLoraAudioMode?: IcLoraAudioMode
  onIcLoraAudioModeChange?: (mode: IcLoraAudioMode) => void
  icLoraLoraStrength?: number
  onIcLoraLoraStrengthChange?: (strength: number) => void
  icLoraFps?: number | null
  onIcLoraFpsChange?: (fps: number | null) => void
  isCatalogIcLora?: boolean
  advancedIcLoraControls?: boolean
  // Catalog-declared controls for the selected IC-LoRA, rendered generically by id + kind.
  controls?: CatalogControl[]
  controlValues?: Record<string, number | string>
  onControlChange?: (id: string, value: number | string) => void
}

export function IcLoraSettingsControls({
  icLoraCondType,
  icLoraSelectorValue,
  icLoraSelectorOptions,
  onIcLoraSelectorChange,
  icLoraStrength,
  onIcLoraStrengthChange,
  availableIcLoras,
  icLoraCustomRef,
  onIcLoraCustomRefChange,
  icLoraLoraStrength,
  onIcLoraLoraStrengthChange,
  isCatalogIcLora,
  controls,
  controlValues,
  onControlChange,
}: IcLoraControlsProps) {
  return (
    <>
            {/* Unified IC-LoRA selector: canny | depth | [catalog IC-LoRAs] | custom. No tooltip:
                the options are self-explanatory and the copy was wrong for catalog recipes. */}
            <SettingsDropdown
              title="CONDITIONING TYPE"
              value={icLoraSelectorValue ?? 'canny'}
              onChange={(v) => onIcLoraSelectorChange?.(v)}
              options={icLoraSelectorOptions ?? []}
              trigger={
                <>
                  <span className="text-zinc-300 font-medium">{(icLoraSelectorOptions ?? []).find(o => o.value === icLoraSelectorValue)?.label ?? 'Canny Edges'}</span>
                  <ChevronUp className="h-3 w-3 text-zinc-500" />
                </>
              }
            />
            {!isCatalogIcLora && icLoraCondType === 'custom' && (
              <>
                <div className="w-px h-4 bg-zinc-700 mx-0.5" />
                <SettingsDropdown
                  title="IC-LORA"
                  value={icLoraCustomRef ?? ''}
                  onChange={(v) => onIcLoraCustomRefChange?.(v || null)}
                  options={(availableIcLoras ?? []).map(m => ({ value: m.path, label: m.name }))}
                  trigger={
                    <>
                      <span className="text-zinc-300 font-medium truncate max-w-[160px]">
                        {availableIcLoras?.find(m => m.path === icLoraCustomRef)?.name
                          ?? (availableIcLoras?.length ? 'Select IC-LoRA' : 'No IC-LoRAs found')}
                      </span>
                      <ChevronUp className="h-3 w-3 text-zinc-500" />
                    </>
                  }
                />
              </>
            )}
            {!isCatalogIcLora && (
              <>
                <div className="w-px h-4 bg-zinc-700 mx-0.5" />
                <SettingsDropdown
                  title="STRENGTH"
                  tooltip={CONTROL_TOOLTIPS.condStrength}
                  value={String(icLoraStrength ?? 1.0)}
                  onChange={(v) => onIcLoraStrengthChange?.(parseFloat(v))}
                  options={[
                    { value: '0.5', label: '0.50' },
                    { value: '0.75', label: '0.75' },
                    { value: '1', label: '1.00' },
                    { value: '1.25', label: '1.25' },
                    { value: '1.5', label: '1.50' },
                    { value: '2', label: '2.00' },
                  ]}
                  trigger={
                    <>
                      <span className="text-zinc-500 text-[10px]">STR</span>
                      <span className="text-zinc-300 font-medium">{(icLoraStrength ?? 1.0).toFixed(2)}</span>
                      <ChevronUp className="h-3 w-3 text-zinc-500" />
                    </>
                  }
                />
              </>
            )}
            {/* Catalog-declared option controls (duration, …) rendered generically as dropdowns.
                Structured controls (position_canvas) carry no options and render elsewhere — skip them. */}
            {isCatalogIcLora && (controls ?? []).filter((c) => c.options != null).map((c) => {
              const current = controlValues?.[c.id] ?? c.default ?? ''
              const title = c.label.toUpperCase()
              return (
                <Fragment key={c.id}>
                  <div className="w-px h-4 bg-zinc-700 mx-0.5" />
                  <SettingsDropdown
                    title={title}
                    tooltip={CONTROL_TOOLTIPS[c.id]}
                    value={String(current)}
                    onChange={(v) => onControlChange?.(c.id, c.kind === 'int' ? parseInt(v, 10) : v)}
                    options={(c.options as (number | string)[]).map((o) => ({ value: String(o), label: controlOptionLabel(c, o) }))}
                    trigger={
                      <>
                        <span className="text-zinc-500 text-[10px]">{title}</span>
                        <span className="text-zinc-300 font-medium">{controlOptionLabel(c, current)}</span>
                        <ChevronUp className="h-3 w-3 text-zinc-500" />
                      </>
                    }
                  />
                </Fragment>
              )
            })}
            {/* LoRA strength: a regular control for every IC-LoRA mode (canny/depth/custom/catalog).
                It scales the adapter merge weight, so it always has an effect. (The advanced knobs —
                S2/Res/Audio/FPS — live in IcLoraAdvancedPanel beside the prompt bar.) A continuous
                slider (like the plain-LoRA strength bar) rather than a dropdown, for finer control. */}
            <div className="w-px h-4 bg-zinc-700 mx-0.5" />
            <Tooltip content={CONTROL_TOOLTIPS.loraStrength}>
              <div className="flex items-center gap-1.5 px-2 py-1.5">
                <span className="text-zinc-500 text-[10px]">LORA</span>
                <input
                  type="range"
                  min={0}
                  max={2}
                  step={0.05}
                  value={icLoraLoraStrength ?? 1.0}
                  onChange={(e) => onIcLoraLoraStrengthChange?.(parseFloat(e.target.value))}
                  className="w-16 h-1 accent-white cursor-pointer"
                  aria-label="LoRA strength"
                />
                <span className="text-zinc-300 font-medium w-9 text-right">{(icLoraLoraStrength ?? 1.0).toFixed(2)}</span>
              </div>
            </Tooltip>
    </>
  )
}
