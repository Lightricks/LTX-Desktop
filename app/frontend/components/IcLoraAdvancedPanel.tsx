import type { ReactNode } from 'react'
import { ChevronUp } from 'lucide-react'
import { SettingsDropdown } from './SettingsDropdown'
import type { IcLoraControlsProps } from './IcLoraSettingsControls'
import type { IcLoraAudioMode } from '../hooks/use-ic-lora'

// One labelled row: caption on the left, the dropdown on the right.
function Row({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-2">
      <span className="text-[10px] text-fg-tertiary uppercase tracking-wide shrink-0">{label}</span>
      {children}
    </div>
  )
}

const triggerValue = (text: string) => (
  <>
    <span className="text-fg-secondary font-medium">{text}</span>
    <ChevronUp className="h-3 w-3 text-fg-tertiary" />
  </>
)

// Advanced IC-LoRA controls, in a side panel beside the prompt bar (so they don't push the
// Generate button off the bottom row). Reuses the shared IcLoraControlsProps bag.
export function IcLoraAdvancedPanel({
  icLoraSkipStage2,
  onIcLoraSkipStage2Change,
  icLoraResolutionFactor,
  onIcLoraResolutionFactorChange,
  icLoraAudioMode,
  onIcLoraAudioModeChange,
  icLoraFps,
  onIcLoraFpsChange,
  advancedIcLoraControls,
}: IcLoraControlsProps) {
  // Advanced-only knobs. (LoRA strength is a regular control — it lives in the bottom bar.)
  if (!advancedIcLoraControls) return null
  return (
    <div className="rounded-lg border border-separator bg-[color-mix(in_srgb,var(--semantic-bg-primary)_90%,transparent)] backdrop-blur p-1.5 flex flex-col gap-0.5 w-40 text-xs">
      <span className="text-[9px] font-semibold text-fg-tertiary uppercase tracking-wider px-1">Advanced</span>

      <Row label="Stage 2">
            <SettingsDropdown
              title="STAGE 2 REFINE"
              value={icLoraSkipStage2 ? 'skip' : 'keep'}
              onChange={(v) => onIcLoraSkipStage2Change?.(v === 'skip')}
              options={[
                { value: 'keep', label: 'On (two-stage)' },
                { value: 'skip', label: 'Off (stage 1 only)' },
              ]}
              trigger={triggerValue(icLoraSkipStage2 ? 'Off' : 'On')}
            />
          </Row>
          {icLoraSkipStage2 && (
            <Row label="Res">
              <SettingsDropdown
                title="RES FACTOR"
                value={String(icLoraResolutionFactor ?? 2.0)}
                onChange={(v) => onIcLoraResolutionFactorChange?.(parseFloat(v))}
                options={[
                  // 0 is the "source dimensions" sentinel — output matches the input
                  // resolution and the multiplier is ignored (see ic_lora_handler).
                  { value: '0', label: 'Source dimensions' },
                  { value: '1', label: '1.00 (half)' },
                  { value: '1.25', label: '1.25' },
                  { value: '1.5', label: '1.50' },
                  { value: '1.75', label: '1.75' },
                  { value: '2', label: '2.00 (native)' },
                ]}
                trigger={triggerValue(
                  icLoraResolutionFactor === 0 ? 'Source' : `×${(icLoraResolutionFactor ?? 2.0).toFixed(2)}`,
                )}
              />
            </Row>
          )}
          <Row label="Audio">
            <SettingsDropdown
              title="AUDIO"
              value={icLoraAudioMode ?? 'generated'}
              onChange={(v) => onIcLoraAudioModeChange?.(v as IcLoraAudioMode)}
              options={[
                { value: 'generated', label: 'Generated' },
                { value: 'source', label: 'Source' },
                { value: 'off', label: 'Off' },
              ]}
              trigger={triggerValue((icLoraAudioMode ?? 'generated').replace(/^\w/, (c: string) => c.toUpperCase()))}
            />
          </Row>
          <Row label="FPS">
            <SettingsDropdown
              title="FPS"
              value={icLoraFps == null ? 'source' : String(icLoraFps)}
              onChange={(v) => onIcLoraFpsChange?.(v === 'source' ? null : parseFloat(v))}
              options={[
                { value: 'source', label: 'Source' },
                { value: '24', label: '24 fps' },
                { value: '16', label: '16 fps' },
                { value: '12', label: '12 fps' },
                { value: '8', label: '8 fps' },
              ]}
              trigger={triggerValue(icLoraFps == null ? 'Src' : String(icLoraFps))}
            />
          </Row>
    </div>
  )
}
