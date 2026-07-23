import { useEffect, useMemo } from 'react'
import { Select } from './ui/select'
import {
  areVideoGenerationSettingsEquivalent,
  resolveVideoGenerationOptions,
  sanitizeVideoGenerationSettings,
  type VideoGenerationModelSpecItem,
} from '../lib/video-generation-model-specs'

export type GenerationMode = 'text-to-video' | 'image-to-video' | 'text-to-image'

export interface LoraSelection {
  ref: string
  name: string
  scale: number
  // Catalog id (e.g. "cozy-felt-style"), when this LoRA was picked from the library —
  // absent for a manually-added/custom LoRA not in the catalog. Lets the prompt enhancer
  // look up trigger/instructions without re-matching on filename or display name.
  catalogId?: string
}

export interface GenerationSettings {
  model: 'fast' | 'pro'
  duration: number
  videoResolution: string
  fps: number
  audio: boolean
  cameraMotion: string
  aspectRatio?: string
  loras?: LoraSelection[]
  // Image-specific settings
  imageResolution: string
  imageAspectRatio: string
  imageSteps: number
  variations?: number  // Number of image variations to generate
  imageEditStrength?: number  // Denoising strength when editing an existing image
}

interface SettingsPanelProps {
  settings: GenerationSettings
  onSettingsChange: (settings: GenerationSettings) => void
  disabled?: boolean
  mode?: GenerationMode
  hasAudio?: boolean
  videoModelSpecs?: VideoGenerationModelSpecItem[] | null
  minimumDuration?: number
  hideDuration?: boolean
  videoSettingsMessage?: string | null
}

export function SettingsPanel({
  settings,
  onSettingsChange,
  disabled,
  mode = 'text-to-video',
  hasAudio = false,
  videoModelSpecs,
  minimumDuration,
  hideDuration = false,
  videoSettingsMessage,
}: SettingsPanelProps) {
  const isImageMode = mode === 'text-to-image'
  const resolvedVideoOptions = useMemo(() => {
    if (isImageMode || !videoModelSpecs || videoModelSpecs.length === 0) return null
    return resolveVideoGenerationOptions({
      settings,
      modelSpecs: videoModelSpecs,
      hasAudio,
      minimumDuration,
      durationSelection: hideDuration ? 'smallest_valid' : 'preserve',
    })
  }, [hasAudio, hideDuration, isImageMode, minimumDuration, settings, videoModelSpecs])

  useEffect(() => {
    if (isImageMode || !videoModelSpecs || videoModelSpecs.length === 0) return
    const sanitized = sanitizeVideoGenerationSettings(settings, videoModelSpecs, {
      hasAudio,
      minimumDuration,
      durationSelection: hideDuration ? 'smallest_valid' : 'preserve',
    })
    if (!sanitized) return
    if (!areVideoGenerationSettingsEquivalent(settings, sanitized)) {
      onSettingsChange(sanitized)
    }
  }, [hasAudio, hideDuration, isImageMode, minimumDuration, onSettingsChange, settings, videoModelSpecs])

  const handleChange = (key: keyof GenerationSettings, value: string | number | boolean) => {
    if (isImageMode) {
      onSettingsChange({ ...settings, [key]: value } as GenerationSettings)
      return
    }
    if (!videoModelSpecs || videoModelSpecs.length === 0) return

    const nextSettings = { ...settings, [key]: value } as GenerationSettings
    const sanitized = sanitizeVideoGenerationSettings(nextSettings, videoModelSpecs, {
      hasAudio,
      minimumDuration,
      durationSelection: hideDuration ? 'smallest_valid' : 'preserve',
    })
    if (sanitized) {
      onSettingsChange(sanitized)
    }
  }

  // Image mode settings
  if (isImageMode) {
    return (
      <div className="space-y-4">
        {/* Aspect Ratio and Quality side by side */}
        <div className="grid grid-cols-2 gap-3">
          <Select
            label="Aspect Ratio"
            value={settings.imageAspectRatio || '16:9'}
            onChange={(e) => handleChange('imageAspectRatio', e.target.value)}
            disabled={disabled}
          >
            <option value="1:1">1:1 (Square)</option>
            <option value="16:9">16:9 (Landscape)</option>
            <option value="9:16">9:16 (Portrait)</option>
            <option value="4:3">4:3 (Standard)</option>
            <option value="3:4">3:4 (Portrait Standard)</option>
            <option value="21:9">21:9 (Cinematic)</option>
          </Select>

          <Select
            label="Quality"
            value={settings.imageSteps || 4}
            onChange={(e) => handleChange('imageSteps', parseInt(e.target.value))}
            disabled={disabled}
          >
            <option value={4}>Fast</option>
            <option value={8}>Balanced</option>
            <option value={12}>High</option>
          </Select>
        </div>
      </div>
    )
  }

  if (!videoModelSpecs || videoModelSpecs.length === 0 || !resolvedVideoOptions || !resolvedVideoOptions.hasCompatibleOptions) {
    return (
      <div className="rounded-lg border border-zinc-800 bg-zinc-950/60 px-3 py-2 text-xs text-zinc-500">
        {videoSettingsMessage || 'Loading generation settings...'}
      </div>
    )
  }

  const showFps = resolvedVideoOptions.fpsOptions.length > 1
  const videoControlsColumns = hideDuration
    ? (showFps ? 'grid-cols-2' : 'grid-cols-1')
    : (showFps ? 'grid-cols-3' : 'grid-cols-2')

  // Video mode settings
  return (
    <div className="space-y-4">
      <Select
        label="Model"
        value={resolvedVideoOptions.selectedModel ?? settings.model}
        onChange={(e) => handleChange('model', e.target.value)}
        disabled={disabled}
      >
        {resolvedVideoOptions.modelOptions.map((item) => (
          <option key={item.pipeline} value={item.pipeline}>
            {item.spec.display_name}
          </option>
        ))}
      </Select>

      {/* Duration, Resolution, FPS Row */}
      <div className={`grid gap-3 ${videoControlsColumns}`}>
        {!hideDuration && (
          <Select
            label="Duration"
            value={resolvedVideoOptions.selectedDuration ?? settings.duration}
            onChange={(e) => handleChange('duration', parseInt(e.target.value))}
            disabled={disabled}
          >
            {resolvedVideoOptions.durationOptions.map((duration) => (
              <option key={duration} value={duration}>
                {duration} sec
              </option>
            ))}
          </Select>
        )}

        <Select
          label="Resolution"
          value={resolvedVideoOptions.selectedResolution ?? settings.videoResolution}
          onChange={(e) => handleChange('videoResolution', e.target.value)}
          disabled={disabled}
        >
          {resolvedVideoOptions.resolutionOptions.map((resolution) => (
            <option key={resolution} value={resolution}>
              {resolution}
            </option>
          ))}
        </Select>

        {showFps && (
          <Select
            label="FPS"
            value={resolvedVideoOptions.selectedFps ?? settings.fps}
            onChange={(e) => handleChange('fps', parseInt(e.target.value))}
            disabled={disabled}
          >
            {resolvedVideoOptions.fpsOptions.map((fps) => (
              <option key={fps} value={fps}>
                {fps}
              </option>
            ))}
          </Select>
        )}
      </div>

      {/* Aspect Ratio */}
      <Select
        label="Aspect Ratio"
        value={settings.aspectRatio || '16:9'}
        onChange={(e) => handleChange('aspectRatio', e.target.value)}
        disabled={disabled}
      >
        <option value="16:9">16:9 Landscape</option>
        <option value="9:16">9:16 Portrait</option>
      </Select>

      {/* Audio and Camera Motion Row */}
      <div className="flex gap-3">
        <div className="w-[140px] flex-shrink-0">
          <Select
            label="Audio"
            badge="PREVIEW"
            value={settings.audio ? 'on' : 'off'}
            onChange={(e) => handleChange('audio', e.target.value === 'on')}
            disabled={disabled}
          >
            <option value="on">On</option>
            <option value="off">Off</option>
          </Select>
        </div>

        <div className="flex-1">
          <Select
            label="Camera Motion"
            value={settings.cameraMotion}
            onChange={(e) => handleChange('cameraMotion', e.target.value)}
            disabled={disabled}
          >
            <option value="none">None</option>
            <option value="static">Static</option>
            <option value="focus_shift">Focus Shift</option>
            <option value="dolly_in">Dolly In</option>
            <option value="dolly_out">Dolly Out</option>
            <option value="dolly_left">Dolly Left</option>
            <option value="dolly_right">Dolly Right</option>
            <option value="jib_up">Jib Up</option>
            <option value="jib_down">Jib Down</option>
          </Select>
        </div>
      </div>
    </div>
  )
}
