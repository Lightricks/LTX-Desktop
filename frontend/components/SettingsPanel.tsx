import { Select } from './ui/select'
import type { GenerationMode } from './ModeTabs'
import {
  FORCED_API_VIDEO_FPS,
  FORCED_API_VIDEO_RESOLUTIONS,
  getAllowedForcedApiDurations,
  sanitizeForcedApiVideoSettings,
} from '../lib/api-video-options'

export interface GenerationSettings {
  model: 'fast' | 'pro' | 'seedance-1.5-pro'
  duration: number
  videoResolution: string
  fps: number
  audio: boolean
  cameraMotion: string
  aspectRatio?: string
  // Image-specific settings
  imageResolution: string
  imageAspectRatio: string
  imageSteps: number
  variations?: number  // Number of image variations to generate
  strength?: number  // Edit strength for img2img (0.0-1.0)
  loraPath?: string | null
  loraWeight?: number
  loraTriggerPhrase?: string | null
  loraTriggerMode?: 'prepend' | 'append' | 'off'
}

interface SettingsPanelProps {
  settings: GenerationSettings
  onSettingsChange: (settings: GenerationSettings) => void
  disabled?: boolean
  mode?: GenerationMode
  forceApiGenerations?: boolean
  hasAudio?: boolean
  hasReplicateApiKey?: boolean
}

export function SettingsPanel({
  settings,
  onSettingsChange,
  disabled,
  mode = 'text-to-video',
  forceApiGenerations = false,
  hasAudio = false,
  hasReplicateApiKey = false,
}: SettingsPanelProps) {
  const isImageMode = mode === 'text-to-image'
  const LOCAL_MAX_DURATION: Record<string, number> = { '540p': 20, '720p': 10, '1080p': 5 }

  const handleChange = (key: keyof GenerationSettings, value: string | number | boolean) => {
    const nextSettings = { ...settings, [key]: value } as GenerationSettings
    if (forceApiGenerations && !isImageMode) {
      onSettingsChange(sanitizeForcedApiVideoSettings(nextSettings, { hasAudio }))
      return
    }

    // Clamp duration when resolution changes for local generation
    if (key === 'videoResolution' && !forceApiGenerations) {
      const maxDur = LOCAL_MAX_DURATION[value as string] ?? 20
      if (nextSettings.duration > maxDur) {
        nextSettings.duration = maxDur
      }
    }

    onSettingsChange(nextSettings)
  }

  const localMaxDuration = LOCAL_MAX_DURATION[settings.videoResolution] ?? 20
  const durationOptions = forceApiGenerations
    ? [...getAllowedForcedApiDurations(settings.model, settings.videoResolution, settings.fps)]
    : [5, 6, 8, 10, 20].filter(d => d <= localMaxDuration)
  const resolutionOptions = forceApiGenerations
    ? (hasAudio ? ['1080p'] : [...FORCED_API_VIDEO_RESOLUTIONS])
    : ['1080p', '720p', '540p']
  const fpsOptions = forceApiGenerations ? [...FORCED_API_VIDEO_FPS] : [24, 25, 50]

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
            <option value="1:1">1:1 — Square</option>
            <option value="16:9">16:9 — YouTube</option>
            <option value="9:16">9:16 — TikTok / Reels</option>
            <option value="4:3">4:3 — Standard</option>
            <option value="3:4">3:4 — Portrait</option>
            <option value="4:5">4:5 — Instagram Post</option>
            <option value="21:9">21:9 — Cinematic</option>
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

        {/* Variations Slider */}
        <div className="space-y-1.5">
          <div className="flex items-center justify-between">
            <label className="text-xs font-medium text-zinc-400">Variations</label>
            <span className="text-xs text-zinc-500">{settings.variations || 1}</span>
          </div>
          <input
            type="range"
            min={1}
            max={12}
            value={settings.variations || 1}
            onChange={(e) => handleChange('variations', parseInt(e.target.value))}
            disabled={disabled}
            className="w-full h-1.5 bg-zinc-700 rounded-full appearance-none cursor-pointer accent-blue-500"
          />
          <div className="flex justify-between text-[10px] text-zinc-600">
            <span>1</span>
            <span>12</span>
          </div>
        </div>

        {/* LoRA */}
        <div className="space-y-1.5">
          <label className="text-xs font-medium text-zinc-400">LoRA</label>
          <div className="flex items-center gap-2">
            <button
              type="button"
              disabled={disabled}
              onClick={async () => {
                try {
                  const files = await window.electronAPI.showOpenFileDialog({
                    title: 'Select LoRA (.safetensors) or config (.json)',
                    filters: [
                      { name: 'LoRA Files', extensions: ['safetensors', 'json'] },
                      { name: 'SafeTensors', extensions: ['safetensors'] },
                      { name: 'Config JSON', extensions: ['json'] },
                    ],
                  })
                  if (!files || files.length === 0) return
                  const filePath = files[0]
                  const ext = filePath.split('.').pop()?.toLowerCase()

                  if (ext === 'json') {
                    // User selected a config JSON — extract trigger phrase
                    try {
                      const { data } = await window.electronAPI.readLocalFile(filePath)
                      const json = JSON.parse(atob(data))
                      const trigger = json.default_caption || json.trigger_phrase || json.instance_prompt || ''
                      if (trigger) {
                        onSettingsChange({ ...settings, loraTriggerPhrase: trigger })
                      }
                    } catch { /* ignore parse errors */ }
                  } else {
                    // User selected a safetensors file
                    onSettingsChange({ ...settings, loraPath: filePath })
                  }
                } catch { /* cancelled */ }
              }}
              className="flex-1 px-3 py-1.5 text-xs text-left bg-zinc-800 border border-zinc-700 rounded-lg hover:border-zinc-500 truncate disabled:opacity-50"
            >
              {settings.loraPath
                ? settings.loraPath.split(/[/\\]/).pop()
                : 'None — click to browse'}
            </button>
            {settings.loraPath && (
              <button
                type="button"
                onClick={() => onSettingsChange({ ...settings, loraPath: null, loraWeight: 1.0, loraTriggerPhrase: null })}
                className="px-2 py-1.5 text-xs text-zinc-400 hover:text-red-400 bg-zinc-800 border border-zinc-700 rounded-lg"
                title="Remove LoRA"
              >
                ✕
              </button>
            )}
          </div>
          {settings.loraPath && (
            <>
              <div className="flex items-center justify-between">
                <label className="text-xs text-zinc-500">Weight</label>
                <div className="flex items-center gap-2">
                  <input
                    type="range"
                    min={0}
                    max={2}
                    step={0.05}
                    value={settings.loraWeight ?? 1.0}
                    onChange={(e) => onSettingsChange({ ...settings, loraWeight: parseFloat(e.target.value) })}
                    disabled={disabled}
                    className="w-24 h-1.5 bg-zinc-700 rounded-full appearance-none cursor-pointer accent-blue-500"
                  />
                  <span className="text-xs text-zinc-500 w-8 text-right">{(settings.loraWeight ?? 1.0).toFixed(2)}</span>
                </div>
              </div>
              {/* Trigger Phrase */}
              <div className="space-y-1">
                <div className="flex items-center justify-between">
                  <label className="text-xs text-zinc-500">Trigger Phrase</label>
                  {!settings.loraTriggerPhrase && (
                    <button
                      type="button"
                      onClick={async () => {
                        try {
                          const files = await window.electronAPI.showOpenFileDialog({
                            title: 'Select LoRA config (.json)',
                            filters: [{ name: 'Config JSON', extensions: ['json'] }],
                          })
                          if (!files || files.length === 0) return
                          const { data } = await window.electronAPI.readLocalFile(files[0])
                          const json = JSON.parse(atob(data))
                          const trigger = json.default_caption || json.trigger_phrase || json.instance_prompt || ''
                          if (trigger) {
                            onSettingsChange({ ...settings, loraTriggerPhrase: trigger, loraTriggerMode: settings.loraTriggerMode || 'prepend' })
                          }
                        } catch { /* cancelled or parse error */ }
                      }}
                      disabled={disabled}
                      className="text-[10px] text-blue-400 hover:text-blue-300"
                    >
                      Load from config
                    </button>
                  )}
                </div>
                <div className="flex items-center gap-1.5">
                  <input
                    type="text"
                    value={settings.loraTriggerPhrase || ''}
                    onChange={(e) => onSettingsChange({ ...settings, loraTriggerPhrase: e.target.value || null })}
                    placeholder="e.g. in the style of xyz"
                    disabled={disabled}
                    className="flex-1 px-2 py-1 text-xs bg-zinc-800 border border-zinc-700 rounded-lg text-zinc-300 placeholder-zinc-600 focus:outline-none focus:border-zinc-500 disabled:opacity-50"
                  />
                  {settings.loraTriggerPhrase && (
                    <button
                      type="button"
                      onClick={() => onSettingsChange({ ...settings, loraTriggerPhrase: null, loraTriggerMode: 'off' })}
                      className="text-zinc-500 hover:text-red-400 text-xs"
                      title="Clear trigger phrase"
                    >
                      ✕
                    </button>
                  )}
                </div>
                {settings.loraTriggerPhrase && (
                  <div className="flex items-center gap-1.5 mt-1">
                    {(['prepend', 'append', 'off'] as const).map((mode) => {
                      const active = (settings.loraTriggerMode || 'prepend') === mode
                      return (
                        <button
                          key={mode}
                          type="button"
                          onClick={() => onSettingsChange({ ...settings, loraTriggerMode: mode })}
                          disabled={disabled}
                          className={`px-2 py-0.5 text-[10px] rounded-md border transition-colors ${
                            active
                              ? mode === 'off'
                                ? 'bg-zinc-700 border-zinc-600 text-zinc-300'
                                : 'bg-blue-600/20 border-blue-500/40 text-blue-400'
                              : 'bg-zinc-800/50 border-zinc-700/50 text-zinc-600 hover:text-zinc-400 hover:border-zinc-600'
                          } disabled:opacity-50`}
                        >
                          {mode === 'prepend' ? 'Prepend' : mode === 'append' ? 'Append' : 'Off'}
                        </button>
                      )
                    })}
                  </div>
                )}
              </div>
            </>
          )}
        </div>
      </div>
    )
  }

  // Video mode settings
  return (
    <div className="space-y-4">
      {/* Model Selection */}
      <Select
        label="Model"
        value={settings.model}
        onChange={(e) => handleChange('model', e.target.value)}
        disabled={disabled}
      >
        {!forceApiGenerations && (
          <option value="fast">LTX 2.3 Fast</option>
        )}
        {forceApiGenerations && (
          <>
            <option value="fast" disabled={hasAudio}>LTX-2.3 Fast (API)</option>
            <option value="pro">LTX-2.3 Pro (API)</option>
          </>
        )}
        <option value="seedance-1.5-pro" disabled={!hasReplicateApiKey}>
          Seedance 1.5 Pro (Cloud){!hasReplicateApiKey ? ' — needs API key' : ''}
        </option>
      </Select>

      {/* Duration, Resolution, FPS Row */}
      <div className="grid grid-cols-3 gap-3">
        <Select
          label="Duration"
          value={settings.duration}
          onChange={(e) => handleChange('duration', parseInt(e.target.value))}
          disabled={disabled}
        >
          {durationOptions.map((duration) => (
            <option key={duration} value={duration}>
              {duration} sec
            </option>
          ))}
        </Select>

        <Select
          label="Resolution"
          value={settings.videoResolution}
          onChange={(e) => handleChange('videoResolution', e.target.value)}
          disabled={disabled}
        >
          {resolutionOptions.map((resolution) => (
            <option key={resolution} value={resolution}>
              {resolution}
            </option>
          ))}
        </Select>

        <Select
          label="FPS"
          value={settings.fps}
          onChange={(e) => handleChange('fps', parseInt(e.target.value))}
          disabled={disabled}
        >
          {fpsOptions.map((fps) => (
            <option key={fps} value={fps}>
              {fps}
            </option>
          ))}
        </Select>
      </div>

      {/* Aspect Ratio */}
      <Select
        label="Aspect Ratio"
        value={settings.aspectRatio || '16:9'}
        onChange={(e) => handleChange('aspectRatio', e.target.value)}
        disabled={disabled}
      >
        {hasAudio ? (
          <option value="16:9">16:9 — YouTube / Landscape</option>
        ) : (
          <>
            <option value="16:9">16:9 — YouTube / Landscape</option>
            <option value="9:16">9:16 — TikTok / Reels / Shorts</option>
          </>
        )}
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
