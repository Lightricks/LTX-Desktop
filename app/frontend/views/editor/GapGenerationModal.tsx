import React, { useMemo, useEffect, useState, useRef } from 'react'
import {
  X, Upload, Video, Image,
  Loader2, Sparkles, RefreshCw, Info
} from 'lucide-react'
import { pathToFileUrl } from '../../lib/file-url'
import { SettingsPanel } from '../../components/SettingsPanel'
import type { GenerationSettings } from '../../components/SettingsPanel'
import type { GenerationMode } from '../../components/SettingsPanel'
import type { VideoGenerationModelSpecItem } from '../../lib/video-generation-model-specs'

interface TimelineGap {
  trackIndex: number
  startTime: number
  endTime: number
}

type GapGenerateMode = 'text-to-video' | 'image-to-video' | 'text-to-image'


interface GapGenerationModalProps {
  selectedGap: TimelineGap | null
  anchorPosition?: { x: number; gapTop: number; gapBottom: number } | null
  gapGenerateMode: GapGenerateMode | null
  setGapGenerateMode: (mode: GapGenerateMode | null) => void
  gapPrompt: string
  setGapPrompt: (prompt: string) => void
  gapSuggesting: boolean
  gapSuggestion: string | null
  gapBeforeFrame: string | null
  gapAfterFrame: string | null
  gapSettings: GenerationSettings
  setGapSettings: (settings: GenerationSettings) => void
  gapVideoModelSpecs: VideoGenerationModelSpecItem[]
  gapVideoSettingsMessage?: string | null
  gapCanGenerateVideo: boolean
  gapImageFile: File | null
  setGapImageFile: (file: File | null) => void
  gapImageInputRef: React.RefObject<HTMLInputElement>
  isRegenerating: boolean
  regenStatusMessage: string
  regenProgress: number
  regenReset: () => void
  handleGapGenerate: () => void
  handleCloseGap: () => void
  setSelectedGap: (gap: TimelineGap | null) => void
  gapApplyAudioToTrack: boolean
  setGapApplyAudioToTrack: (v: boolean) => void
  regenerateSuggestion: () => void
  gapSuggestionError?: boolean
  gapSuggestionNoApiKey?: boolean
}

export function GapGenerationModal({
  selectedGap,
  gapGenerateMode,
  setGapGenerateMode,
  gapPrompt,
  setGapPrompt,
  gapSuggesting,
  gapSuggestion,
  gapBeforeFrame,
  gapAfterFrame,
  gapSettings,
  setGapSettings,
  gapVideoModelSpecs,
  gapVideoSettingsMessage,
  gapCanGenerateVideo,
  gapImageFile,
  setGapImageFile,
  gapImageInputRef,
  isRegenerating,
  regenStatusMessage,
  regenProgress,
  regenReset,
  handleGapGenerate,
  handleCloseGap,
  setSelectedGap,
  gapApplyAudioToTrack,
  setGapApplyAudioToTrack,
  regenerateSuggestion,
  gapSuggestionError,
  gapSuggestionNoApiKey,
  anchorPosition,
}: GapGenerationModalProps) {
  if (!selectedGap) return null

  const isVideoMode = gapGenerateMode === 'text-to-video' || gapGenerateMode === 'image-to-video'
  const isImageMode = gapGenerateMode === 'text-to-image'

  const gapImageUrl = useMemo(() => {
    if (!gapImageFile) return null
    return URL.createObjectURL(gapImageFile)
  }, [gapImageFile])

  const modalTitle = isVideoMode
    ? (gapImageFile ? 'Image to Video' : 'Generate Video')
    : 'Generate Image'

  const settingsMode: GenerationMode = isVideoMode
    ? (gapImageFile ? 'image-to-video' : 'text-to-video')
    : 'text-to-image'

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.stopPropagation()
        if (gapGenerateMode) {
          setGapGenerateMode(null)
          regenReset()
        } else {
          setSelectedGap(null)
        }
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [gapGenerateMode, setGapGenerateMode, regenReset, setSelectedGap])

  const [startFrameEnabled, setStartFrameEnabled] = useState(true)
  const [endFrameEnabled, setEndFrameEnabled] = useState(false)
  const [startFrameOverride, setStartFrameOverride] = useState<string | null>(null)
  const [endFrameOverride, setEndFrameOverride] = useState<string | null>(null)
  const startFrameInputRef = useRef<HTMLInputElement>(null)
  const endFrameInputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    setStartFrameEnabled(true)
    setEndFrameEnabled(false)
    setStartFrameOverride(null)
    setEndFrameOverride(null)
  }, [gapGenerateMode])

  const displayedBeforeFrame = startFrameOverride ?? gapBeforeFrame
  const displayedAfterFrame = endFrameOverride ?? gapAfterFrame

  const handleFrameFileChange = (
    e: React.ChangeEvent<HTMLInputElement>,
    setter: (v: string | null) => void,
    onSelect: () => void
  ) => {
    const file = e.target.files?.[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = (ev) => { setter(ev.target?.result as string); onSelect() }
    reader.readAsDataURL(file)
    e.target.value = ''
  }

  return (
    <>
      {gapGenerateMode && (
        <div className="fixed inset-0 z-[100] flex flex-col items-center bg-black/60 backdrop-blur-sm p-4 overflow-y-auto">
          <div className="bg-surface-primary border border-separator rounded-xl shadow-2xl w-[420px] max-h-[calc(100vh-2rem)] flex flex-col overflow-hidden my-auto shrink-0">
            {/* Header */}
            <div className="flex items-center justify-between px-5 py-4 border-b border-separator">
              <div className="flex items-center gap-3">
                <div className={`w-8 h-8 rounded-xl flex items-center justify-center ${
                  isVideoMode ? 'bg-blue-600/20' : 'bg-success-soft'
                }`}>
                  {isVideoMode
                    ? <Video className="h-4 w-4 text-blue-400" />
                    : <Image className="h-4 w-4 text-fg-success" />}
                </div>
                <div>
                  <h2 className="text-sm font-semibold text-fg-primary">{modalTitle}</h2>
                  <p className="text-[11px] text-fg-tertiary">
                    Fill {(selectedGap.endTime - selectedGap.startTime).toFixed(1)}s gap on Track {selectedGap.trackIndex + 1}
                  </p>
                </div>
              </div>
              <button
                onClick={() => { setGapGenerateMode(null); regenReset() }}
                className="p-1.5 rounded-lg hover:bg-action text-fg-tertiary hover:text-fg-secondary transition-colors"
              >
                <X className="h-4 w-4" />
              </button>
            </div>

            {/* Timeline visualization */}
            <div className="px-5 pt-4 pb-2 space-y-2">
              {/* Label + toggle row */}
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-1.5">
                  <span className="text-xs text-fg-secondary font-medium">
                    {isVideoMode ? 'Generate from' : 'Context frames'}
                  </span>
                  <div className="relative group/info">
                    <Info className="h-3 w-3 text-fg-tertiary cursor-help" />
                    <div className="absolute left-0 top-full mt-2 w-60 p-2.5 bg-action border border-separator rounded-lg text-[10px] text-fg-secondary leading-relaxed invisible group-hover/info:visible opacity-0 group-hover/info:opacity-100 transition-opacity pointer-events-none shadow-xl z-20">
                      {isVideoMode ? (
                        <>
                          <p>Only one conditioning frame can be used at a time.</p>
                          <p className="mt-1.5">If <strong className="text-fg-primary">End frame</strong> is selected, it will be treated as the start frame, since the model does not currently support generating from an end frame. The video will then be generated from that frame and played in reverse.</p>
                        </>
                      ) : (
                        <p>Select frames from adjacent clips to provide context for the prompt.</p>
                      )}
                    </div>
                  </div>
                </div>
                {/* Segmented toggle */}
                <div className="flex bg-action rounded-lg p-0.5 gap-0.5">
                  <button
                    onClick={() => { if (startFrameEnabled) { setStartFrameEnabled(false) } else { setStartFrameEnabled(true); setEndFrameEnabled(false) } }}
                    className={`px-3 py-1 rounded-md text-xs font-medium transition-all ${
                      startFrameEnabled ? 'bg-action-active text-fg-primary shadow-sm' : 'text-fg-tertiary hover:text-fg-secondary'
                    }`}
                  >
                    Start frame
                  </button>
                  <button
                    onClick={() => { if (endFrameEnabled) { setEndFrameEnabled(false) } else { setEndFrameEnabled(true); setStartFrameEnabled(false) } }}
                    className={`px-3 py-1 rounded-md text-xs font-medium transition-all ${
                      endFrameEnabled ? 'bg-action-active text-fg-primary shadow-sm' : 'text-fg-tertiary hover:text-fg-secondary'
                    }`}
                  >
                    End frame
                  </button>
                </div>
              </div>

              {/* Frame strip */}
              <div className="flex h-[96px] rounded-xl overflow-hidden bg-surface-secondary border border-separator">
                {/* Before frame */}
                {displayedBeforeFrame ? (
                  <div
                    className="relative w-[38%] h-full flex-shrink-0 overflow-hidden rounded-l-xl group/before cursor-pointer"
                    onClick={() => { if (startFrameEnabled) { setStartFrameEnabled(false) } else { setStartFrameEnabled(true); setEndFrameEnabled(false) } }}
                  >
                    <img
                      src={pathToFileUrl(displayedBeforeFrame)}
                      alt=""
                      className={`w-full h-full object-cover transition-all duration-300 ${
                        !startFrameEnabled ? 'grayscale opacity-50' : ''
                      }`}
                    />
                    {/* Replace button */}
                    <div className="absolute top-1 left-1 inline-flex items-start opacity-0 group-hover/before:opacity-100 transition-all group/replace-start">
                      <button
                        onClick={(e) => { e.stopPropagation(); startFrameInputRef.current?.click() }}
                        className="p-1 rounded bg-black/50 text-fg-secondary hover:text-fg-primary hover:bg-black/75"
                      >
                        <Upload className="h-2.5 w-2.5" />
                      </button>
                      <div className="absolute left-0 top-full mt-1 px-1.5 py-0.5 bg-action border border-separator rounded text-[9px] text-fg-secondary whitespace-nowrap invisible group-hover/replace-start:visible pointer-events-none z-30">
                        Replace image
                      </div>
                    </div>
                    {/* Selection border */}
                    {startFrameEnabled && (
                      <div className="absolute inset-0 rounded-l-xl border-2 border-blue-500 pointer-events-none" />
                    )}
                  </div>
                ) : (
                  <div className="w-[38%] h-full flex-shrink-0 bg-action flex items-center justify-center">
                    <span className="text-fg-tertiary text-[8px]">No clip</span>
                  </div>
                )}

                {/* Center gap area */}
                <div className="flex-1 relative overflow-hidden">
                  {isVideoMode && gapImageFile && gapImageUrl ? (
                    <div className="relative w-full h-full group/center">
                      <img src={gapImageUrl} alt="" className="w-full h-full object-cover" />
                      <div className="absolute inset-0 ring-2 ring-inset ring-blue-500/50 pointer-events-none" />
                      <button
                        onClick={() => setGapImageFile(null)}
                        className="absolute top-1 right-1 p-0.5 rounded-full bg-black/70 text-fg-white hover:text-fg-danger opacity-0 group-hover/center:opacity-100 transition-opacity"
                      >
                        <X className="h-2.5 w-2.5" />
                      </button>
                      <div className="absolute bottom-0 inset-x-0 bg-gradient-to-t from-black/70 to-transparent py-1 px-2 flex items-center justify-center">
                        <span className="text-[8px] text-blue-200/90 font-medium">
                          Source frame
                        </span>
                      </div>
                    </div>
                  ) : (
                    <div className="w-full h-full relative">
                      <div className="absolute inset-0 bg-[color-mix(in_srgb,var(--semantic-bg-action-secondary-enabled)_70%,transparent)]" />
                      <div className="absolute inset-0 border border-dashed border-separator" />
                      <div className="absolute inset-0 flex flex-col items-center justify-center gap-1 px-3">
                        <Sparkles className="h-3.5 w-3.5 text-blue-400/40" />
                        <span className="text-xs text-fg-tertiary font-medium text-center">AI fills this gap</span>
                      </div>
                    </div>
                  )}
                </div>

                {/* After frame */}
                {displayedAfterFrame ? (
                  <div
                    className="relative w-[38%] h-full flex-shrink-0 overflow-hidden rounded-r-xl group/after cursor-pointer"
                    onClick={() => { if (endFrameEnabled) { setEndFrameEnabled(false) } else { setEndFrameEnabled(true); setStartFrameEnabled(false) } }}
                  >
                    <img
                      src={pathToFileUrl(displayedAfterFrame)}
                      alt=""
                      className={`w-full h-full object-cover transition-all duration-300 ${
                        !endFrameEnabled ? 'grayscale opacity-50' : ''
                      }`}
                    />
                    {/* Replace button */}
                    <div className="absolute top-1 left-1 inline-flex items-start opacity-0 group-hover/after:opacity-100 transition-all group/replace-end">
                      <button
                        onClick={(e) => { e.stopPropagation(); endFrameInputRef.current?.click() }}
                        className="p-1 rounded bg-black/50 text-fg-secondary hover:text-fg-primary hover:bg-black/75"
                      >
                        <Upload className="h-2.5 w-2.5" />
                      </button>
                      <div className="absolute left-0 top-full mt-1 px-1.5 py-0.5 bg-action border border-separator rounded text-[9px] text-fg-secondary whitespace-nowrap invisible group-hover/replace-end:visible pointer-events-none z-30">
                        Replace image
                      </div>
                    </div>
                    {/* Selection border */}
                    {endFrameEnabled && (
                      <div className="absolute inset-0 rounded-r-xl border-2 border-blue-500 pointer-events-none" />
                    )}
                  </div>
                ) : (
                  <div className="w-[38%] h-full flex-shrink-0 bg-action flex items-center justify-center">
                    <span className="text-fg-tertiary text-[8px]">No clip</span>
                  </div>
                )}
              </div>
            </div>

            {/* Body */}
            <div className="flex-1 overflow-auto px-5 py-4 space-y-4">
              {/* Prompt */}
              <div>
                <div className="flex items-center justify-between mb-1.5">
                  <label className="text-xs text-fg-tertiary uppercase font-semibold">Prompt</label>
                  <div className="flex items-center gap-2">
                    {gapSuggesting && (
                      <div className="flex items-center gap-1.5 text-[10px] text-fg-warning">
                        <Loader2 className="h-3 w-3 animate-spin" />
                        <span>Analyzing...</span>
                      </div>
                    )}
                    {!gapSuggesting && gapSuggestion && gapPrompt === gapSuggestion && (
                      <div className="flex items-center gap-1 text-[10px] text-fg-success">
                        <Sparkles className="h-3 w-3" />
                        <span>AI-suggested</span>
                      </div>
                    )}
                    {!gapSuggesting && !gapSuggestionNoApiKey && (
                      <button
                        onClick={regenerateSuggestion}
                        className="flex items-center gap-1 text-[10px] text-fg-tertiary hover:text-fg-secondary transition-colors px-1.5 py-0.5 rounded hover:bg-action"
                        title="Re-analyze surrounding clips and generate a new prompt suggestion"
                      >
                        <RefreshCw className="h-3 w-3" />
                        <span>Re-analyze</span>
                      </button>
                    )}
                  </div>
                </div>
                <div className="relative">
                  <textarea
                    value={gapPrompt}
                    onChange={(e) => setGapPrompt(e.target.value)}
                    onKeyDown={(e) => e.stopPropagation()}
                    placeholder={gapSuggesting
                      ? 'Analyzing surrounding shots for context...'
                      : isImageMode
                      ? 'Describe the image to generate...'
                      : (gapImageFile ? 'Describe the video to generate from the image...' : 'Describe the video shot to generate...')}
                    className={`w-full bg-action border rounded-xl p-3 text-sm text-fg-primary resize-none focus:outline-none focus:ring-1 placeholder-fg-tertiary ${
                      gapSuggesting
                        ? 'border-warning/40 focus:border-warning/50 focus:ring-warning/30 animate-pulse'
                        : 'border-separator focus:border-blue-500/50 focus:ring-blue-500/30'
                    }`}
                    rows={3}
                  />
                  {gapSuggestion && gapPrompt !== gapSuggestion && !gapSuggesting && (
                    <button
                      onClick={() => setGapPrompt(gapSuggestion)}
                      className="absolute top-1.5 right-1.5 px-2 py-1 rounded-lg bg-warning-soft border border-warning/30 text-fg-warning text-[10px] hover:bg-warning-soft transition-colors flex items-center gap-1"
                      title="Use AI-suggested prompt"
                    >
                      <Sparkles className="h-2.5 w-2.5" />
                      Use suggestion
                    </button>
                  )}
                </div>
                {gapSuggestionNoApiKey && (
                  <div className="mt-1.5 space-y-1.5">
                    <p className="text-[10px] text-fg-tertiary">
                      Gemini API key required for AI prompt suggestions.
                    </p>
                    <button
                      onClick={() => {
                        window.dispatchEvent(new CustomEvent('open-settings', { detail: { tab: 'apiKeys' } }))
                      }}
                      className="px-2.5 py-1 bg-blue-600 text-fg-primary text-[10px] rounded hover:bg-blue-500 transition-colors"
                    >
                      Configure in Settings
                    </button>
                  </div>
                )}
                {gapSuggestionError && !gapSuggesting && !gapSuggestion && (
                  <p className="text-[10px] text-fg-tertiary mt-1.5">Could not suggest a prompt. Type your own or try again.</p>
                )}
              </div>

              {/* Settings */}
              <div className="[&_select]:h-8 [&_select]:text-xs [&_select]:py-1 [&_label]:text-[10px] [&_label]:mb-1">
                <SettingsPanel
                  settings={gapSettings}
                  onSettingsChange={setGapSettings}
                  disabled={isRegenerating}
                  mode={settingsMode}
                  videoModelSpecs={gapVideoModelSpecs}
                  minimumDuration={selectedGap.endTime - selectedGap.startTime}
                  hideDuration={isVideoMode}
                  videoSettingsMessage={gapVideoSettingsMessage}
                />
              </div>

              {/* Apply audio to audio track toggle — only in video mode when audio is on */}
              {isVideoMode && gapSettings.audio && (
                <div
                  className={`flex items-center justify-between px-1 py-2 ${
                    isRegenerating ? 'opacity-40 pointer-events-none' : 'cursor-pointer'
                  }`}
                  onClick={() => !isRegenerating && setGapApplyAudioToTrack(!gapApplyAudioToTrack)}
                >
                  <div>
                    <span className="text-xs text-fg-secondary">Apply audio to audio track</span>
                    <p className="text-[10px] text-fg-tertiary mt-0.5">Place the generated audio as a linked clip on the audio track</p>
                  </div>
                  <div className={`relative w-9 h-5 rounded-full transition-colors flex-shrink-0 ${
                    gapApplyAudioToTrack ? 'bg-blue-600' : 'bg-action-hover'
                  }`}>
                    <div className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white shadow-sm transition-transform pointer-events-none ${
                      gapApplyAudioToTrack ? 'translate-x-4' : 'translate-x-0'
                    }`} />
                  </div>
                </div>
              )}

              {/* Progress */}
              {isRegenerating && (
                <div className="bg-action rounded-xl p-3 border border-separator">
                  <div className="flex items-center gap-2 mb-2">
                    <Loader2 className="h-3.5 w-3.5 text-blue-400 animate-spin" />
                    <span className="text-xs text-fg-secondary">{regenStatusMessage || 'Generating...'}</span>
                  </div>
                  <div className="h-1.5 bg-action-hover rounded-full overflow-hidden">
                    <div
                      className="h-full bg-blue-500 rounded-full transition-all duration-300"
                      style={{ width: `${regenProgress * 100}%` }}
                    />
                  </div>
                </div>
              )}
            </div>

            {/* Hidden file input */}
            <input
              ref={gapImageInputRef}
              type="file"
              accept="image/*"
              onChange={(e) => {
                const file = e.target.files?.[0]
                if (file) setGapImageFile(file)
                if (gapImageInputRef.current) gapImageInputRef.current.value = ''
              }}
              className="hidden"
            />

            {/* Footer */}
            <div className="px-5 py-3 flex items-center justify-end gap-2">
              <button
                onClick={() => { setGapGenerateMode(null); regenReset() }}
                className="px-3 py-1.5 rounded-md bg-action text-fg-secondary text-sm hover:bg-action-hover transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handleGapGenerate}
                disabled={isRegenerating || !gapPrompt.trim() || (isVideoMode && !gapCanGenerateVideo)}
                className="px-4 py-1.5 rounded-md bg-blue-600 text-fg-primary text-sm hover:bg-blue-500 transition-colors font-medium disabled:opacity-40 disabled:cursor-not-allowed flex items-center gap-1.5"
              >
                {isRegenerating ? (
                  <>
                    <Loader2 className="h-3 w-3 animate-spin" />
                    Generating...
                  </>
                ) : (
                  'Generate'
                )}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Gap action bar - shown when gap is selected but no generate mode yet */}
      {!gapGenerateMode && (() => {
        // Smart positioning: anchor to the clicked gap, with edge-case clamping
        const POPOVER_W = 200
        const POPOVER_H = 136
        const GAP_PX = 4
        const MARGIN = 8
        const vw = typeof window !== 'undefined' ? window.innerWidth : 1280
        const vh = typeof window !== 'undefined' ? window.innerHeight : 800

        const cx = anchorPosition?.x ?? vw / 2
        const gapTop = anchorPosition?.gapTop ?? vh - 220 - 52
        const gapBottom = anchorPosition?.gapBottom ?? vh - 220

        // Horizontal: center on gap, clamped so popover stays in viewport
        const left = Math.max(MARGIN, Math.min(cx - POPOVER_W / 2, vw - POPOVER_W - MARGIN))

        // Vertical: prefer below gap, flip above if not enough space below
        const spaceBelow = vh - gapBottom - GAP_PX
        const openAbove = spaceBelow < POPOVER_H + MARGIN
        const rawTop = openAbove ? gapTop - GAP_PX - POPOVER_H : gapBottom + GAP_PX
        const top = Math.max(MARGIN, Math.min(rawTop, vh - POPOVER_H - MARGIN))

        return (
        <>
        <div className="fixed inset-0 z-[90]" onClick={() => setSelectedGap(null)} />
        <div
          className="fixed z-[100] bg-surface-secondary border border-separator rounded-lg shadow-2xl overflow-hidden py-1"
          style={{ left, top, width: POPOVER_W }}
        >
          {/* Title */}
          <p className="text-[10px] text-fg-tertiary font-medium px-3 pt-1.5 pb-1.5">
            {(selectedGap.endTime - selectedGap.startTime).toFixed(1)}s gap selected
          </p>
          <div className="h-px bg-action mx-0 mb-1" />
          {/* Menu items */}
          <button
            onClick={() => setGapGenerateMode('text-to-image')}
            className="w-full px-3 py-1.5 text-left text-xs text-fg-primary hover:bg-action transition-colors"
          >
            Fill with Image
          </button>
          <button
            onClick={() => setGapGenerateMode('text-to-video')}
            className="w-full px-3 py-1.5 text-left text-xs text-fg-primary hover:bg-action transition-colors"
          >
            Fill with Video
          </button>
          <div className="h-px bg-action mx-0 my-1" />
          <button
            onClick={handleCloseGap}
            className="w-full px-3 py-1.5 text-left text-xs text-fg-secondary hover:bg-action hover:text-fg-secondary transition-colors flex items-center justify-between"
          >
            <span>Close gap</span>
            <kbd className="px-1 py-0.5 rounded bg-action border border-separator text-fg-tertiary text-[9px] font-mono leading-none">Del</kbd>
          </button>
        </div>
        </>
        )
      })()}

      {/* Hidden file inputs for replacing start/end frames */}
      <input
        ref={startFrameInputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(e) => handleFrameFileChange(e, setStartFrameOverride, () => { setStartFrameEnabled(true); setEndFrameEnabled(false) })}
      />
      <input
        ref={endFrameInputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(e) => handleFrameFileChange(e, setEndFrameOverride, () => { setEndFrameEnabled(true); setStartFrameEnabled(false) })}
      />
    </>
  )
}
