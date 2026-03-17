import { useState, useRef, useEffect, useCallback, useMemo } from 'react'
import type { TimelineClip, Track, SubtitleClip, Asset } from '../../types/project'
import { DEFAULT_COLOR_CORRECTION } from '../../types/project'
import type { GenerationSettings } from '../../components/SettingsPanel'
import { copyToAssetFolder } from '../../lib/asset-copy'
import { backendFetch } from '../../lib/backend'
import { fileUrlToPath } from '../../lib/url-to-path'

export interface UseGapGenerationParams {
  clips: TimelineClip[]
  tracks: Track[]
  setClips: React.Dispatch<React.SetStateAction<TimelineClip[]>>
  setTracks: React.Dispatch<React.SetStateAction<Track[]>>
  setSubtitles: React.Dispatch<React.SetStateAction<SubtitleClip[]>>
  currentProjectId: string | null
  addAsset: (projectId: string, asset: Omit<Asset, 'id' | 'createdAt'>) => Asset
  resolveClipSrc: (clip: TimelineClip | null) => string
  regenGenerate: (prompt: string, imagePath: string | null, settings: GenerationSettings) => Promise<{ success: boolean; videoUrl: string | null; videoPath: string | null }>
  regenGenerateImage: (prompt: string, settings: GenerationSettings) => Promise<{ success: boolean }>
  regenVideoUrl: string | null
  regenVideoPath: string | null
  regenImageUrl: string | null
  regenImagePath: string | null
  isRegenerating: boolean
  regenProgress: number
  regenCancel: () => void
  regenReset: () => void
  regenError: string | null
  projectId: string
}

export function useGapGeneration({
  clips,
  tracks,
  setClips,
  setTracks,
  setSubtitles,
  currentProjectId,
  addAsset,
  resolveClipSrc,
  regenGenerate,
  regenGenerateImage,
  regenVideoUrl,
  regenVideoPath,
  regenImageUrl,
  regenImagePath,
  isRegenerating,
  regenProgress,
  regenCancel,
  regenReset,
  regenError,
  projectId,
}: UseGapGenerationParams) {
  // Gap selection and generation
  const [selectedGap, setSelectedGap] = useState<{ trackIndex: number; startTime: number; endTime: number } | null>(null)
  const [gapGenerateMode, setGapGenerateMode] = useState<'text-to-video' | 'image-to-video' | 'text-to-image' | null>(null)
  const gapGenerateModeRef = useRef(gapGenerateMode)
  gapGenerateModeRef.current = gapGenerateMode
  const [gapPrompt, setGapPrompt] = useState('')
  const [gapSettings, setGapSettings] = useState<GenerationSettings>({
    model: 'fast',
    duration: 5,
    videoResolution: '540p',
    fps: 24,
    audio: true,
    cameraMotion: 'none',
    imageResolution: '1080p',
    imageAspectRatio: '16:9',
    imageSteps: 30,
  })
  const [gapImageFile, setGapImageFile] = useState<File | null>(null)
  const gapImageInputRef = useRef<HTMLInputElement>(null)
  const [gapApplyAudioToTrack, setGapApplyAudioToTrack] = useState(true)

  useEffect(() => {
    if (gapGenerateMode === 'text-to-image' && gapImageFile) {
      setGapImageFile(null)
    }
  }, [gapGenerateMode, gapImageFile])

  // Tracks the gap currently being generated in the background (after modal closes)
  const [generatingGap, setGeneratingGap] = useState<{
    trackIndex: number; startTime: number; endTime: number
    mode: 'text-to-video' | 'image-to-video' | 'text-to-image'
    prompt: string; settings: GenerationSettings
    imageFile: File | null; applyAudio: boolean
  } | null>(null)

  // Gap context-aware prompt suggestion
  const [gapSuggesting, setGapSuggesting] = useState(false)
  const [gapSuggestion, setGapSuggestion] = useState<string | null>(null)
  const [gapSuggestionError, setGapSuggestionError] = useState(false)
  const [gapSuggestionNoApiKey, setGapSuggestionNoApiKey] = useState(false)
  const gapSuggestionAbortRef = useRef<AbortController | null>(null)
  // Frames extracted from neighboring clips for the gap animation header
  const [gapBeforeFrame, setGapBeforeFrame] = useState<string | null>(null)
  const [gapAfterFrame, setGapAfterFrame] = useState<string | null>(null)

  // --- Gap detection: find empty spaces between clips on each non-subtitle track ---
  const timelineGaps = useMemo(() => {
    const gaps: { trackIndex: number; startTime: number; endTime: number }[] = []
    
    tracks.forEach((track, trackIdx) => {
      if (track.type === 'subtitle') return
      
      const trackClips = clips
        .filter(c => c.trackIndex === trackIdx)
        .sort((a, b) => a.startTime - b.startTime)
      
      if (trackClips.length === 0) return
      
      if (trackClips[0].startTime > 0.05) {
        gaps.push({ trackIndex: trackIdx, startTime: 0, endTime: trackClips[0].startTime })
      }
      
      for (let i = 0; i < trackClips.length - 1; i++) {
        const endOfCurrent = trackClips[i].startTime + trackClips[i].duration
        const startOfNext = trackClips[i + 1].startTime
        if (startOfNext - endOfCurrent > 0.05) {
          gaps.push({ trackIndex: trackIdx, startTime: endOfCurrent, endTime: startOfNext })
        }
      }
    })
    
    return gaps
  }, [clips, tracks])

  // Delete gap: ripple all clips on the same track (and optionally all tracks) to close it
  const deleteGap = useCallback((gap: { trackIndex: number; startTime: number; endTime: number }) => {
    const gapDuration = gap.endTime - gap.startTime
    
    setClips(prev => prev.map(c => {
      if (c.startTime >= gap.endTime) {
        return { ...c, startTime: Math.max(0, c.startTime - gapDuration) }
      }
      return c
    }))
    
    setSubtitles(prev => prev.map(s => {
      if (s.startTime >= gap.endTime) {
        return { ...s, startTime: Math.max(0, s.startTime - gapDuration), endTime: Math.max(0.1, s.endTime - gapDuration) }
      }
      return s
    }))
    
    setSelectedGap(null)
  }, [])

  // Handle starting generation in a gap
  const handleGapGenerate = useCallback(async () => {
    if (!selectedGap || !gapGenerateMode || !gapPrompt.trim() || !currentProjectId) return
    
    const gap = selectedGap
    const mode = gapGenerateMode
    const gapDuration = gap.endTime - gap.startTime
    
    const finalPrompt = gapPrompt.trim()
    
    const settings: GenerationSettings = {
      ...gapSettings,
      duration: Math.min(Math.max(1, Math.round(gapDuration)), gapSettings.model === 'pro' ? 10 : 20),
    }

    // Save generating gap state so we can show indicator and place result later
    setGeneratingGap({
      trackIndex: gap.trackIndex,
      startTime: gap.startTime,
      endTime: gap.endTime,
      mode,
      prompt: finalPrompt,
      settings,
      imageFile: gapImageFile,
      applyAudio: gapApplyAudioToTrack,
    })

    // Close the modal immediately so user can keep editing
    setSelectedGap(null)
    setGapGenerateMode(null)
    
    try {
      if (mode === 'text-to-image') {
        await regenGenerateImage(finalPrompt, settings)
      } else {
        // Convert File to filesystem path for the JSON-based generate API
        let imagePath: string | null = null
        if (gapImageFile) {
          const electronPath = (gapImageFile as any).path as string | undefined
          if (electronPath) {
            imagePath = electronPath
          } else {
            // In-memory file (e.g. canvas capture) — save to temp file
            const buf = await gapImageFile.arrayBuffer()
            const b64 = btoa(String.fromCharCode(...new Uint8Array(buf)))
            const modelsPath = await window.electronAPI.getModelsPath()
            const tmpDir = modelsPath.replace(/[/\\]models$/, '')
            const tmpPath = `${tmpDir}/tmp_gap_image_${Date.now()}.png`
            await window.electronAPI.saveFile(tmpPath, b64, 'base64')
            imagePath = tmpPath
          }
        }
        await regenGenerate(finalPrompt, imagePath, settings)
      }
    } catch (err) {
      console.error('Gap generation failed:', err)
      setGeneratingGap(null)
    }
  }, [selectedGap, gapGenerateMode, gapPrompt, gapSettings, gapImageFile, gapApplyAudioToTrack, currentProjectId, regenGenerate, regenGenerateImage])

  // When generation completes, place the result in the gap
  useEffect(() => {
    if (!generatingGap || isRegenerating) return
    
    const isImageResult = generatingGap.mode === 'text-to-image'
    const origUrl = isImageResult ? regenImageUrl : regenVideoUrl
    const origPath = isImageResult ? regenImagePath : regenVideoPath
    if (!origUrl || !currentProjectId) {
      // Generation ended with no result (cancelled or failed) - clean up
      if (!isRegenerating && generatingGap) {
        setGeneratingGap(null)
        if (!regenError) regenReset()
      }
      return
    }

    const gap = generatingGap
    const gapDuration = gap.endTime - gap.startTime
    const type = isImageResult ? 'image' : 'video'

    ;(async () => {
      const srcPath = origPath || origUrl
      const copied = await copyToAssetFolder(srcPath, projectId)
      const finalPath = copied?.path ?? srcPath
      const finalUrl = copied?.url ?? origUrl

      const asset = addAsset(currentProjectId, {
        type: type as 'image' | 'video',
        path: finalPath,
        url: finalUrl,
        prompt: gap.prompt,
        resolution: isImageResult ? gap.settings.imageResolution : gap.settings.videoResolution,
        duration: type === 'video' ? gapDuration : undefined,
        generationParams: {
          mode: (isImageResult ? 'text-to-image' : (gap.imageFile ? 'image-to-video' : 'text-to-video')) as 'text-to-video' | 'image-to-video' | 'text-to-image',
          prompt: gap.prompt,
          model: gap.settings.model,
          duration: Math.min(Math.max(1, Math.round(gapDuration)), gap.settings.model === 'pro' ? 10 : 20),
          resolution: isImageResult ? gap.settings.imageResolution : gap.settings.videoResolution,
          fps: gap.settings.fps,
          audio: gap.settings.audio,
          cameraMotion: gap.settings.cameraMotion,
          imageAspectRatio: gap.settings.imageAspectRatio,
          imageSteps: gap.settings.imageSteps,
        },
        takes: [{
          url: finalUrl,
          path: finalPath,
          createdAt: Date.now(),
        }],
        activeTakeIndex: 0,
      })
      
      const videoClipId = `clip-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
      const audioClipId = `clip-${Date.now()}-a-${Math.random().toString(36).substr(2, 9)}`

      // Determine if we should create a linked audio clip
      const shouldCreateAudio = type === 'video' && gap.applyAudio && gap.settings.audio

      // Find or create an audio track for the linked audio clip
      let audioTrackIndex = -1
      if (shouldCreateAudio) {
        audioTrackIndex = tracks.findIndex(t => t.kind === 'audio' && !t.locked && t.sourcePatched !== false)
        if (audioTrackIndex < 0) {
          const audioTrackCount = tracks.filter(t => t.kind === 'audio').length
          const newAudioTrack: Track = {
            id: `track-${Date.now()}-audio`,
            name: `A${audioTrackCount + 1}`,
            muted: false,
            locked: false,
            kind: 'audio',
          }
          audioTrackIndex = tracks.length
          setTracks(prev => [...prev, newAudioTrack])
        }
      }

      const newClip: TimelineClip = {
        id: videoClipId,
        assetId: asset.id,
        type: type === 'image' ? 'image' : 'video',
        startTime: gap.startTime,
        duration: gapDuration,
        trimStart: 0,
        trimEnd: 0,
        speed: 1,
        reversed: false,
        muted: false,
        volume: 1,
        trackIndex: gap.trackIndex,
        asset,
        flipH: false,
        flipV: false,
        transitionIn: { type: 'none', duration: 0 },
        transitionOut: { type: 'none', duration: 0 },
        colorCorrection: { ...DEFAULT_COLOR_CORRECTION },
        opacity: 100,
        ...(shouldCreateAudio && audioTrackIndex >= 0 ? { linkedClipIds: [audioClipId] } : {}),
      }

      const newClips: TimelineClip[] = [newClip]

      if (shouldCreateAudio && audioTrackIndex >= 0) {
        newClips.push({
          id: audioClipId,
          assetId: asset.id,
          type: 'audio',
          startTime: gap.startTime,
          duration: gapDuration,
          trimStart: 0,
          trimEnd: 0,
          speed: 1,
          reversed: false,
          muted: false,
          volume: 1,
          trackIndex: audioTrackIndex,
          asset,
          flipH: false,
          flipV: false,
          transitionIn: { type: 'none', duration: 0 },
          transitionOut: { type: 'none', duration: 0 },
          colorCorrection: { ...DEFAULT_COLOR_CORRECTION },
          opacity: 100,
          linkedClipIds: [videoClipId],
        })
      }
      
      setClips(prev => [...prev, ...newClips])
      
      // Clean up generating state
      setGeneratingGap(null)
      setGapPrompt('')
      setGapImageFile(null)
      regenReset()
    })()
    
  }, [regenVideoUrl, regenImageUrl, isRegenerating, generatingGap, regenError])

  // --- Gap context-aware prompt suggestion ---
  // Use refs so the async function always reads the latest values without re-creating
  const gapPromptRef = useRef(gapPrompt)
  gapPromptRef.current = gapPrompt
  const selectedGapRef = useRef(selectedGap)
  selectedGapRef.current = selectedGap
  const gapGenerateModeLocalRef = useRef(gapGenerateMode)
  gapGenerateModeLocalRef.current = gapGenerateMode
  const clipsRef = useRef(clips)
  clipsRef.current = clips
  const gapImageFileRef = useRef(gapImageFile)
  gapImageFileRef.current = gapImageFile

  // Stable function that never changes identity — reads everything from refs
  const runSuggestion = useCallback(async (forceReplace: boolean = false) => {
    const gap = selectedGapRef.current
    const mode = gapGenerateModeLocalRef.current
    if (!gap || !mode) return
    
    gapSuggestionAbortRef.current?.abort()
    const abortController = new AbortController()
    gapSuggestionAbortRef.current = abortController
    
    try {
      setGapSuggesting(true)
      setGapSuggestion(null)
      setGapSuggestionError(false)
      setGapSuggestionNoApiKey(false)
      
      const trackClips = clipsRef.current
        .filter(c => c.trackIndex === gap.trackIndex && c.type !== 'audio')
        .sort((a, b) => a.startTime - b.startTime)

      const clipBefore = trackClips.find(c => {
        const clipEnd = c.startTime + c.duration
        return Math.abs(clipEnd - gap.startTime) < 0.05
      })

      const clipAfter = trackClips.find(c => {
        return Math.abs(c.startTime - gap.endTime) < 0.05
      })

      if (!clipBefore && !clipAfter) {
        setGapSuggesting(false)
        return
      }

      // beforeFrame/afterFrame are now file paths (sent to backend which reads the file)
      let beforeFrame = ''
      let afterFrame = ''
      let beforePrompt = ''
      let afterPrompt = ''
      // file:// URLs for displaying frames in the UI
      let beforeFrameUrl = ''
      let afterFrameUrl = ''

      const framePromises: Promise<void>[] = []

      if (clipBefore) {
        const clipSrc = resolveClipSrc(clipBefore)
        beforePrompt = clipBefore.asset?.prompt || ''
        if (clipSrc) {
          if (clipBefore.asset?.type === 'video') {
            const seekTime = clipBefore.trimStart + clipBefore.duration * clipBefore.speed - 0.1
            framePromises.push(
              window.electronAPI.extractVideoFrame(clipSrc, Math.max(0, seekTime), 512, 3)
                .then(result => { beforeFrame = result.path; beforeFrameUrl = result.url })
                .catch(() => {})
            )
          } else if (clipBefore.asset?.type === 'image') {
            beforeFrame = fileUrlToPath(clipSrc) || ''
            beforeFrameUrl = clipSrc
          }
        }
      }

      if (clipAfter) {
        const clipSrc = resolveClipSrc(clipAfter)
        afterPrompt = clipAfter.asset?.prompt || ''
        if (clipSrc) {
          if (clipAfter.asset?.type === 'video') {
            framePromises.push(
              window.electronAPI.extractVideoFrame(clipSrc, clipAfter.trimStart + 0.1, 512, 3)
                .then(result => { afterFrame = result.path; afterFrameUrl = result.url })
                .catch(() => {})
            )
          } else if (clipAfter.asset?.type === 'image') {
            afterFrame = fileUrlToPath(clipSrc) || ''
            afterFrameUrl = clipSrc
          }
        }
      }

      await Promise.all(framePromises)

      if (abortController.signal.aborted) return

      if (beforeFrameUrl) setGapBeforeFrame(beforeFrameUrl)
      if (afterFrameUrl) setGapAfterFrame(afterFrameUrl)
      
      if (!beforeFrame && !afterFrame && !beforePrompt && !afterPrompt) {
        setGapSuggesting(false)
        return
      }
      
      // Extract file path from the user's input image if present (for I2V suggestions)
      let inputImagePath = ''
      const imageFile = gapImageFileRef.current
      if (imageFile && mode === 'image-to-video') {
        const electronPath = (imageFile as any).path as string | undefined
        if (electronPath) {
          inputImagePath = electronPath
        }
      }
      
      const response = await backendFetch('/api/suggest-gap-prompt', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          gapDuration: gap.endTime - gap.startTime,
          mode,
          beforePrompt,
          afterPrompt,
          beforeFrame,
          afterFrame,
          ...(inputImagePath ? { inputImage: inputImagePath } : {}),
        }),
        signal: abortController.signal,
      })
      
      if (abortController.signal.aborted) return

      if (!response.ok) {
        let isApiKeyError = response.status === 401 || response.status === 403
        try {
          const errData = await response.json()
          const errStr = JSON.stringify(errData).toLowerCase()
          if (errStr.includes('api_key') || errStr.includes('gemini') || errStr.includes('no api key') || errStr.includes('api key')) {
            isApiKeyError = true
          }
        } catch {}
        if (isApiKeyError) setGapSuggestionNoApiKey(true)
        else setGapSuggestionError(true)
      } else {
        const data = await response.json()
        if (data.suggested_prompt && !abortController.signal.aborted) {
          setGapSuggestion(data.suggested_prompt)
          if (forceReplace || !gapPromptRef.current.trim()) {
            setGapPrompt(data.suggested_prompt)
          }
        }
      }
    } catch (err) {
      if (err instanceof Error && err.name === 'AbortError') return
      console.warn('Gap prompt suggestion failed:', err)
      setGapSuggestionError(true)
    } finally {
      if (!abortController.signal.aborted) {
        setGapSuggesting(false)
      }
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [resolveClipSrc])

  // Track whether we've already fired the initial suggestion for this gap+mode combo
  const suggestionFiredKeyRef = useRef<string | null>(null)

  // Auto-run suggestion ONCE when the panel opens (gap + mode set)
  useEffect(() => {
    if (!selectedGap || !gapGenerateMode) {
      setGapSuggesting(false)
      setGapSuggestion(null)
      setGapSuggestionError(false)
      setGapSuggestionNoApiKey(false)
      setGapBeforeFrame(null)
      setGapAfterFrame(null)
      gapSuggestionAbortRef.current?.abort()
      suggestionFiredKeyRef.current = null
      return
    }
    
    // Build a key from the gap identity + mode so we only fire once per unique open
    const key = `${selectedGap.trackIndex}:${selectedGap.startTime}:${selectedGap.endTime}:${gapGenerateMode}`
    if (suggestionFiredKeyRef.current === key) return
    suggestionFiredKeyRef.current = key
    
    runSuggestion(false)
    
    return () => { gapSuggestionAbortRef.current?.abort() }
  }, [selectedGap, gapGenerateMode, runSuggestion])

  // Auto re-analyze when the user adds/removes an input image in I2V mode
  const prevImageFileRef = useRef<File | null>(null)
  useEffect(() => {
    const prev = prevImageFileRef.current
    prevImageFileRef.current = gapImageFile
    
    // Only trigger when the image actually changed (not on initial mount) and in I2V mode
    if (prev === gapImageFile) return
    if (gapGenerateMode !== 'image-to-video') return
    if (!selectedGap) return
    
    // Re-run suggestion with force replace since context changed
    runSuggestion(true)
  }, [gapImageFile, gapGenerateMode, selectedGap, runSuggestion])

  // Manual regenerate: force-replaces the prompt with the new suggestion
  const regenerateSuggestion = useCallback(() => {
    runSuggestion(true)
  }, [runSuggestion])

  // Cancel an in-progress gap generation
  const cancelGapGeneration = useCallback(() => {
    regenCancel()
    regenReset()
    setGeneratingGap(null)
  }, [regenCancel, regenReset])

  return {
    // State
    selectedGap,
    setSelectedGap,
    gapGenerateMode,
    setGapGenerateMode,
    gapGenerateModeRef,
    gapPrompt,
    setGapPrompt,
    gapSettings,
    setGapSettings,
    gapImageFile,
    setGapImageFile,
    gapImageInputRef,
    gapSuggesting,
    gapSuggestion,
    gapSuggestionError,
    gapSuggestionNoApiKey,
    gapBeforeFrame,
    gapAfterFrame,
    gapApplyAudioToTrack,
    setGapApplyAudioToTrack,
    regenerateSuggestion,
    // Background generation tracking
    generatingGap,
    isRegenerating,
    regenProgress,
    cancelGapGeneration,
    // Computed
    timelineGaps,
    // Actions
    deleteGap,
    handleGapGenerate,
  }
}
