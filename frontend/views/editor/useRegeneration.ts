import { useState, useCallback, useEffect } from 'react'
import type { Asset, TimelineClip } from '../../types/project'
import type { GenerationSettings } from '../../components/SettingsPanel'
import { copyToAssetFolder } from '../../lib/asset-copy'
import { backendFetch } from '../../lib/backend'
import { fileUrlToPath } from '../../lib/url-to-path'
import { sanitizeForcedApiVideoSettings } from '../../lib/api-video-options'
import { logger } from '../../lib/logger'

export interface UseRegenerationParams {
  clips: TimelineClip[]
  setClips: React.Dispatch<React.SetStateAction<TimelineClip[]>>
  assets: Asset[]
  currentProjectId: string | null
  addAsset: (projectId: string, asset: Omit<Asset, 'id' | 'createdAt'>) => Asset
  updateAsset: (projectId: string, assetId: string, updates: Partial<Asset>) => void
  addTakeToAsset: (projectId: string, assetId: string, take: { url: string; path: string; createdAt: number }) => void
  deleteTakeFromAsset: (projectId: string, assetId: string, takeIndex: number) => void
  resolveClipSrc: (clip: TimelineClip | null) => string
  // Generation hook values
  regenGenerate: (prompt: string, imagePath: string | null, settings: GenerationSettings) => Promise<void>
  regenGenerateImage: (prompt: string, settings: GenerationSettings) => Promise<void>
  regenVideoUrl: string | null
  regenVideoPath: string | null
  regenImageUrl: string | null
  isRegenerating: boolean
  regenProgress: number
  regenStatusMessage: string
  regenCancel: () => void
  regenReset: () => void
  regenError: string | null
  assetSavePath: string | undefined | null
  shouldVideoGenerateWithLtxApi: boolean
}

export function useRegeneration(params: UseRegenerationParams) {
  const {
    clips, setClips, assets, currentProjectId,
    addAsset, updateAsset, addTakeToAsset, deleteTakeFromAsset,
    resolveClipSrc,
    regenGenerate, regenGenerateImage,
    regenVideoUrl, regenVideoPath, regenImageUrl,
    isRegenerating, regenProgress, regenStatusMessage,
    regenCancel, regenReset,
    regenError,
    assetSavePath,
    shouldVideoGenerateWithLtxApi,
  } = params

  // Track which asset/clip is being regenerated
  const [regeneratingAssetId, setRegeneratingAssetId] = useState<string | null>(null)
  const [regeneratingClipId, setRegeneratingClipId] = useState<string | null>(null)

  // IC-LoRA panel state
  const [showICLoraPanel, setShowICLoraPanel] = useState(false)
  const [icLoraSourceClipId, setIcLoraSourceClipId] = useState<string | null>(null)

  // Error state for imported assets that can't auto-generate a prompt
  const [regenerationPreError, setRegenerationPreError] = useState<string | null>(null)
  const dismissRegenerationPreError = useCallback(() => setRegenerationPreError(null), [])

  // Image-to-Video generation from an image clip on the timeline
  const [i2vClipId, setI2vClipId] = useState<string | null>(null)
  const [i2vPrompt, setI2vPrompt] = useState('')
  const [i2vSettings, setI2vSettings] = useState<GenerationSettings>({
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

  useEffect(() => {
    if (!shouldVideoGenerateWithLtxApi) return
    setI2vSettings((prev) => sanitizeForcedApiVideoSettings(prev))
  }, [shouldVideoGenerateWithLtxApi])

  const handleI2vGenerate = useCallback(async () => {
    if (!i2vClipId || !i2vPrompt.trim() || !currentProjectId) return

    const clip = clips.find(c => c.id === i2vClipId)
    if (!clip) return

    // Get the image URL for this clip and extract the filesystem path
    const imageUrl = resolveClipSrc(clip)
    if (!imageUrl) return

    const imagePath = fileUrlToPath(imageUrl)
    if (!imagePath) {
      logger.error(`I2V: cannot extract path from ${imageUrl}`)
      return
    }

    const rawSettings: GenerationSettings = {
      ...i2vSettings,
      duration: Math.min(Math.max(1, Math.round(clip.duration)), i2vSettings.model === 'pro' ? 10 : 20),
    }
    const settings = shouldVideoGenerateWithLtxApi ? sanitizeForcedApiVideoSettings(rawSettings) : rawSettings

    try {
      await regenGenerate(i2vPrompt, imagePath, settings)
    } catch (err) {
      logger.error(`I2V generation failed: ${err}`)
    }
  }, [i2vClipId, i2vPrompt, i2vSettings, currentProjectId, clips, resolveClipSrc, regenGenerate, shouldVideoGenerateWithLtxApi])

  // When I2V generation completes, replace the image clip with a video clip
  useEffect(() => {
    if (!i2vClipId || isRegenerating) return
    if (!regenVideoUrl || !currentProjectId) return

    const clip = clips.find(c => c.id === i2vClipId)
    if (!clip) { setI2vClipId(null); return }

      ;(async () => {
        const { path: finalPath, url: finalUrl } = await copyToAssetFolder(regenVideoPath || regenVideoUrl, regenVideoUrl, assetSavePath)
        const savedI2vSettings = shouldVideoGenerateWithLtxApi
          ? sanitizeForcedApiVideoSettings({
              ...i2vSettings,
              duration: Math.min(Math.max(1, Math.round(clip.duration)), i2vSettings.model === 'pro' ? 10 : 20),
            })
          : {
              ...i2vSettings,
              duration: Math.min(Math.max(1, Math.round(clip.duration)), i2vSettings.model === 'pro' ? 10 : 20),
            }

        const asset = addAsset(currentProjectId, {
          type: 'video',
          path: finalPath,
          url: finalUrl,
          prompt: i2vPrompt,
          resolution: savedI2vSettings.videoResolution,
          duration: clip.duration,
          generationParams: {
            mode: 'image-to-video',
            prompt: i2vPrompt,
            model: savedI2vSettings.model,
            duration: savedI2vSettings.duration,
            resolution: savedI2vSettings.videoResolution,
            fps: savedI2vSettings.fps,
            audio: savedI2vSettings.audio,
            cameraMotion: savedI2vSettings.cameraMotion,
          },
        takes: [{
          url: finalUrl,
          path: finalPath,
          createdAt: Date.now(),
        }],
        activeTakeIndex: 0,
      })

      setClips(prev => prev.map(c => {
        if (c.id !== i2vClipId) return c
        return {
          ...c,
          assetId: asset.id,
          type: 'video' as const,
          asset,
        }
      }))

      setI2vClipId(null)
      setI2vPrompt('')
      regenReset()
    })()

  }, [regenVideoUrl, isRegenerating, regenVideoPath, currentProjectId, clips, i2vClipId, i2vPrompt, i2vSettings, addAsset, setClips, regenReset, assetSavePath, shouldVideoGenerateWithLtxApi])

  // Clean up I2V state when generation fails
  useEffect(() => {
    if (!i2vClipId || isRegenerating || !regenError) return
    setI2vClipId(null)
    setI2vPrompt('')
  }, [regenError, i2vClipId, isRegenerating])

  const handleRegenerate = useCallback(async (assetId: string, clipId?: string) => {
    if (!currentProjectId || isRegenerating) return
    const asset = assets.find(a => a.id === assetId)
    if (!asset) return

    setRegeneratingAssetId(assetId)
    setRegeneratingClipId(clipId || null)

    // Mark the clip as regenerating for visual feedback
    if (clipId) {
      setClips(prev => prev.map(c => c.id === clipId ? { ...c, isRegenerating: true } : c))
    }

    // If the asset has no generationParams (imported asset), auto-generate a prompt from the first frame via Gemini
    let params = asset.generationParams
    if (!params) {
      try {
        const clipSrc = resolveClipSrc(clips.find(c => c.id === clipId) || { asset, assetId: asset.id } as any)
        let framePath = ''
        if (asset.type === 'video' && clipSrc) {
          const result = await window.electronAPI.extractVideoFrame(clipSrc, 0.1, 512, 3)
          framePath = result.path
        } else if (asset.type === 'image' && clipSrc) {
          framePath = fileUrlToPath(clipSrc) || ''
        }

        if (framePath) {
          // Ask Gemini to describe the frame
          const resp = await backendFetch('/api/suggest-gap-prompt', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              gapDuration: asset.duration || 5,
              mode: asset.type === 'image' ? 'text-to-image' : 'text-to-video',
              beforePrompt: '',
              afterPrompt: '',
              beforeFrame: framePath,
              afterFrame: '',
            }),
          })
          if (resp.ok) {
            const data = await resp.json()
            if (data.suggested_prompt) {
              params = {
                mode: asset.type === 'image' ? 'text-to-image' : 'text-to-video',
                prompt: data.suggested_prompt,
                model: 'fast',
                duration: asset.duration || 5,
                resolution: asset.resolution || '768x512',
                fps: 24,
                audio: false,
                cameraMotion: 'none',
              }
              // Save the generated params back to the asset so future regenerations are instant
              // (update the asset in project context)
              if (asset.id && currentProjectId) {
                updateAsset(currentProjectId, asset.id, { generationParams: params })
              }
            }
          }
        }
      } catch (err) {
        logger.warn(`Failed to auto-generate prompt for imported asset: ${err}`)
      }

      if (!params) {
        // Still no params — can't regenerate
        setRegeneratingAssetId(null)
        setRegeneratingClipId(null)
        if (clipId) {
          setClips(prev => prev.map(c => c.id === clipId ? { ...c, isRegenerating: false } : c))
        }
        setRegenerationPreError('Could not auto-generate a prompt for this clip. Try using "Send to GenSpace" instead.')
        return
      }
    }

    if (params.mode === 'retake') {
      setRegeneratingAssetId(null)
      setRegeneratingClipId(null)
      if (clipId) {
        setClips(prev => prev.map(c => c.id === clipId ? { ...c, isRegenerating: false } : c))
      }
      setRegenerationPreError('Retake assets cannot be regenerated yet. Try using Retake from the clip menu instead.')
      return
    }

    if (params.mode === 'text-to-image') {
      regenGenerateImage(params.prompt, {
        model: params.model as 'fast' | 'pro',
        duration: params.duration,
        videoResolution: '540p',
        fps: params.fps,
        audio: params.audio,
        cameraMotion: params.cameraMotion,
        imageResolution: params.resolution,
        imageAspectRatio: params.imageAspectRatio || '16:9',
        imageSteps: params.imageSteps || 4,
        variations: 1,
      })
    } else {
      // For video generation (T2V or I2V)
      // Extract filesystem path from the input image URL if present
      const imagePath = params.mode === 'image-to-video' && params.inputImageUrl
        ? fileUrlToPath(params.inputImageUrl)
        : null

      const rawVideoSettings: GenerationSettings = {
        model: params.model as 'fast' | 'pro',
        duration: params.duration,
        videoResolution: params.resolution,
        fps: params.fps,
        audio: params.audio,
        cameraMotion: params.cameraMotion,
        imageResolution: '1080p',
        imageAspectRatio: params.imageAspectRatio || '16:9',
        imageSteps: params.imageSteps || 4,
      }
      const videoSettings = shouldVideoGenerateWithLtxApi
        ? sanitizeForcedApiVideoSettings(rawVideoSettings)
        : rawVideoSettings

      regenGenerate(params.prompt, imagePath, videoSettings)
    }
  }, [currentProjectId, isRegenerating, assets, clips, regenGenerate, regenGenerateImage, resolveClipSrc, updateAsset, shouldVideoGenerateWithLtxApi])

  const handleCancelRegeneration = useCallback(() => {
    regenCancel()
    // Clear the regenerating visual on the clip
    if (regeneratingClipId) {
      setClips(prev => prev.map(c => c.id === regeneratingClipId ? { ...c, isRegenerating: false } : c))
    }
    setRegeneratingAssetId(null)
    setRegeneratingClipId(null)
    regenReset()
  }, [regenCancel, regenReset, regeneratingClipId])

  // Handle regeneration video result
  useEffect(() => {
    if (regenVideoUrl && regenVideoPath && regeneratingAssetId && currentProjectId && !isRegenerating) {
      ;(async () => {
        const { path: finalPath, url: finalUrl } = await copyToAssetFolder(regenVideoPath, regenVideoUrl, assetSavePath)

        addTakeToAsset(currentProjectId, regeneratingAssetId, {
          url: finalUrl,
          path: finalPath,
          createdAt: Date.now(),
        })

        if (regeneratingClipId) {
          setClips(prev => prev.map(c => {
            if (c.id !== regeneratingClipId) return c
            const asset = assets.find(a => a.id === c.assetId)
            const newTakeIdx = asset?.takes ? asset.takes.length : 1
            return { ...c, isRegenerating: false, takeIndex: newTakeIdx }
          }))
        }

        setRegeneratingAssetId(null)
        setRegeneratingClipId(null)
        regenReset()
      })()
    }
  }, [regenVideoUrl, regenVideoPath, regeneratingAssetId, currentProjectId, isRegenerating])

  // Handle regeneration image result
  useEffect(() => {
    if (regenImageUrl && regeneratingAssetId && currentProjectId && !isRegenerating) {
      ;(async () => {
        const { path: finalPath, url: finalUrl } = await copyToAssetFolder(regenImageUrl, regenImageUrl, assetSavePath)

        addTakeToAsset(currentProjectId, regeneratingAssetId, {
          url: finalUrl,
          path: finalPath,
          createdAt: Date.now(),
        })

        if (regeneratingClipId) {
          setClips(prev => prev.map(c => {
            if (c.id !== regeneratingClipId) return c
            const asset = assets.find(a => a.id === c.assetId)
            const newTakeIdx = asset?.takes ? asset.takes.length : 1
            return { ...c, isRegenerating: false, takeIndex: newTakeIdx }
          }))
        }

        setRegeneratingAssetId(null)
        setRegeneratingClipId(null)
        regenReset()
      })()
    }
  }, [regenImageUrl, regeneratingAssetId, currentProjectId, isRegenerating])

  // Clean up regeneration state when generation fails (let the error dialog handle regenReset)
  useEffect(() => {
    if (!regeneratingAssetId || isRegenerating || !regenError) return
    if (regeneratingClipId) {
      setClips(prev => prev.map(c =>
        c.id === regeneratingClipId ? { ...c, isRegenerating: false } : c
      ))
    }
    setRegeneratingAssetId(null)
    setRegeneratingClipId(null)
    // Do NOT call regenReset() — let the error dialog handle it
  }, [regenError, regeneratingAssetId, isRegenerating])

  // IC-LoRA result handler
  const handleICLoraResult = useCallback((result: { videoPath: string; sourceClipId: string | null }) => {
    if (!currentProjectId) return

    const pathNormalized = result.videoPath.replace(/\\/g, '/')
    const videoUrl = pathNormalized.startsWith('/') ? `file://${pathNormalized}` : `file:///${pathNormalized}`

    if (result.sourceClipId) {
      // Add as a new take on the source clip's asset
      const clip = clips.find(c => c.id === result.sourceClipId)
      const asset = clip?.assetId ? assets.find(a => a.id === clip.assetId) : null
      if (asset) {
        const newTakeIdx = asset.takes ? asset.takes.length : 1
        addTakeToAsset(currentProjectId, asset.id, {
          url: videoUrl,
          path: result.videoPath,
          createdAt: Date.now(),
        })
        setClips(prev => prev.map(c => {
          if (c.id !== result.sourceClipId) return c
          return { ...c, takeIndex: newTakeIdx }
        }))
      }
    } else {
      // Add as a new asset to the project
      addAsset(currentProjectId, {
        type: 'video',
        path: result.videoPath,
        url: videoUrl,
        prompt: 'IC-LoRA generation',
        resolution: '',
        takes: [{ url: videoUrl, path: result.videoPath, createdAt: Date.now() }],
        activeTakeIndex: 0,
      })
    }

    setShowICLoraPanel(false)
    setIcLoraSourceClipId(null)
  }, [currentProjectId, clips, assets, addTakeToAsset, addAsset])

  // Handle take navigation on a clip (also updates linked audio/video clips)
  const handleClipTakeChange = useCallback((clipId: string, direction: 'prev' | 'next') => {
    setClips(prev => {
      const clip = prev.find(c => c.id === clipId)
      if (!clip?.asset) return prev
      const asset = assets.find(a => a.id === clip.assetId)
      if (!asset?.takes || asset.takes.length <= 1) return prev

      const currentIdx = clip.takeIndex ?? (asset.activeTakeIndex ?? asset.takes.length - 1)
      let newIdx = direction === 'prev' ? currentIdx - 1 : currentIdx + 1
      newIdx = Math.max(0, Math.min(newIdx, asset.takes.length - 1))

      // Collect all clip IDs that should switch: this clip + all linked clips with the same asset
      const linkedIds = new Set(clip.linkedClipIds || [])
      linkedIds.add(clipId)

      return prev.map(c => {
        if (!linkedIds.has(c.id)) return c
        // Only update clips that share the same asset (linked audio/video pairs)
        if (c.assetId !== clip.assetId) return c
        return { ...c, takeIndex: newIdx }
      })
    })
  }, [assets])

  // Delete the currently displayed take from a clip's asset
  const handleDeleteTake = useCallback((clipId: string) => {
    if (!currentProjectId) return
    const clip = clips.find(c => c.id === clipId)
    if (!clip?.assetId) return
    const asset = assets.find(a => a.id === clip.assetId)
    if (!asset?.takes || asset.takes.length <= 1) return // Can't delete the only take

    const takeIdx = clip.takeIndex ?? (asset.activeTakeIndex ?? asset.takes.length - 1)

    // Delete the take from the asset in context
    deleteTakeFromAsset(currentProjectId, asset.id, takeIdx)

    // Update ALL clips that reference this asset: adjust their takeIndex
    setClips(prev => prev.map(c => {
      if (c.assetId !== asset.id) return c
      const cIdx = c.takeIndex ?? (asset.activeTakeIndex ?? asset.takes!.length - 1)
      if (cIdx === takeIdx) {
        // This clip was showing the deleted take → move to the previous one (or 0)
        return { ...c, takeIndex: Math.max(0, takeIdx - 1) }
      } else if (cIdx > takeIdx) {
        // This clip was showing a take after the deleted one → shift down by 1
        return { ...c, takeIndex: cIdx - 1 }
      }
      return c
    }))
  }, [clips, assets, currentProjectId, deleteTakeFromAsset])

  return {
    // State
    regeneratingAssetId, setRegeneratingAssetId,
    regeneratingClipId, setRegeneratingClipId,
    showICLoraPanel, setShowICLoraPanel,
    icLoraSourceClipId, setIcLoraSourceClipId,
    i2vClipId, setI2vClipId,
    i2vPrompt, setI2vPrompt,
    i2vSettings, setI2vSettings,
    // Pre-generation error (e.g. imported asset can't auto-generate prompt)
    regenerationPreError, dismissRegenerationPreError,
    // Passthrough from generation hook
    isRegenerating, regenProgress, regenStatusMessage,
    // Actions
    handleI2vGenerate,
    handleRegenerate,
    handleCancelRegeneration,
    handleICLoraResult,
    handleClipTakeChange,
    handleDeleteTake,
  }
}
