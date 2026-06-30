import React from 'react'
import { useT } from '../../lib/i18n'
import {
  Clipboard, Copy, Scissors, Trash2, Layers, Type, X, RefreshCw,
  ZoomIn, Film, Eye, FolderOpen, RotateCcw, Volume2, VolumeX,
  FlipHorizontal2, FlipVertical2, Link2, Unlink2,
  ChevronLeft, ChevronRight, Sparkles,
  Video, Camera,
} from 'lucide-react'
import type { Asset, TimelineClip, Track, TextOverlayStyle } from '../../types/project-model'
import { TEXT_PRESETS, translatePresetName } from '../../types/project'
import { COLOR_LABELS } from './video-editor-utils'

export type ClipContextMenuState =
  | { kind: 'background'; x: number; y: number }
  | { kind: 'clip'; clipId: string; x: number; y: number }

export interface ClipContextMenuProps {
  clipContextMenu: ClipContextMenuState
  contextClip: TimelineClip | null
  clipContextMenuRef: React.RefObject<HTMLDivElement>
  clips: TimelineClip[]
  tracks: Track[]
  selectedClipIds: Set<string>
  setSelectedClipIds: React.Dispatch<React.SetStateAction<Set<string>>>
  currentTime: number
  hasClipboard: boolean
  isRegenerating: boolean
  currentProjectId: string | null
  updateAsset: (projectId: string, assetId: string, updates: Partial<Asset>) => void
  handleCopy: () => void
  handleCut: () => void
  handlePaste: () => void
  setClipContextMenu: React.Dispatch<React.SetStateAction<ClipContextMenuState | null>>
  addTextClip: (style?: Partial<TextOverlayStyle>, startTime?: number) => void
  setClips: React.Dispatch<React.SetStateAction<TimelineClip[]>>
  handleRegenerate: (assetId: string, clipId: string) => void
  handleCancelRegeneration: () => void
  handleClipTakeChange: (clipId: string, direction: 'prev' | 'next') => void
  handleDeleteTake: (clipId: string) => void
  duplicateClip: (clipId: string) => void
  splitClipAtPlayhead: (clipId: string, atTime?: number, batchClipIds?: string[]) => void
  removeClip: (clipId: string) => void
  updateClip: (clipId: string, updates: Partial<TimelineClip>) => void
  getLiveAsset: (clip: TimelineClip) => Asset | null | undefined
  getMaxClipDuration: (clip: TimelineClip) => number
  onRevealAsset: (assetId: string) => void
  onCreateVideoFromImage: (clip: TimelineClip) => void
  onRetakeClip: (clip: TimelineClip) => void
  onICLoraClip: (clip: TimelineClip) => void
  canUseIcLora: boolean
  onCaptureFrameForVideo: (clip: TimelineClip) => void
  onCreateVideoFromAudio: (clip: TimelineClip) => void
}

// Reusable menu item component
function MenuItem({ icon: Icon, iconClass, label, shortcut, badge, badgeClass, disabled, danger, title, onClick }: {
  icon: any; iconClass?: string; label: string; shortcut?: string; badge?: string; badgeClass?: string
  disabled?: boolean; danger?: boolean; title?: string; onClick: () => void
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      title={title}
      className={`w-full text-left px-3 py-1.5 flex items-center gap-3 transition-colors ${
        disabled ? 'opacity-40 cursor-not-allowed' : 'hover:bg-zinc-700'
      } ${danger ? 'text-red-400' : 'text-zinc-300'}`}
    >
      <Icon className={`h-3.5 w-3.5 flex-shrink-0 ${iconClass || (danger ? '' : 'text-zinc-500')}`} />
      <span className="flex-1 truncate">{label}</span>
      {badge && <span className={`text-[10px] font-medium flex-shrink-0 ${badgeClass || 'text-zinc-500'}`}>{badge}</span>}
      {shortcut && <span className="text-zinc-600 text-[10px] flex-shrink-0">{shortcut}</span>}
    </button>
  )
}

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <div className="px-3 pt-1.5 pb-0.5 text-[9px] text-zinc-500 font-semibold uppercase tracking-widest select-none">
      {children}
    </div>
  )
}

function Divider() {
  return <div className="h-px bg-zinc-700 my-1" />
}

export function ClipContextMenu({
  clipContextMenu,
  contextClip,
  clipContextMenuRef,
  clips,
  selectedClipIds,
  setSelectedClipIds,
  currentTime,
  hasClipboard,
  isRegenerating,
  currentProjectId,
  updateAsset,
  handleCopy,
  handleCut,
  handlePaste,
  setClipContextMenu,
  addTextClip,
  setClips,
  handleRegenerate,
  handleCancelRegeneration,
  handleClipTakeChange,
  handleDeleteTake,
  duplicateClip,
  splitClipAtPlayhead,
  removeClip,
  updateClip,
  getLiveAsset,
  getMaxClipDuration,
  onRevealAsset,
  onCreateVideoFromImage,
  onRetakeClip,
  onICLoraClip,
  canUseIcLora,
  onCaptureFrameForVideo,
  onCreateVideoFromAudio,
}: ClipContextMenuProps) {
  const { t } = useT()
  const close = () => setClipContextMenu(null)
  const isBackground = clipContextMenu.kind === 'background'

  // Check if all selected clips are in the same linked group — if so, treat as single selection
  const multiSelected = (() => {
    if (selectedClipIds.size <= 1) return false
    if (!contextClip) return clipContextMenu.kind === 'background'
    // Expand the linked group of the context clip
    const linkedGroup = new Set([contextClip.id])
    const queue = [contextClip.id]
    while (queue.length > 0) {
      const id = queue.pop()!
      const c = clips.find(cl => cl.id === id)
      if (c?.linkedClipIds) {
        for (const lid of c.linkedClipIds) {
          if (!linkedGroup.has(lid)) { linkedGroup.add(lid); queue.push(lid) }
        }
      }
    }
    // If every selected clip is in this linked group, it's a single logical selection
    for (const sid of selectedClipIds) {
      if (!linkedGroup.has(sid)) return true
    }
    return false
  })()

  return (
    <div
      ref={clipContextMenuRef}
      className="fixed bg-zinc-800 border border-zinc-700 rounded-xl shadow-2xl py-1.5 z-[60] min-w-[220px] max-w-[280px] text-xs"
      style={{ left: clipContextMenu.x, top: clipContextMenu.y }}
      onClick={(e) => e.stopPropagation()}
    >
      {/* ════════════════════════════════════════════════
          BACKGROUND CONTEXT MENU (empty area right-click)
          ════════════════════════════════════════════════ */}
      {isBackground ? (
        <>
          <MenuItem icon={Clipboard} label={t('ctx.paste')} shortcut="Ctrl+V" disabled={!hasClipboard}
            onClick={() => { handlePaste(); close() }} />
          <Divider />
          <MenuItem icon={Type} iconClass="text-cyan-400" label={t('ctx.addText')}
            onClick={() => { addTextClip(undefined, currentTime); close() }} />
          {TEXT_PRESETS.slice(0, 4).map(preset => (
            <button
              key={preset.id}
              onClick={() => { addTextClip(preset.style, currentTime); close() }}
              className="w-full text-left px-3 py-1.5 text-zinc-400 hover:bg-zinc-700 flex items-center gap-3 pl-9"
            >
              <span className="text-[10px] text-cyan-500/70 flex-shrink-0">T</span>
              <span className="flex-1 truncate">{translatePresetName(preset.name, t)}</span>
            </button>
          ))}
          <Divider />
          <MenuItem icon={Layers} label={t('editor.selectAll')} shortcut="Ctrl+A"
            onClick={() => { setSelectedClipIds(new Set(clips.map(c => c.id))); close() }} />
        </>
      ) : multiSelected ? (
        /* ════════════════════════════════════════════════
           MULTI-CLIP CONTEXT MENU
           ════════════════════════════════════════════════ */
        <MultiClipMenu
          clips={clips}
          selectedClipIds={selectedClipIds} setSelectedClipIds={setSelectedClipIds}
          hasClipboard={hasClipboard}
          currentProjectId={currentProjectId}
          updateAsset={updateAsset}
          handleCopy={handleCopy} handleCut={handleCut} handlePaste={handlePaste}
          setClips={setClips}
          getMaxClipDuration={getMaxClipDuration}
          close={close}
        />
      ) : contextClip ? (
        /* ════════════════════════════════════════════════
           SINGLE CLIP CONTEXT MENU
           (For linked groups treated as single, prefer the video/image clip)
           ════════════════════════════════════════════════ */
        <SingleClipMenu
          contextClip={(() => {
            if (contextClip.type !== 'audio') return contextClip
            if (contextClip.linkedClipIds?.length) {
              const primary = clips.find(c => contextClip.linkedClipIds!.includes(c.id) && (c.type === 'video' || c.type === 'image'))
              if (primary) return primary
            }
            return contextClip
          })()}
          clips={clips}
          hasClipboard={hasClipboard}
          isRegenerating={isRegenerating}
          currentProjectId={currentProjectId}
          updateAsset={updateAsset}
          handleCopy={handleCopy} handleCut={handleCut} handlePaste={handlePaste}
          setClips={setClips}
          handleRegenerate={handleRegenerate}
          handleCancelRegeneration={handleCancelRegeneration}
          handleClipTakeChange={handleClipTakeChange}
          handleDeleteTake={handleDeleteTake}
          duplicateClip={duplicateClip}
          splitClipAtPlayhead={splitClipAtPlayhead}
          removeClip={removeClip}
          updateClip={updateClip}
          getLiveAsset={getLiveAsset}
          getMaxClipDuration={getMaxClipDuration}
          onRevealAsset={onRevealAsset}
          onCreateVideoFromImage={onCreateVideoFromImage}
          onRetakeClip={onRetakeClip}
          onICLoraClip={onICLoraClip}
          canUseIcLora={canUseIcLora}
          onCaptureFrameForVideo={onCaptureFrameForVideo}
          onCreateVideoFromAudio={onCreateVideoFromAudio}
          close={close}
        />
      ) : null}
    </div>
  )
}

/* ──────────────────────────────────────────────
   Single Clip Menu
   Layout:
   1. Clipboard        (Cut / Copy / Paste)
   2. Edit             (Duplicate / Split)
   3. Properties       (Speed / Reverse / Mute)
   4. Transform        (Flip H / Flip V)
   5. Structure        (Link / Move Track)
   6. AI Tools         (Regenerate / Upscale / I2V / Retake / IC-LoRA)
   7. Navigation       (Reveal in Assets / Explorer)
   8. Delete           (always last, red)
   ────────────────────────────────────────────── */
function SingleClipMenu({
  contextClip, clips, hasClipboard,
  isRegenerating,
  currentProjectId, updateAsset,
  handleCopy, handleCut, handlePaste, setClips,
  handleRegenerate, handleCancelRegeneration,
  handleClipTakeChange, handleDeleteTake,
  duplicateClip, splitClipAtPlayhead, removeClip, updateClip,
  getLiveAsset, getMaxClipDuration,
  onRevealAsset,
  onCreateVideoFromImage, onRetakeClip, onICLoraClip, canUseIcLora,
  onCaptureFrameForVideo,
  onCreateVideoFromAudio,
  close,
}: {
  contextClip: TimelineClip
  clips: TimelineClip[]
  hasClipboard: boolean
  isRegenerating: boolean
  currentProjectId: string | null
  updateAsset: (projectId: string, assetId: string, updates: Partial<Asset>) => void
  handleCopy: () => void; handleCut: () => void; handlePaste: () => void
  setClips: React.Dispatch<React.SetStateAction<TimelineClip[]>>
  handleRegenerate: (assetId: string, clipId: string) => void
  handleCancelRegeneration: () => void
  handleClipTakeChange: (clipId: string, direction: 'prev' | 'next') => void
  handleDeleteTake: (clipId: string) => void
  duplicateClip: (clipId: string) => void
  splitClipAtPlayhead: (clipId: string, atTime?: number, batchClipIds?: string[]) => void
  removeClip: (clipId: string) => void
  updateClip: (clipId: string, updates: Partial<TimelineClip>) => void
  getLiveAsset: (clip: TimelineClip) => Asset | null | undefined
  getMaxClipDuration: (clip: TimelineClip) => number
  onRevealAsset: (assetId: string) => void
  onCreateVideoFromImage: (clip: TimelineClip) => void
  onRetakeClip: (clip: TimelineClip) => void
  onICLoraClip: (clip: TimelineClip) => void
  canUseIcLora: boolean
  onCaptureFrameForVideo: (clip: TimelineClip) => void
  onCreateVideoFromAudio: (clip: TimelineClip) => void
  close: () => void
}) {
  const { t } = useT()
  const liveAsset = getLiveAsset(contextClip)
  const isAdjustment = contextClip.type === 'adjustment'
  const isVideo = contextClip.type === 'video'
  const isImage = contextClip.type === 'image'
  const hasAI = liveAsset && !isAdjustment

  return (
    <>
      {/* ── 1. Clipboard ── */}
      <MenuItem icon={Scissors} label={t('editor.cut')} shortcut="Ctrl+X" onClick={() => { handleCut(); close() }} />
      <MenuItem icon={Copy} label={t('editor.copy')} shortcut="Ctrl+C" onClick={() => { handleCopy(); close() }} />
      <MenuItem icon={Clipboard} label={t('ctx.paste')} shortcut="Ctrl+V" disabled={!hasClipboard} onClick={() => { handlePaste(); close() }} />

      <Divider />

      {/* ── 2. Edit ── */}
      <MenuItem icon={Copy} label={t('ctx.duplicate')} onClick={() => { duplicateClip(contextClip.id); close() }} />
      <MenuItem icon={Scissors} label={t('menu.splitAtPlayhead')} shortcut="B" onClick={() => { splitClipAtPlayhead(contextClip.id); close() }} />

      <Divider />

      {/* ── 3. Properties ── */}
      <SectionLabel>{t('menu.speed')}</SectionLabel>
      <div className="flex items-center gap-1 px-3 py-1">
        {[0.25, 0.5, 1, 1.5, 2, 4].map(speed => (
          <button
            key={speed}
            onClick={() => {
              const oldSpeed = contextClip.speed
              let newDuration = contextClip.duration * (oldSpeed / speed)
              const maxDur = getMaxClipDuration({ ...contextClip, speed })
              newDuration = Math.min(newDuration, maxDur)
              newDuration = Math.max(0.5, newDuration)
              updateClip(contextClip.id, { speed, duration: newDuration })
              close()
            }}
            className={`px-1.5 py-0.5 rounded text-[10px] font-medium transition-colors ${
              contextClip.speed === speed
                ? 'bg-blue-600 text-white'
                : 'bg-zinc-700 text-zinc-400 hover:bg-zinc-600 hover:text-white'
            }`}
          >
            {speed}x
          </button>
        ))}
      </div>
      <MenuItem icon={RotateCcw} label={t('ctx.reverse')}
        badge={contextClip.reversed ? t('menu.on') : undefined} badgeClass="text-blue-400"
        onClick={() => { updateClip(contextClip.id, { reversed: !contextClip.reversed }); close() }} />
      <MenuItem
        icon={contextClip.muted ? VolumeX : Volume2}
        label={contextClip.muted ? t('ctx.unmute') : t('ctx.mute')}
        badge={contextClip.muted ? t('menu.muted') : undefined} badgeClass="text-red-400"
        onClick={() => { updateClip(contextClip.id, { muted: !contextClip.muted }); close() }} />

      <Divider />

      {/* ── 4. Transform ── */}
      <MenuItem icon={FlipHorizontal2} label={t('ctx.flipHorizontal')}
        badge={contextClip.flipH ? t('menu.on') : undefined} badgeClass="text-cyan-400"
        onClick={() => { updateClip(contextClip.id, { flipH: !contextClip.flipH }); close() }} />
      <MenuItem icon={FlipVertical2} label={t('ctx.flipVertical')}
        badge={contextClip.flipV ? t('menu.on') : undefined} badgeClass="text-cyan-400"
        onClick={() => { updateClip(contextClip.id, { flipV: !contextClip.flipV }); close() }} />

      <Divider />

      {/* ── 5. Structure (Link / Track) ── */}
      {contextClip.linkedClipIds?.length ? (
        <MenuItem icon={Unlink2} label={t('ctx.unlinkAudio')} onClick={() => {
          const allLinked = new Set(contextClip.linkedClipIds!)
          setClips(prev => prev.map(c => {
            if (c.id === contextClip.id) return { ...c, linkedClipIds: undefined }
            if (allLinked.has(c.id)) {
              const remaining = (c.linkedClipIds || []).filter(lid => lid !== contextClip.id)
              return { ...c, linkedClipIds: remaining.length ? remaining : undefined }
            }
            return c
          }))
          close()
        }} />
      ) : null}
      {!contextClip.linkedClipIds?.length && (contextClip.type === 'video' || contextClip.type === 'audio') && (() => {
        const oppositeType = contextClip.type === 'video' ? 'audio' : 'video'
        const candidates = clips.filter(c =>
          c.type === oppositeType && !c.linkedClipIds?.length &&
          c.assetId === contextClip.assetId && Math.abs(c.startTime - contextClip.startTime) < 0.05
        )
        if (!candidates.length) return null
        return (
          <MenuItem icon={Link2} label={t('ctx.linkAudio')} onClick={() => {
            const candidateIds = candidates.map(c => c.id)
            setClips(prev => prev.map(c => {
              if (c.id === contextClip.id) return { ...c, linkedClipIds: candidateIds }
              if (candidateIds.includes(c.id)) return { ...c, linkedClipIds: [contextClip.id] }
              return c
            }))
            close()
          }} />
        )
      })()}

      {/* ── 6. Color Label ── */}
      <Divider />
      <SectionLabel>{t('menu.label')}</SectionLabel>
      <div className="px-3 py-1.5 flex items-center gap-1 flex-wrap">
        <button
          onClick={() => {
            if (contextClip.assetId) {
              // Update ALL clips that share this asset
              setClips(prev => prev.map(c => c.assetId === contextClip.assetId ? { ...c, colorLabel: undefined } : c))
              if (currentProjectId) updateAsset(currentProjectId, contextClip.assetId, { colorLabel: undefined })
            } else {
              updateClip(contextClip.id, { colorLabel: undefined })
            }
            close()
          }}
          className={`w-4 h-4 rounded-full border-2 flex items-center justify-center transition-all ${
            !contextClip.colorLabel ? 'border-white scale-110' : 'border-zinc-600 hover:border-zinc-400'
          }`}
          title={t('ctx.noLabel')}
        >
          <X className="h-2 w-2 text-zinc-400" />
        </button>
        {COLOR_LABELS.map(cl => (
          <button
            key={cl.id}
            onClick={() => {
              if (contextClip.assetId) {
                // Update ALL clips that share this asset
                setClips(prev => prev.map(c => c.assetId === contextClip.assetId ? { ...c, colorLabel: cl.id } : c))
                if (currentProjectId) updateAsset(currentProjectId, contextClip.assetId, { colorLabel: cl.id })
              } else {
                updateClip(contextClip.id, { colorLabel: cl.id })
              }
              close()
            }}
            className={`w-4 h-4 rounded-full transition-all ${
              contextClip.colorLabel === cl.id ? 'ring-2 ring-white ring-offset-1 ring-offset-zinc-800 scale-110' : 'hover:scale-125'
            }`}
            style={{ backgroundColor: cl.color }}
            title={cl.label}
          />
        ))}
      </div>

      {/* ── 7. AI Tools ── */}
      {hasAI && (
        <>
          <Divider />
          <SectionLabel>{t('menu.aiTools')}</SectionLabel>

          {contextClip.isRegenerating ? (
            <MenuItem icon={X} iconClass="text-red-400" label={t('ctx.cancelRegeneration')} onClick={() => { handleCancelRegeneration(); close() }} />
          ) : (
            <MenuItem icon={RefreshCw} iconClass="text-blue-400" label={t('ctx.regenerateShot')}
              disabled={isRegenerating} onClick={() => { handleRegenerate(contextClip.assetId!, contextClip.id); close() }} />
          )}

          {/* Take navigation */}
          {liveAsset!.takes && liveAsset!.takes.length > 1 && (
            <div className="px-3 py-1 flex items-center gap-2">
              <span className="text-[10px] text-zinc-500 flex-shrink-0">{t('menu.take')}</span>
              <button onClick={(e) => { e.stopPropagation(); handleClipTakeChange(contextClip.id, 'prev') }}
                className="p-0.5 rounded hover:bg-zinc-600 text-zinc-400 hover:text-white">
                <ChevronLeft className="h-3 w-3" />
              </button>
              <span className="text-[10px] text-zinc-300 min-w-[28px] text-center tabular-nums">
                {(contextClip.takeIndex ?? (liveAsset!.activeTakeIndex ?? liveAsset!.takes!.length - 1)) + 1}/{liveAsset!.takes!.length}
              </span>
              <button onClick={(e) => { e.stopPropagation(); handleClipTakeChange(contextClip.id, 'next') }}
                className="p-0.5 rounded hover:bg-zinc-600 text-zinc-400 hover:text-white">
                <ChevronRight className="h-3 w-3" />
              </button>
              <button
                onClick={(e) => {
                  e.stopPropagation()
                  if (confirm(t('assets.deleteTakeConfirm').replace('{{n}}', String((contextClip.takeIndex ?? (liveAsset!.activeTakeIndex ?? liveAsset!.takes!.length - 1)) + 1)))) {
                    handleDeleteTake(contextClip.id)
                  }
                }}
                className="p-0.5 rounded hover:bg-red-900/50 text-zinc-500 hover:text-red-400 ml-auto"
                title={t('ctx.deleteTake')}
              >
                <Trash2 className="h-3 w-3" />
              </button>
            </div>
          )}

          {isVideo && contextClip.assetId && (
            <MenuItem icon={ZoomIn} iconClass="text-zinc-500" label={t('ctx.upscale')}
              disabled={true} title={t('ctx.comingSoon')} onClick={() => {}} />
          )}
          {isImage && (
            <MenuItem icon={Film} iconClass="text-blue-400" label={t('ctx.imageToVideo')}
              onClick={() => { onCreateVideoFromImage(contextClip); close() }} />
          )}
          {isVideo && contextClip.assetId && (
            <>
              <MenuItem icon={Film} iconClass="text-blue-400" label={t('ctx.retakeSection')}
                onClick={() => { onRetakeClip(contextClip); close() }} />
              {canUseIcLora && (
                <MenuItem icon={Sparkles} iconClass="text-amber-400" label={t('ctx.icLoraStyle')}
                  onClick={() => { onICLoraClip(contextClip); close() }} />
              )}
            </>
          )}
          {contextClip.type === 'audio' && !contextClip.linkedClipIds?.length && (
            <MenuItem
              icon={Film}
              iconClass="text-emerald-400"
              label={t('ctx.createVideoA2V')}
              onClick={() => { onCreateVideoFromAudio(contextClip); close() }}
            />
          )}
          {(isVideo || isImage) && (
            <div className="relative group/capture">
              <button
                className="w-full text-left px-3 py-1.5 flex items-center gap-3 transition-colors hover:bg-zinc-700 text-zinc-300"
              >
                <Camera className="h-3.5 w-3.5 flex-shrink-0 text-amber-400" />
                <span className="flex-1 truncate">{t('menu.useFrameAs')}</span>
                <ChevronRight className="h-3 w-3 text-zinc-500" />
              </button>
              <div className="absolute left-full top-0 ml-0.5 min-w-[200px] bg-zinc-800 border border-zinc-700 rounded-lg shadow-xl py-1 z-[70] hidden group-hover/capture:block">
                <MenuItem icon={Video} iconClass="text-blue-400" label={t('ctx.generateInGenSpace')}
                  onClick={() => { onCaptureFrameForVideo(contextClip); close() }} />
                {isImage && (
                  <MenuItem icon={Film} iconClass="text-blue-400" label={t('ctx.imageToVideo')}
                    onClick={() => { onCreateVideoFromImage(contextClip); close() }} />
                )}
              </div>
            </div>
          )}
        </>
      )}

      {/* ── 7. Navigation ── */}
      {(contextClip.assetId || getLiveAsset(contextClip)?.path) && (
        <>
          <Divider />
          {contextClip.assetId && (
            <MenuItem icon={Eye} label={t('ctx.revealInAssets')} onClick={() => {
              onRevealAsset(contextClip.assetId!)
              close()
            }} />
          )}
          {(() => {
            if (!liveAsset) return null
            let filePath = liveAsset.path
            if (liveAsset.takes && liveAsset.takes.length > 0) {
              const takeIdx = contextClip.takeIndex ?? (liveAsset.activeTakeIndex ?? liveAsset.takes.length - 1)
              filePath = liveAsset.takes[Math.max(0, Math.min(takeIdx, liveAsset.takes.length - 1))].path
            }
            if (!filePath) return null
            const label = window.electronAPI?.platform === 'darwin' ? t('ctx.revealInFinder')
              : window.electronAPI?.platform === 'linux' ? t('menu.showInFiles')
              : t('ctx.revealInExplorer')
            return <MenuItem icon={FolderOpen} label={label} onClick={() => { window.electronAPI?.showItemInFolder({ filePath }); close() }} />
          })()}
        </>
      )}

      {/* ── 8. Delete (always last, always red) ── */}
      <Divider />
      <MenuItem icon={Trash2} label={t('editor.delete')} shortcut="Del" danger onClick={() => { removeClip(contextClip.id); close() }} />
    </>
  )
}

/* ──────────────────────────────────────────────
   Multi-Clip Menu
   Layout:
   1. Header           (N clips selected)
   2. Clipboard        (Cut / Copy / Paste)
   3. Properties       (Speed / Mute / Reverse)
   4. Transform        (Flip H / Flip V)
   5. Structure        (Link / Unlink / Move Track)
   6. Delete           (always last, red)
   ────────────────────────────────────────────── */
function MultiClipMenu({
  clips, selectedClipIds, setSelectedClipIds, hasClipboard,
  currentProjectId, updateAsset,
  handleCopy, handleCut, handlePaste, setClips, getMaxClipDuration, close,
}: {
  clips: TimelineClip[]
  selectedClipIds: Set<string>; setSelectedClipIds: React.Dispatch<React.SetStateAction<Set<string>>>
  hasClipboard: boolean
  currentProjectId: string | null
  updateAsset: (projectId: string, assetId: string, updates: Partial<Asset>) => void
  handleCopy: () => void; handleCut: () => void; handlePaste: () => void
  setClips: React.Dispatch<React.SetStateAction<TimelineClip[]>>
  getMaxClipDuration: (clip: TimelineClip) => number
  close: () => void
}) {
  const { t } = useT()
  const n = selectedClipIds.size
  const selectedClips = clips.filter(c => selectedClipIds.has(c.id))
  const allMuted = selectedClips.every(c => c.muted)
  const allReversed = selectedClips.every(c => c.reversed)
  const allFlipH = selectedClips.every(c => c.flipH)
  const allFlipV = selectedClips.every(c => c.flipV)
  const batchUpdate = (updates: Partial<TimelineClip>) => {
    setClips(prev => prev.map(c => selectedClipIds.has(c.id) ? { ...c, ...updates } : c))
    close()
  }

  const anyLinked = selectedClips.some(c => c.linkedClipIds?.length)
  const hasVideoAndAudio = selectedClips.some(c => c.type === 'video') && selectedClips.some(c => c.type === 'audio')
  const allFullyLinked = selectedClips.every(c => {
    const others = [...selectedClipIds].filter(id => id !== c.id)
    return others.every(oid => c.linkedClipIds?.includes(oid))
  })

  return (
    <>
      {/* ── Header ── */}
      <SectionLabel>{t('menu.nClipsSelected').replace('{{n}}', String(n))}</SectionLabel>

      {/* ── 1. Clipboard ── */}
      <MenuItem icon={Scissors} label={t('editor.cut')} shortcut="Ctrl+X" onClick={() => { handleCut(); close() }} />
      <MenuItem icon={Copy} label={t('editor.copy')} shortcut="Ctrl+C" onClick={() => { handleCopy(); close() }} />
      <MenuItem icon={Clipboard} label={t('ctx.paste')} shortcut="Ctrl+V" disabled={!hasClipboard} onClick={() => { handlePaste(); close() }} />

      <Divider />

      {/* ── 2. Properties ── */}
      <SectionLabel>{t('menu.speed')}</SectionLabel>
      <div className="flex items-center gap-1 px-3 py-1">
        {[0.25, 0.5, 1, 1.5, 2, 4].map(speed => (
          <button
            key={speed}
            onClick={() => {
              setClips(prev => prev.map(c => {
                if (!selectedClipIds.has(c.id)) return c
                const oldSpeed = c.speed
                let newDuration = c.duration * (oldSpeed / speed)
                const maxDur = getMaxClipDuration({ ...c, speed })
                newDuration = Math.min(newDuration, maxDur)
                newDuration = Math.max(0.5, newDuration)
                return { ...c, speed, duration: newDuration }
              }))
              close()
            }}
            className="px-1.5 py-0.5 rounded text-[10px] font-medium transition-colors bg-zinc-700 text-zinc-400 hover:bg-zinc-600 hover:text-white"
          >
            {speed}x
          </button>
        ))}
      </div>
      <MenuItem icon={allMuted ? VolumeX : Volume2} label={allMuted ? t('ctx.unmuteAll') : t('ctx.muteAll')}
        badge={allMuted ? t('menu.allMuted') : undefined} badgeClass="text-red-400"
        onClick={() => batchUpdate({ muted: !allMuted })} />
      <MenuItem icon={RotateCcw} label={allReversed ? t('ctx.unreverseAll') : t('ctx.reverseAll')}
        badge={allReversed ? t('menu.allOn') : undefined} badgeClass="text-blue-400"
        onClick={() => batchUpdate({ reversed: !allReversed })} />

      <Divider />

      {/* ── 3. Transform ── */}
      <MenuItem icon={FlipHorizontal2} label={allFlipH ? t('ctx.unflipAllHorizontal') : t('ctx.flipAllHorizontal')}
        badge={allFlipH ? t('menu.allOn') : undefined} badgeClass="text-cyan-400"
        onClick={() => batchUpdate({ flipH: !allFlipH })} />
      <MenuItem icon={FlipVertical2} label={allFlipV ? t('ctx.unflipAllVertical') : t('ctx.flipAllVertical')}
        badge={allFlipV ? t('menu.allOn') : undefined} badgeClass="text-cyan-400"
        onClick={() => batchUpdate({ flipV: !allFlipV })} />

      <Divider />

      {/* ── 4. Structure ── */}
      {anyLinked && (
        <MenuItem icon={Unlink2} label={t('ctx.unlink')} onClick={() => {
          const selIds = new Set(selectedClipIds)
          setClips(prev => prev.map(c => {
            if (!selIds.has(c.id)) return c
            const remaining = (c.linkedClipIds || []).filter(lid => !selIds.has(lid))
            return { ...c, linkedClipIds: remaining.length ? remaining : undefined }
          }))
          close()
        }} />
      )}
      {hasVideoAndAudio && !allFullyLinked && (
        <MenuItem icon={Link2} label={t('ctx.link')} onClick={() => {
          const selIds = [...selectedClipIds]
          setClips(prev => prev.map(c => {
            if (!selectedClipIds.has(c.id)) return c
            const otherIds = selIds.filter(id => id !== c.id)
            const existingLinks = new Set(c.linkedClipIds || [])
            otherIds.forEach(id => existingLinks.add(id))
            return { ...c, linkedClipIds: [...existingLinks] }
          }))
          close()
        }} />
      )}

      {/* ── 5. Color Label ── */}
      <Divider />
      <SectionLabel>{t('menu.label')}</SectionLabel>
      <div className="px-3 py-1.5 flex items-center gap-1 flex-wrap">
        <button
          onClick={() => {
            const affectedAssetIds = new Set(selectedClips.map(c => c.assetId).filter(Boolean) as string[])
            // Update selected clips AND all other clips sharing the same assets
            setClips(prev => prev.map(c => {
              if (selectedClipIds.has(c.id) || (c.assetId && affectedAssetIds.has(c.assetId))) {
                return { ...c, colorLabel: undefined }
              }
              return c
            }))
            if (currentProjectId) {
              affectedAssetIds.forEach(id => updateAsset(currentProjectId!, id, { colorLabel: undefined }))
            }
            close()
          }}
          className="w-4 h-4 rounded-full border-2 border-zinc-600 hover:border-zinc-400 flex items-center justify-center transition-all"
          title={t('ctx.noLabel')}
        >
          <X className="h-2 w-2 text-zinc-400" />
        </button>
        {COLOR_LABELS.map(cl => (
          <button
            key={cl.id}
            onClick={() => {
              const affectedAssetIds = new Set(selectedClips.map(c => c.assetId).filter(Boolean) as string[])
              // Update selected clips AND all other clips sharing the same assets
              setClips(prev => prev.map(c => {
                if (selectedClipIds.has(c.id) || (c.assetId && affectedAssetIds.has(c.assetId))) {
                  return { ...c, colorLabel: cl.id }
                }
                return c
              }))
              if (currentProjectId) {
                affectedAssetIds.forEach(id => updateAsset(currentProjectId!, id, { colorLabel: cl.id }))
              }
              close()
            }}
            className="w-4 h-4 rounded-full hover:scale-125 transition-all"
            style={{ backgroundColor: cl.color }}
            title={cl.label}
          />
        ))}
      </div>

      {/* ── 6. Delete ── */}
      <Divider />
      <MenuItem icon={Trash2} label={t('editor.delete')} shortcut="Del" danger onClick={() => {
        setClips(prev => prev.filter(c => !selectedClipIds.has(c.id)).map(c => {
          if (!c.linkedClipIds) return c
          const remaining = c.linkedClipIds.filter(lid => !selectedClipIds.has(lid))
          return { ...c, linkedClipIds: remaining.length ? remaining : undefined }
        }))
        setSelectedClipIds(new Set())
        close()
      }} />
    </>
  )
}
