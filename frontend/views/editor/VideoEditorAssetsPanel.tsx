import React, {
  forwardRef, useCallback, useEffect, useImperativeHandle, useRef, useState,
} from 'react'
import {
  FolderPlus, Folder, Upload, ChevronLeft, ChevronDown, ChevronRight, ChevronUp,
  X, RefreshCw, Loader2, Trash2, Music, Layers, Video, Image,
  LayoutGrid, List, ArrowUpDown, Pencil,
} from 'lucide-react'
import { useShallow } from 'zustand/react/shallow'
import type { Asset } from '../../types/project'
import { VideoThumbnailCard } from './VideoThumbnailCard'
import { getColorLabel } from './video-editor-utils'
import { Tooltip } from '../../components/ui/tooltip'
import { AssetContextMenu } from './AssetContextMenu'
import { TakeContextMenu } from './TakeContextMenu'
import { pathToFileUrl } from '../../lib/file-url'
import type { AssetListFilters } from './editor-state'
import { selectAssetBins, selectAssets, selectVisibleAssets } from './editor-selectors'
import { useEditorActions, useEditorStore } from './editor-store'

export interface VideoEditorAssetsPanelHandle {
  revealAsset: (assetId: string) => void
  deleteAsset: (target?: string | string[]) => void
}

export interface VideoEditorAssetsPanelProps {
  openSourceAsset: (asset: Asset) => void
  handleImportFile: (e: React.ChangeEvent<HTMLInputElement>) => void
  handleRegenerate: (assetId: string) => void
  handleCancelRegeneration: () => void
  isRegenerating: boolean
  regeneratingAssetId: string | null
  regenProgress: number
  regenStatusMessage: string
}

export const VideoEditorAssetsPanel = forwardRef<VideoEditorAssetsPanelHandle, VideoEditorAssetsPanelProps>(function VideoEditorAssetsPanel(props, ref) {
  const {
    openSourceAsset,
    handleImportFile,
    handleRegenerate,
    handleCancelRegeneration,
    isRegenerating,
    regeneratingAssetId,
    regenProgress,
    regenStatusMessage,
  } = props
  const actions = useEditorActions()
  const assets = useEditorStore(selectAssets)

  const [takesViewAssetId, setTakesViewAssetId] = useState<string | null>(null)
  const [creatingBin, setCreatingBin] = useState(false)
  const [newBinName, setNewBinName] = useState('')
  const [selectedBin, setSelectedBin] = useState<string | null>(null)
  const [assetFilter, setAssetFilter] = useState<'all' | 'video' | 'image' | 'audio'>('all')
  const [selectedAssetIds, setSelectedAssetIds] = useState<Set<string>>(new Set())
  const [assetLasso, setAssetLasso] = useState<{ startX: number; startY: number; currentX: number; currentY: number } | null>(null)
  const [assetContextMenu, setAssetContextMenu] = useState<{ assetId: string; x: number; y: number } | null>(null)
  const [takeContextMenu, setTakeContextMenu] = useState<{ assetId: string; takeIndex: number; x: number; y: number } | null>(null)
  const [binContextMenu, setBinContextMenu] = useState<{ bin: string; x: number; y: number } | null>(null)
  const [assetViewMode, setAssetViewMode] = useState<'grid' | 'list'>('grid')
  const [listSortCol, setListSortCol] = useState<'name' | 'type' | 'duration' | 'resolution' | 'date' | 'color'>('name')
  const [listSortDir, setListSortDir] = useState<'asc' | 'desc'>('asc')

  const fileInputRef = useRef<HTMLInputElement>(null)
  const assetGridRef = useRef<HTMLDivElement>(null)
  const newBinInputRef = useRef<HTMLInputElement>(null)
  const assetContextMenuRef = useRef<HTMLDivElement>(null)
  const takeContextMenuRef = useRef<HTMLDivElement>(null)
  const binContextMenuRef = useRef<HTMLDivElement>(null)
  const bins = useEditorStore(useShallow(selectAssetBins))
  const assetFilters: AssetListFilters = {
    assetFilter,
    selectedBin,
    assetViewMode,
    listSortCol,
    listSortDir,
  }
  const visibleAssets = useEditorStore(useShallow(state => selectVisibleAssets(state, assetFilters)))
  const filteredAssets = assetViewMode === 'list'
    ? assets.filter(asset => visibleAssets.some(visible => visible.id === asset.id))
    : visibleAssets

  const deleteAsset = useCallback((target?: string | string[]) => {
    const rawIds = target === undefined
      ? [...selectedAssetIds]
      : typeof target === 'string'
      ? [target]
      : target
    const ids = Array.from(new Set(rawIds.filter(Boolean)))
    if (ids.length === 0) return

    actions.deleteAssets(ids)
    setSelectedAssetIds(prev => {
      if (prev.size === 0) return prev
      const next = new Set(prev)
      ids.forEach(id => next.delete(id))
      return next
    })
    if (takesViewAssetId && ids.includes(takesViewAssetId)) {
      setTakesViewAssetId(null)
    }
  }, [actions, selectedAssetIds, takesViewAssetId])

  const revealAsset = useCallback((assetId: string) => {
    const asset = assets.find(a => a.id === assetId)
    if (!asset) return
    setAssetFilter('all')
    setSelectedBin(asset.bin ?? null)
    setTakesViewAssetId(null)
    setSelectedAssetIds(new Set([asset.id]))
    setTimeout(() => {
      const card = assetGridRef.current?.querySelector(`[data-asset-id="${asset.id}"]`)
      card?.scrollIntoView({ behavior: 'smooth', block: 'center' })
    }, 100)
  }, [assets])

  useImperativeHandle(ref, () => ({
    revealAsset,
    deleteAsset,
  }), [deleteAsset, revealAsset])

  const setAssetActiveTake = useCallback((assetId: string, takeIndex: number) => {
    actions.setAssetActiveTake(assetId, takeIndex)
  }, [actions])

  const addClipToTimeline = useCallback((asset: Asset, trackIndex = 0, startTime?: number) => {
    actions.insertAssetsToTimeline({ assets: [asset], trackIndex, startTime })
  }, [actions])

  const createAssetFromTake = useCallback((asset: Asset, take: NonNullable<Asset['takes']>[number]): Asset => ({
    id: `asset-${Date.now()}-${Math.random().toString(36).slice(2, 11)}`,
    type: asset.type,
    path: take.path,
    bigThumbnailPath: take.bigThumbnailPath,
    smallThumbnailPath: take.smallThumbnailPath,
    width: take.width,
    height: take.height,
    prompt: asset.prompt,
    resolution: asset.resolution,
    duration: asset.duration,
    generationParams: asset.generationParams,
    takes: [{
      path: take.path,
      bigThumbnailPath: take.bigThumbnailPath,
      smallThumbnailPath: take.smallThumbnailPath,
      width: take.width,
      height: take.height,
      createdAt: take.createdAt,
    }],
    activeTakeIndex: 0,
    createdAt: Date.now(),
  }), [])

  useEffect(() => {
    if (!takesViewAssetId) return
    const takesAsset = assets.find(a => a.id === takesViewAssetId)
    if (!takesAsset || !takesAsset.takes || takesAsset.takes.length <= 1) {
      setTakesViewAssetId(null)
    }
  }, [assets, takesViewAssetId])

  useEffect(() => {
    if (!creatingBin) return
    setTimeout(() => newBinInputRef.current?.focus(), 0)
  }, [creatingBin])

  useEffect(() => {
    if (!assetContextMenu) return
    const handler = () => setAssetContextMenu(null)
    window.addEventListener('click', handler)
    return () => window.removeEventListener('click', handler)
  }, [assetContextMenu])

  useEffect(() => {
    if (!assetContextMenu || !assetContextMenuRef.current) return
    const el = assetContextMenuRef.current
    const rect = el.getBoundingClientRect()
    const vw = window.innerWidth
    const vh = window.innerHeight
    let { x, y } = assetContextMenu
    let adjusted = false
    if (rect.right > vw - 8) { x = vw - rect.width - 8; adjusted = true }
    if (rect.bottom > vh - 8) { y = vh - rect.height - 8; adjusted = true }
    if (x < 8) { x = 8; adjusted = true }
    if (y < 8) { y = 8; adjusted = true }
    if (adjusted) {
      el.style.left = `${x}px`
      el.style.top = `${y}px`
    }
  }, [assetContextMenu])

  useEffect(() => {
    if (!takeContextMenu) return
    const handler = () => setTakeContextMenu(null)
    window.addEventListener('click', handler)
    return () => window.removeEventListener('click', handler)
  }, [takeContextMenu])

  useEffect(() => {
    if (!takeContextMenu || !takeContextMenuRef.current) return
    const el = takeContextMenuRef.current
    const rect = el.getBoundingClientRect()
    const vw = window.innerWidth
    const vh = window.innerHeight
    let { x, y } = takeContextMenu
    let adjusted = false
    if (rect.right > vw - 8) { x = vw - rect.width - 8; adjusted = true }
    if (rect.bottom > vh - 8) { y = vh - rect.height - 8; adjusted = true }
    if (x < 8) { x = 8; adjusted = true }
    if (y < 8) { y = 8; adjusted = true }
    if (adjusted) {
      el.style.left = `${x}px`
      el.style.top = `${y}px`
    }
  }, [takeContextMenu])

  useEffect(() => {
    if (!binContextMenu) return
    const handler = () => setBinContextMenu(null)
    window.addEventListener('click', handler)
    return () => window.removeEventListener('click', handler)
  }, [binContextMenu])

  useEffect(() => {
    if (!binContextMenu || !binContextMenuRef.current) return
    const el = binContextMenuRef.current
    const rect = el.getBoundingClientRect()
    const vw = window.innerWidth
    const vh = window.innerHeight
    let { x, y } = binContextMenu
    let adjusted = false
    if (rect.right > vw - 8) { x = vw - rect.width - 8; adjusted = true }
    if (rect.bottom > vh - 8) { y = vh - rect.height - 8; adjusted = true }
    if (x < 8) { x = 8; adjusted = true }
    if (y < 8) { y = 8; adjusted = true }
    if (adjusted) {
      el.style.left = `${x}px`
      el.style.top = `${y}px`
    }
  }, [binContextMenu])

  const toggleSort = (col: typeof listSortCol) => {
    if (listSortCol === col) {
      setListSortDir(d => d === 'asc' ? 'desc' : 'asc')
    } else {
      setListSortCol(col)
      setListSortDir('asc')
    }
  }

  return (
    <div className="flex flex-col min-h-0 h-full border-r border-zinc-800">
      <div className="p-4 pb-2 space-y-2 flex-shrink-0">
        {!takesViewAssetId ? (
          <>
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-semibold text-white">Assets</h3>
              <div className="flex items-center gap-1">
                <Tooltip content="Create bin" side="right">
                  <button
                    onClick={() => setCreatingBin(true)}
                    className="p-1.5 rounded-lg hover:bg-zinc-800 text-zinc-400 hover:text-white transition-colors"
                  >
                    <FolderPlus className="h-4 w-4" />
                  </button>
                </Tooltip>
                <Tooltip content="Import media" side="right">
                  <button
                    onClick={() => fileInputRef.current?.click()}
                    className="p-1.5 rounded-lg hover:bg-zinc-800 text-zinc-400 hover:text-white transition-colors"
                  >
                    <Upload className="h-4 w-4" />
                  </button>
                </Tooltip>
              </div>
            </div>

            <div className="flex items-center gap-1.5">
              <div className="flex gap-1 bg-zinc-900 rounded-lg p-0.5 flex-1">
                {(['all', 'video', 'image', 'audio'] as const).map(filter => (
                  <button
                    key={filter}
                    onClick={() => setAssetFilter(filter)}
                    className={`flex-1 px-1.5 py-0.5 rounded text-[10px] font-medium transition-colors ${
                      assetFilter === filter
                        ? 'bg-zinc-800 text-white'
                        : 'text-zinc-500 hover:text-zinc-300'
                    }`}
                  >
                    {filter.charAt(0).toUpperCase() + filter.slice(1)}
                  </button>
                ))}
              </div>
              <div className="flex bg-zinc-900 rounded-lg p-0.5">
                <Tooltip content="Grid view" side="right">
                  <button
                    onClick={() => setAssetViewMode('grid')}
                    className={`p-1 rounded transition-colors ${assetViewMode === 'grid' ? 'bg-zinc-800 text-white' : 'text-zinc-500 hover:text-zinc-300'}`}
                  >
                    <LayoutGrid className="h-3 w-3" />
                  </button>
                </Tooltip>
                <Tooltip content="List view" side="right">
                  <button
                    onClick={() => setAssetViewMode('list')}
                    className={`p-1 rounded transition-colors ${assetViewMode === 'list' ? 'bg-zinc-800 text-white' : 'text-zinc-500 hover:text-zinc-300'}`}
                  >
                    <List className="h-3 w-3" />
                  </button>
                </Tooltip>
              </div>
            </div>

            {(bins.length > 0 || creatingBin) && (
              <div className="flex flex-wrap gap-1">
                <button
                  onClick={() => setSelectedBin(null)}
                  className={`px-2 py-0.5 rounded text-[10px] font-medium transition-colors flex items-center gap-1 ${
                    selectedBin === null
                      ? 'bg-blue-600/30 text-blue-300 border border-blue-500/40'
                      : 'bg-zinc-800 text-zinc-500 hover:text-zinc-300 border border-transparent'
                  }`}
                >
                  All
                </button>
                {bins.map(bin => (
                  <button
                    key={bin}
                    onClick={() => setSelectedBin(selectedBin === bin ? null : bin)}
                    onContextMenu={(e) => {
                      e.preventDefault()
                      e.stopPropagation()
                      setBinContextMenu({ bin, x: e.clientX, y: e.clientY })
                    }}
                    onDragOver={(e) => {
                      e.preventDefault()
                      e.currentTarget.classList.add('ring-2', 'ring-blue-400')
                    }}
                    onDragLeave={(e) => {
                      e.currentTarget.classList.remove('ring-2', 'ring-blue-400')
                    }}
                    onDrop={(e) => {
                      e.preventDefault()
                      e.currentTarget.classList.remove('ring-2', 'ring-blue-400')
                      const assetIdsJson = e.dataTransfer.getData('assetIds')
                      if (assetIdsJson) {
                        try {
                          const ids: string[] = JSON.parse(assetIdsJson)
                          actions.assignAssetsToBin(ids, bin)
                          setSelectedAssetIds(new Set())
                        } catch {
                          // ignore parse errors
                        }
                      } else {
                        const assetId = e.dataTransfer.getData('assetId')
                        if (assetId) actions.assignAssetsToBin([assetId], bin)
                      }
                    }}
                    className={`px-2 py-0.5 rounded text-[10px] font-medium transition-colors flex items-center gap-1 group/bin ${
                      selectedBin === bin
                        ? 'bg-blue-600/30 text-blue-300 border border-blue-500/40'
                        : 'bg-zinc-800 text-zinc-500 hover:text-zinc-300 border border-transparent'
                    }`}
                  >
                    <Folder className="h-3 w-3" />
                    {bin}
                    <span className="text-zinc-600 text-[9px]">
                      {assets.filter(a => a.bin === bin).length}
                    </span>
                  </button>
                ))}
                {creatingBin && (
                  <div className="flex items-center gap-1">
                    <input
                      ref={newBinInputRef}
                      type="text"
                      value={newBinName}
                      onChange={(e) => setNewBinName(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' && newBinName.trim()) {
                          if (selectedAssetIds.size > 0) {
                            const binName = newBinName.trim()
                            actions.assignAssetsToBin([...selectedAssetIds], binName)
                            setSelectedAssetIds(new Set())
                          }
                          setCreatingBin(false)
                          setNewBinName('')
                        }
                        if (e.key === 'Escape') {
                          setCreatingBin(false)
                          setNewBinName('')
                        }
                      }}
                      onBlur={() => {
                        if (newBinName.trim() && selectedAssetIds.size > 0) {
                          const binName = newBinName.trim()
                          actions.assignAssetsToBin([...selectedAssetIds], binName)
                          setSelectedAssetIds(new Set())
                        }
                        setCreatingBin(false)
                        setNewBinName('')
                      }}
                      placeholder="Bin name..."
                      className="w-20 px-1.5 py-0.5 rounded text-[10px] bg-zinc-800 border border-zinc-600 text-white placeholder-zinc-600 focus:outline-none focus:border-blue-500"
                    />
                  </div>
                )}
              </div>
            )}

            <input
              ref={fileInputRef}
              type="file"
              accept="video/*,audio/*,image/*"
              multiple
              onChange={handleImportFile}
              className="hidden"
            />
          </>
        ) : (
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-white">Takes</h3>
          </div>
        )}
      </div>

      {takesViewAssetId ? (
        (() => {
          const takesAsset = assets.find(asset => asset.id === takesViewAssetId)
          if (!takesAsset?.takes || takesAsset.takes.length <= 1) return null
          const takes = takesAsset.takes
          return (
            <div className="flex-1 overflow-auto p-3 pt-0">
              <div className="flex items-center gap-2 mb-3">
                <Tooltip content="Back to assets" side="right">
                  <button
                    onClick={() => setTakesViewAssetId(null)}
                    className="p-1 rounded-lg hover:bg-zinc-800 text-zinc-400 hover:text-white transition-colors"
                  >
                    <ChevronLeft className="h-4 w-4" />
                  </button>
                </Tooltip>
                <div className="flex-1 min-w-0">
                  <p className="text-xs font-medium text-white truncate">
                    {takesAsset.prompt?.slice(0, 40) || 'Asset'}{(takesAsset.prompt?.length ?? 0) > 40 ? '...' : ''}
                  </p>
                  <p className="text-[10px] text-zinc-500">
                    {takes.length} takes
                  </p>
                </div>
                {takesAsset.generationParams && (
                  isRegenerating && regeneratingAssetId === takesAsset.id ? (
                    <button
                      onClick={() => handleCancelRegeneration()}
                      className="px-2 py-1 rounded-lg bg-red-900/20 text-red-400 hover:bg-red-900/40 transition-colors text-[10px] font-medium flex items-center gap-1 border border-red-500/30"
                    >
                      <X className="h-3 w-3" />
                      Cancel
                    </button>
                  ) : (
                    <button
                      onClick={() => handleRegenerate(takesAsset.id)}
                      disabled={isRegenerating}
                      className="px-2 py-1 rounded-lg bg-blue-600/20 text-blue-300 hover:bg-blue-600/40 transition-colors text-[10px] font-medium flex items-center gap-1 disabled:opacity-50"
                    >
                      <RefreshCw className="h-3 w-3" />
                      New Take
                    </button>
                  )
                )}
              </div>

              <div className="grid grid-cols-2 gap-2">
                {takes.map((take, idx) => {
                  const isActive = (takesAsset.activeTakeIndex ?? 0) === idx
                  return (
                    <div
                      key={idx}
                      className={`relative group rounded-lg overflow-hidden border-2 transition-all cursor-pointer ${
                        isActive
                          ? 'border-blue-500 ring-2 ring-blue-500/40 shadow-lg shadow-blue-500/20'
                          : 'border-zinc-800 hover:border-zinc-600'
                      }`}
                      onClick={() => {
                        setAssetActiveTake(takesAsset.id, idx)
                      }}
                      onDoubleClick={() => {
                        setAssetActiveTake(takesAsset.id, idx)
                        openSourceAsset({
                          ...takesAsset,
                          path: take.path,
                          bigThumbnailPath: take.bigThumbnailPath,
                          smallThumbnailPath: take.smallThumbnailPath,
                        })
                      }}
                      onContextMenu={(e) => {
                        e.preventDefault()
                        e.stopPropagation()
                        setTakeContextMenu({ assetId: takesAsset.id, takeIndex: idx, x: e.clientX, y: e.clientY })
                      }}
                    >
                      {takesAsset.type === 'video' ? (
                        <VideoThumbnailCard
                          videoUrl={pathToFileUrl(take.path)}
                          thumbnailUrl={take.smallThumbnailPath ? pathToFileUrl(take.smallThumbnailPath) : undefined}
                        />
                      ) : (
                        take.smallThumbnailPath ? (
                          <img src={pathToFileUrl(take.smallThumbnailPath)} alt="" className="w-full aspect-video object-cover" />
                        ) : (
                          <div className="w-full aspect-video bg-zinc-800" />
                        )
                      )}

                      {isActive && <div className="absolute inset-0 bg-blue-600/15 pointer-events-none" />}

                      <div className="absolute bottom-1 left-1 flex items-center gap-1.5">
                        <span className={`px-1.5 py-0.5 rounded text-[10px] font-medium ${
                          isActive ? 'bg-blue-500 text-white' : 'bg-black/70 text-zinc-300'
                        }`}>
                          Take {idx + 1}
                        </span>
                      </div>

                      {isActive && (
                        <div className="absolute top-1.5 left-1.5 px-1.5 py-0.5 rounded bg-blue-500 text-white text-[9px] font-semibold">
                          Active
                        </div>
                      )}

                      <div className="absolute bottom-1 right-1 px-1 py-0.5 rounded bg-black/70 text-[9px] text-zinc-400 opacity-0 group-hover:opacity-100 transition-opacity">
                        {new Date(take.createdAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                      </div>

                      {takes.length > 1 && (
                      <Tooltip content="Delete take" side="right">
                        <button
                          className="absolute top-1.5 right-1.5 p-1 rounded-md bg-black/70 text-zinc-400 hover:text-red-400 hover:bg-red-900/60 opacity-0 group-hover:opacity-100 transition-all z-10"
                          onClick={(e) => {
                            e.stopPropagation()
                            if (confirm(`Delete take ${idx + 1}?`)) {
                              actions.deleteAssetTake(takesAsset.id, idx)
                            }
                          }}
                        >
                            <Trash2 className="h-3 w-3" />
                          </button>
                        </Tooltip>
                      )}

                      {isRegenerating && regeneratingAssetId === takesAsset.id && idx === takes.length - 1 && (
                        <div className="absolute inset-0 bg-blue-900/40 backdrop-blur-sm flex flex-col items-center justify-center z-20">
                          <Loader2 className="h-5 w-5 text-blue-300 animate-spin mb-1" />
                          <span className="text-[9px] text-blue-200 font-medium">{regenProgress}%</span>
                          <span className="text-[8px] text-blue-300/70 mb-1.5">{regenStatusMessage}</span>
                          <button
                            onClick={(e) => { e.stopPropagation(); handleCancelRegeneration() }}
                            className="px-2 py-0.5 rounded bg-zinc-800/80 border border-zinc-600/60 text-[9px] text-zinc-300 hover:text-red-400 hover:border-red-500/50 hover:bg-red-900/30 transition-colors"
                          >
                            Cancel
                          </button>
                        </div>
                      )}
                    </div>
                  )
                })}
              </div>
            </div>
          )
        })()
      ) : (
        <div
          className="flex-1 overflow-auto p-3 pt-0 relative select-none"
          ref={assetGridRef}
          onMouseDown={(e) => {
            if ((e.target as HTMLElement).closest('[data-asset-card]')) return
            if (e.button !== 0) return
            const rect = assetGridRef.current?.getBoundingClientRect()
            if (!rect) return
            const scrollTop = assetGridRef.current?.scrollTop || 0
            const x = e.clientX - rect.left
            const y = e.clientY - rect.top + scrollTop
            setAssetLasso({ startX: x, startY: y, currentX: x, currentY: y })
            if (!e.ctrlKey && !e.metaKey && !e.shiftKey) setSelectedAssetIds(new Set())
          }}
          onMouseMove={(e) => {
            if (!assetLasso || !assetGridRef.current) return
            const rect = assetGridRef.current.getBoundingClientRect()
            const scrollTop = assetGridRef.current.scrollTop || 0
            const x = e.clientX - rect.left
            const y = e.clientY - rect.top + scrollTop
            setAssetLasso(prev => (prev ? { ...prev, currentX: x, currentY: y } : null))
            const lassoLeft = Math.min(assetLasso.startX, x)
            const lassoRight = Math.max(assetLasso.startX, x)
            const lassoTop = Math.min(assetLasso.startY, y)
            const lassoBottom = Math.max(assetLasso.startY, y)
            const newSelected = new Set<string>(e.ctrlKey || e.metaKey || e.shiftKey ? selectedAssetIds : [])
            const cards = assetGridRef.current.querySelectorAll('[data-asset-card]')
            cards.forEach(card => {
              const cardRect = card.getBoundingClientRect()
              const cardLeft = cardRect.left - rect.left
              const cardRight = cardRect.right - rect.left
              const cardTop = cardRect.top - rect.top + scrollTop
              const cardBottom = cardRect.bottom - rect.top + scrollTop
              if (cardLeft < lassoRight && cardRight > lassoLeft && cardTop < lassoBottom && cardBottom > lassoTop) {
                const id = (card as HTMLElement).dataset.assetId
                if (id) newSelected.add(id)
              }
            })
            setSelectedAssetIds(newSelected)
          }}
          onMouseUp={() => setAssetLasso(null)}
          onMouseLeave={() => setAssetLasso(null)}
        >
          {assetLasso && (() => {
            const left = Math.min(assetLasso.startX, assetLasso.currentX)
            const top = Math.min(assetLasso.startY, assetLasso.currentY)
            const width = Math.abs(assetLasso.currentX - assetLasso.startX)
            const height = Math.abs(assetLasso.currentY - assetLasso.startY)
            if (width < 3 && height < 3) return null
            return (
              <div
                className="absolute border border-blue-400 bg-blue-500/15 rounded-sm pointer-events-none z-30"
                style={{ left, top, width, height }}
              />
            )
          })()}

          {filteredAssets.length === 0 ? (
            <div className="text-center py-8">
              <p className="text-sm text-zinc-500">No assets yet</p>
              <p className="text-xs text-zinc-600 mt-1">Generate in Gen Space or import</p>
              <button
                onClick={() => fileInputRef.current?.click()}
                className="mt-3 px-3 py-1.5 rounded-lg bg-zinc-800 text-zinc-300 text-xs hover:bg-zinc-700 transition-colors"
              >
                Import Media
              </button>
            </div>
          ) : assetViewMode === 'grid' ? (
            <div className="grid grid-cols-2 gap-2">
              {filteredAssets.map(asset => {
                const cl = getColorLabel(asset.colorLabel)
                return (
                  <div
                    key={asset.id}
                    data-asset-card
                    data-asset-id={asset.id}
                    className={`relative group cursor-pointer rounded-lg overflow-hidden border-2 transition-all ${
                      selectedAssetIds.has(asset.id)
                        ? 'border-blue-500 ring-2 ring-blue-500/40 shadow-lg shadow-blue-500/20'
                        : 'border-zinc-800 hover:border-zinc-600'
                    }`}
                    draggable
                    onDragStart={(e) => {
                      if (selectedAssetIds.size > 0 && selectedAssetIds.has(asset.id)) {
                        e.dataTransfer.setData('assetIds', JSON.stringify([...selectedAssetIds]))
                      } else {
                        e.dataTransfer.setData('assetId', asset.id)
                      }
                      e.dataTransfer.setData('asset', JSON.stringify(asset))
                      e.dataTransfer.effectAllowed = 'copy'
                    }}
                    onClick={(e) => {
                      e.stopPropagation()
                      if (e.ctrlKey || e.metaKey) {
                        setSelectedAssetIds(prev => {
                          const next = new Set(prev)
                          if (next.has(asset.id)) next.delete(asset.id)
                          else next.add(asset.id)
                          return next
                        })
                      } else if (e.shiftKey && selectedAssetIds.size > 0) {
                        const lastId = [...selectedAssetIds].pop()
                        const lastIdx = filteredAssets.findIndex(a => a.id === lastId)
                        const thisIdx = filteredAssets.findIndex(a => a.id === asset.id)
                        if (lastIdx >= 0 && thisIdx >= 0) {
                          const start = Math.min(lastIdx, thisIdx)
                          const end = Math.max(lastIdx, thisIdx)
                          const next = new Set(selectedAssetIds)
                          for (let i = start; i <= end; i++) next.add(filteredAssets[i].id)
                          setSelectedAssetIds(next)
                        }
                      } else if (selectedAssetIds.has(asset.id) && selectedAssetIds.size === 1) {
                        setSelectedAssetIds(new Set())
                      } else {
                        setSelectedAssetIds(new Set([asset.id]))
                      }
                    }}
                    onDoubleClick={(e) => {
                      e.stopPropagation()
                      openSourceAsset(asset)
                    }}
                    onContextMenu={(e) => {
                      e.preventDefault()
                      e.stopPropagation()
                      if (!selectedAssetIds.has(asset.id)) {
                        setSelectedAssetIds(new Set([asset.id]))
                      }
                      setAssetContextMenu({ assetId: asset.id, x: e.clientX, y: e.clientY })
                    }}
                  >
                    {cl && (
                      <>
                        <div className="absolute top-0 left-0 right-0 h-[3px] z-10" style={{ backgroundColor: cl.color }} />
                        <div className="absolute top-0 left-0 bottom-0 w-[3px] z-10" style={{ backgroundColor: cl.color }} />
                      </>
                    )}
                    {asset.type === 'video' ? (
                      <VideoThumbnailCard
                        videoUrl={pathToFileUrl(asset.path)}
                        thumbnailUrl={asset.smallThumbnailPath ? pathToFileUrl(asset.smallThumbnailPath) : undefined}
                      />
                    ) : asset.type === 'audio' ? (
                      <div className="w-full aspect-video bg-gradient-to-br from-emerald-900/60 to-zinc-900 flex flex-col items-center justify-center gap-1.5">
                        <Music className="h-6 w-6 text-emerald-400" />
                        <div className="flex items-center gap-0.5">
                          {[3, 5, 8, 6, 9, 4, 7, 5, 3, 6, 8, 4].map((h, i) => (
                            <div
                              key={i}
                              className="w-0.5 rounded-full bg-emerald-500/60"
                              style={{ height: `${h * 1.5}px` }}
                            />
                          ))}
                        </div>
                        <p className="text-[9px] text-emerald-300/70 truncate max-w-[90%] px-1">
                          {asset.path || 'Audio'}
                        </p>
                      </div>
                    ) : asset.type === 'adjustment' ? (
                      <div className="w-full aspect-video bg-gradient-to-br from-blue-900/40 to-zinc-900 flex flex-col items-center justify-center gap-1.5 border border-dashed border-blue-500/30">
                        <Layers className="h-6 w-6 text-blue-400" />
                        <p className="text-[9px] text-blue-300/70 font-medium">Adjustment Layer</p>
                      </div>
                    ) : (
                      asset.smallThumbnailPath ? (
                        <img src={pathToFileUrl(asset.smallThumbnailPath)} alt="" className="w-full aspect-video object-cover" />
                      ) : (
                        <div className="w-full aspect-video bg-zinc-800" />
                      )
                    )}
                    {selectedAssetIds.has(asset.id) && <div className="absolute inset-0 bg-blue-600/25 pointer-events-none z-[1]" />}
                    {!selectedAssetIds.has(asset.id) && (
                      <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center pointer-events-none" />
                    )}
                    <div className="absolute top-1 right-1 flex items-center gap-0.5 opacity-0 group-hover:opacity-100 transition-all z-10">
                      {asset.generationParams && (
                        <Tooltip content="Regenerate" side="right">
                          <button
                            onClick={(e) => {
                              e.stopPropagation()
                              handleRegenerate(asset.id)
                            }}
                            disabled={isRegenerating}
                            className={`p-1 rounded bg-black/70 transition-colors ${
                              isRegenerating && regeneratingAssetId === asset.id
                                ? 'text-blue-400 animate-spin'
                                : 'text-zinc-400 hover:text-blue-400 hover:bg-blue-900/50'
                            }`}
                          >
                            <RefreshCw className="h-3 w-3" />
                          </button>
                        </Tooltip>
                      )}
                      <Tooltip content="Delete asset" side="right">
                        <button
                          onClick={(e) => {
                            e.stopPropagation()
                            deleteAsset(asset.id)
                          }}
                          className="p-1 rounded bg-black/70 text-zinc-500 hover:text-red-400 hover:bg-red-900/50 transition-colors"
                        >
                          <X className="h-3 w-3" />
                        </button>
                      </Tooltip>
                    </div>
                    {isRegenerating && regeneratingAssetId === asset.id && (
                      <div className="absolute inset-0 bg-blue-900/40 backdrop-blur-sm flex flex-col items-center justify-center z-20">
                        <Loader2 className="h-5 w-5 text-blue-300 animate-spin mb-1" />
                        <span className="text-[9px] text-blue-200 font-medium">{regenProgress}%</span>
                        <span className="text-[8px] text-blue-300/70 mb-1.5">{regenStatusMessage}</span>
                        <button
                          onClick={(e) => { e.stopPropagation(); handleCancelRegeneration() }}
                          className="px-2 py-0.5 rounded bg-zinc-800/80 border border-zinc-600/60 text-[9px] text-zinc-300 hover:text-red-400 hover:border-red-500/50 hover:bg-red-900/30 transition-colors"
                        >
                          Cancel
                        </button>
                      </div>
                    )}
                    {asset.takes && asset.takes.length > 1 && (
                      <div className="absolute bottom-1 right-1 flex items-center gap-0.5 rounded bg-black/80 z-10">
                        <Tooltip content="Previous take" side="right">
                          <button
                            onClick={(e) => {
                              e.stopPropagation()
                              const idx = Math.max(0, (asset.activeTakeIndex ?? 0) - 1)
                              setAssetActiveTake(asset.id, idx)
                            }}
                            disabled={(asset.activeTakeIndex ?? 0) === 0}
                            className="p-0.5 text-blue-300 hover:text-white disabled:text-zinc-600 transition-colors"
                          >
                            <ChevronLeft className="h-3 w-3" />
                          </button>
                        </Tooltip>
                        <button
                          onClick={(e) => {
                            e.stopPropagation()
                            setTakesViewAssetId(asset.id)
                            setSelectedAssetIds(new Set())
                          }}
                          className="px-0.5 cursor-pointer hover:text-white transition-colors flex items-center gap-1"
                          title="View all takes"
                        >
                          <Layers className="h-2.5 w-2.5 text-blue-400" />
                          <span className="text-[9px] text-blue-300 font-medium">
                            {(asset.activeTakeIndex ?? 0) + 1}/{asset.takes.length}
                          </span>
                        </button>
                        <Tooltip content="Next take" side="right">
                          <button
                            onClick={(e) => {
                              e.stopPropagation()
                              if (asset.takes) {
                                const idx = Math.min(asset.takes.length - 1, (asset.activeTakeIndex ?? 0) + 1)
                                setAssetActiveTake(asset.id, idx)
                              }
                            }}
                            disabled={asset.takes && (asset.activeTakeIndex ?? 0) >= asset.takes.length - 1}
                            className="p-0.5 text-blue-300 hover:text-white disabled:text-zinc-600 transition-colors"
                          >
                            <ChevronRight className="h-3 w-3" />
                          </button>
                        </Tooltip>
                      </div>
                    )}
                    {asset.bin && (
                      <div className="absolute top-1.5 left-8 flex items-center gap-0.5 px-1 py-0.5 rounded bg-black/70 text-[9px] text-blue-300 opacity-0 group-hover:opacity-100 transition-opacity z-10">
                        <Folder className="h-2.5 w-2.5" />
                        {asset.bin}
                      </div>
                    )}
                    <div className="absolute bottom-1 left-1 flex items-center gap-1 px-1.5 py-0.5 rounded bg-black/70 text-[10px] text-white">
                      {asset.type === 'video' ? <Video className="h-3 w-3" /> : asset.type === 'audio' ? <Music className="h-3 w-3" /> : asset.type === 'adjustment' ? <Layers className="h-3 w-3" /> : <Image className="h-3 w-3" />}
                      {asset.type === 'adjustment' ? 'Adj' : asset.duration ? `${asset.duration.toFixed(1)}s` : ''}
                    </div>
                  </div>
                )
              })}
            </div>
          ) : (
            <div className="flex flex-col">
              <div className="flex items-center gap-1 px-2 py-1 border-b border-zinc-800 bg-zinc-900/80 sticky top-0 z-10">
                <div className="w-2 flex-shrink-0" />
                <div className="w-8 flex-shrink-0" />
                {([
                  { col: 'name' as const, label: 'Name', flex: 'flex-1 min-w-0' },
                  { col: 'type' as const, label: 'Type', flex: 'w-14 flex-shrink-0 text-center' },
                  { col: 'duration' as const, label: 'Duration', flex: 'w-16 flex-shrink-0 text-right' },
                  { col: 'resolution' as const, label: 'Res', flex: 'w-14 flex-shrink-0 text-right' },
                  { col: 'date' as const, label: 'Date', flex: 'w-16 flex-shrink-0 text-right' },
                  { col: 'color' as const, label: 'Color', flex: 'w-10 flex-shrink-0 text-center' },
                ]).map(({ col, label, flex }) => (
                  <button
                    key={col}
                    onClick={() => toggleSort(col)}
                    className={`${flex} flex items-center gap-0.5 text-[9px] font-semibold uppercase tracking-wider transition-colors cursor-pointer select-none ${
                      listSortCol === col ? 'text-blue-400' : 'text-zinc-500 hover:text-zinc-300'
                    }`}
                  >
                    <span className="truncate">{label}</span>
                    {listSortCol === col ? (
                      listSortDir === 'asc' ? <ChevronUp className="h-2.5 w-2.5 flex-shrink-0" /> : <ChevronDown className="h-2.5 w-2.5 flex-shrink-0" />
                    ) : (
                      <ArrowUpDown className="h-2.5 w-2.5 flex-shrink-0 opacity-0 group-hover:opacity-50" />
                    )}
                  </button>
                ))}
                <div className="w-6 flex-shrink-0" />
              </div>
              {visibleAssets.map(asset => {
                const cl = getColorLabel(asset.colorLabel)
                const name = asset.path ? asset.path.split(/[/\\]/).pop() || asset.path : asset.type === 'adjustment' ? 'Adjustment Layer' : asset.type.charAt(0).toUpperCase() + asset.type.slice(1)
                const dateStr = new Date(asset.createdAt).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
                return (
                  <div
                    key={asset.id}
                    data-asset-card
                    data-asset-id={asset.id}
                    className={`group flex items-center gap-1 px-2 py-1 cursor-pointer transition-all ${
                      selectedAssetIds.has(asset.id)
                        ? 'bg-blue-600/20 ring-1 ring-blue-500/50'
                        : 'hover:bg-zinc-800/60'
                    }`}
                    draggable
                    onDragStart={(e) => {
                      if (selectedAssetIds.size > 0 && selectedAssetIds.has(asset.id)) {
                        e.dataTransfer.setData('assetIds', JSON.stringify([...selectedAssetIds]))
                      } else {
                        e.dataTransfer.setData('assetId', asset.id)
                      }
                      e.dataTransfer.setData('asset', JSON.stringify(asset))
                      e.dataTransfer.effectAllowed = 'copy'
                    }}
                    onClick={(e) => {
                      e.stopPropagation()
                      if (e.ctrlKey || e.metaKey) {
                        setSelectedAssetIds(prev => {
                          const next = new Set(prev)
                          if (next.has(asset.id)) next.delete(asset.id)
                          else next.add(asset.id)
                          return next
                        })
                      } else if (e.shiftKey && selectedAssetIds.size > 0) {
                        const lastId = [...selectedAssetIds].pop()
                        const lastIdx = filteredAssets.findIndex(a => a.id === lastId)
                        const thisIdx = filteredAssets.findIndex(a => a.id === asset.id)
                        if (lastIdx >= 0 && thisIdx >= 0) {
                          const start = Math.min(lastIdx, thisIdx)
                          const end = Math.max(lastIdx, thisIdx)
                          const next = new Set(selectedAssetIds)
                          for (let i = start; i <= end; i++) next.add(filteredAssets[i].id)
                          setSelectedAssetIds(next)
                        }
                      } else if (selectedAssetIds.has(asset.id) && selectedAssetIds.size === 1) {
                        setSelectedAssetIds(new Set())
                      } else {
                        setSelectedAssetIds(new Set([asset.id]))
                      }
                    }}
                    onDoubleClick={(e) => {
                      e.stopPropagation()
                      openSourceAsset(asset)
                    }}
                    onContextMenu={(e) => {
                      e.preventDefault()
                      e.stopPropagation()
                      if (!selectedAssetIds.has(asset.id)) {
                        setSelectedAssetIds(new Set([asset.id]))
                      }
                      setAssetContextMenu({ assetId: asset.id, x: e.clientX, y: e.clientY })
                    }}
                  >
                    {cl ? (
                      <div className="w-2 h-2 rounded-full flex-shrink-0" style={{ backgroundColor: cl.color }} />
                    ) : (
                      <div className="w-2 flex-shrink-0" />
                    )}
                    <div className="w-8 h-6 flex-shrink-0 rounded overflow-hidden bg-zinc-800">
                      {asset.type === 'video' ? (
                        asset.smallThumbnailPath ? (
                          <img src={pathToFileUrl(asset.smallThumbnailPath)} alt="" className="w-full h-full object-cover" />
                        ) : (
                          <div className="w-full h-full bg-zinc-800" />
                        )
                      ) : asset.type === 'audio' ? (
                        <div className="w-full h-full flex items-center justify-center bg-emerald-900/40"><Music className="h-2.5 w-2.5 text-emerald-400" /></div>
                      ) : asset.type === 'adjustment' ? (
                        <div className="w-full h-full flex items-center justify-center bg-blue-900/30"><Layers className="h-2.5 w-2.5 text-blue-400" /></div>
                      ) : (
                        asset.smallThumbnailPath ? (
                          <img src={pathToFileUrl(asset.smallThumbnailPath)} alt="" className="w-full h-full object-cover" />
                        ) : (
                          <div className="w-full h-full bg-zinc-800" />
                        )
                      )}
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-[10px] text-zinc-200 truncate leading-tight">{name}</p>
                      {asset.takes && asset.takes.length > 1 && (
                        <span className="text-[8px] text-blue-400">{asset.takes.length} takes</span>
                      )}
                    </div>
                    <span className="w-14 flex-shrink-0 text-center text-[9px] text-zinc-500 uppercase font-medium">{asset.type}</span>
                    <span className="w-16 flex-shrink-0 text-right text-[9px] text-zinc-500 tabular-nums">
                      {asset.duration != null ? `${asset.duration.toFixed(1)}s` : '—'}
                    </span>
                    <span className="w-14 flex-shrink-0 text-right text-[9px] text-zinc-500">
                      {asset.resolution || '—'}
                    </span>
                    <span className="w-16 flex-shrink-0 text-right text-[9px] text-zinc-500">{dateStr}</span>
                    <div className="w-10 flex-shrink-0 flex items-center justify-center">
                      {cl ? (
                        <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: cl.color }} title={cl.label} />
                      ) : (
                        <span className="text-[9px] text-zinc-600">—</span>
                      )}
                    </div>
                    <Tooltip content="Delete asset" side="right">
                      <button
                        onClick={(e) => {
                          e.stopPropagation()
                          deleteAsset(asset.id)
                        }}
                        className="w-6 flex-shrink-0 flex items-center justify-center p-0.5 rounded text-zinc-600 hover:text-red-400 opacity-0 group-hover:opacity-100 transition-all"
                      >
                        <X className="h-3 w-3" />
                      </button>
                    </Tooltip>
                  </div>
                )
              })}
            </div>
          )}
        </div>
      )}

      {assetContextMenu && (() => {
        const asset = assets.find(a => a.id === assetContextMenu.assetId)
        if (!asset) return null
        const targetIds = selectedAssetIds.size > 0 && selectedAssetIds.has(asset.id) ? [...selectedAssetIds] : [asset.id]
        return (
          <AssetContextMenu
            asset={asset}
            targetIds={targetIds}
            assetContextMenu={assetContextMenu}
            assetContextMenuRef={assetContextMenuRef}
            addClipToTimeline={addClipToTimeline}
            handleRegenerate={handleRegenerate}
            handleCancelRegeneration={handleCancelRegeneration}
            setTakesViewAssetId={setTakesViewAssetId}
            setSelectedAssetIds={setSelectedAssetIds}
            setAssetContextMenu={setAssetContextMenu}
            createAssetFromTake={createAssetFromTake}
          />
        )
      })()}

      {takeContextMenu && (() => {
        const tcAsset = assets.find(a => a.id === takeContextMenu.assetId)
        if (!tcAsset?.takes) return null
        const take = tcAsset.takes[takeContextMenu.takeIndex]
        if (!take) return null
        return (
          <TakeContextMenu
            tcAsset={tcAsset}
            take={take}
            takeIndex={takeContextMenu.takeIndex}
            takeContextMenu={takeContextMenu}
            takeContextMenuRef={takeContextMenuRef}
            addClipToTimeline={addClipToTimeline}
            createAssetFromTake={createAssetFromTake}
            setTakeContextMenu={setTakeContextMenu}
          />
        )
      })()}

      {binContextMenu && (
        <div
          ref={binContextMenuRef}
          className="fixed bg-zinc-800 border border-zinc-700 rounded-xl shadow-2xl py-1.5 z-[60] min-w-[160px] text-xs"
          style={{ left: binContextMenu.x, top: binContextMenu.y }}
          onClick={(e) => e.stopPropagation()}
        >
          <button
            onClick={() => {
              const newName = prompt('Rename bin:', binContextMenu.bin)
              if (newName?.trim() && newName.trim() !== binContextMenu.bin) {
                actions.renameBin(binContextMenu.bin, newName.trim())
                if (selectedBin === binContextMenu.bin) setSelectedBin(newName.trim())
              }
              setBinContextMenu(null)
            }}
            className="w-full text-left px-3 py-1.5 text-zinc-300 hover:bg-zinc-700 flex items-center gap-3"
          >
            <Pencil className="h-3.5 w-3.5 text-zinc-500" />
            <span>Rename Bin</span>
          </button>
          <button
            onClick={() => {
              actions.clearBin(binContextMenu.bin)
              if (selectedBin === binContextMenu.bin) setSelectedBin(null)
              setBinContextMenu(null)
            }}
            className="w-full text-left px-3 py-1.5 text-red-400 hover:bg-zinc-700 flex items-center gap-3"
          >
            <Trash2 className="h-3.5 w-3.5" />
            <span>Delete Bin</span>
          </button>
        </div>
      )}
    </div>
  )
})
