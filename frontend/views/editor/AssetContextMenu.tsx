import React, { type RefObject } from 'react'
import {
  Plus, X, RefreshCw, ChevronLeft, ChevronRight, Layers, GitMerge,
  FolderPlus, Folder, Trash2, FolderOpen,
} from 'lucide-react'
import { useShallow } from 'zustand/react/shallow'
import type { Asset } from '../../types/project'
import { COLOR_LABELS } from './video-editor-utils'
import { selectAssetBins, selectAssets, selectRegenerationState } from './editor-selectors'
import { useEditorActions, useEditorStore } from './editor-store'

export interface AssetContextMenuProps {
  asset: Asset
  targetIds: string[]
  assetContextMenu: { assetId: string; x: number; y: number }
  assetContextMenuRef: RefObject<HTMLDivElement>
  addClipToTimeline: (asset: Asset, trackIndex?: number, startTime?: number) => void
  handleRegenerate: (assetId: string) => void
  handleCancelRegeneration: () => void
  setTakesViewAssetId: (assetId: string | null) => void
  setSelectedAssetIds: React.Dispatch<React.SetStateAction<Set<string>>>
  setAssetContextMenu: React.Dispatch<React.SetStateAction<{ assetId: string; x: number; y: number } | null>>
  createAssetFromTake: (asset: Asset, take: NonNullable<Asset['takes']>[number]) => Asset
}

export function AssetContextMenu({
  asset,
  targetIds,
  assetContextMenu,
  assetContextMenuRef,
  addClipToTimeline,
  handleRegenerate,
  handleCancelRegeneration,
  setTakesViewAssetId,
  setSelectedAssetIds,
  setAssetContextMenu,
  createAssetFromTake,
}: AssetContextMenuProps) {
  const actions = useEditorActions()
  const assets = useEditorStore(selectAssets)
  const bins = useEditorStore(useShallow(selectAssetBins))
  const regenerationState = useEditorStore(selectRegenerationState)
  const isRegenerating = regenerationState.regeneratingAssetId !== null || regenerationState.regeneratingClipId !== null
  const regeneratingAssetId = regenerationState.regeneratingAssetId
  const isMulti = targetIds.length > 1

  const closeMenu = () => setAssetContextMenu(null)
  const clearSelection = () => setSelectedAssetIds(new Set())

  const setActiveTake = (assetId: string, takeIndex: number) => {
    actions.setAssetActiveTake(assetId, takeIndex)
  }

  const setColor = (colorLabel?: string) => {
    for (const id of targetIds) {
      actions.setAssetColorLabel(id, colorLabel)
    }
    closeMenu()
  }

  const moveToBin = (bin?: string) => {
    actions.assignAssetsToBin(targetIds, bin)
    clearSelection()
    closeMenu()
  }

  const deleteTargetAssets = () => {
    actions.deleteAssets(targetIds)
    clearSelection()
    closeMenu()
  }

  return (
    <div
      ref={assetContextMenuRef}
      className="fixed bg-zinc-800 border border-zinc-700 rounded-xl shadow-2xl py-1.5 z-[60] min-w-[180px] text-xs"
      style={{ left: assetContextMenu.x, top: assetContextMenu.y }}
      onClick={(e) => e.stopPropagation()}
    >
      {isMulti && (
        <div className="px-3 py-1 text-[10px] text-blue-400 font-medium">
          {targetIds.length} assets selected
        </div>
      )}

      {!isMulti && (
        <button
          onClick={() => {
            addClipToTimeline(asset, 0)
            closeMenu()
          }}
          className="w-full text-left px-3 py-1.5 text-zinc-300 hover:bg-zinc-700 flex items-center gap-3"
        >
          <Plus className="h-3.5 w-3.5 text-zinc-500" />
          <span>Add to Timeline</span>
        </button>
      )}

      {!isMulti && asset.path && (
        <button
          onClick={() => {
            window.electronAPI?.showItemInFolder({ filePath: asset.path! })
            closeMenu()
          }}
          className="w-full text-left px-3 py-1.5 text-zinc-300 hover:bg-zinc-700 flex items-center gap-3"
        >
          <FolderOpen className="h-3.5 w-3.5 text-zinc-500" />
          <span>Show in Explorer</span>
        </button>
      )}

      {!isMulti && asset.generationParams && (
        <>
          {isRegenerating && regeneratingAssetId === asset.id ? (
            <button
              onClick={() => {
                handleCancelRegeneration()
                closeMenu()
              }}
              className="w-full text-left px-3 py-1.5 text-red-400 hover:bg-zinc-700 flex items-center gap-3"
            >
              <X className="h-3.5 w-3.5" />
              <span>Cancel Regeneration</span>
            </button>
          ) : (
            <button
              onClick={() => {
                handleRegenerate(asset.id)
                closeMenu()
              }}
              disabled={isRegenerating}
              className="w-full text-left px-3 py-1.5 text-zinc-300 hover:bg-zinc-700 flex items-center gap-3 disabled:opacity-50"
            >
              <RefreshCw className="h-3.5 w-3.5 text-zinc-500" />
              <span>Regenerate</span>
            </button>
          )}
        </>
      )}

      {!isMulti && asset.takes && asset.takes.length > 1 && (
        <>
          <div className="px-3 py-1.5 flex items-center gap-2">
            <span className="text-[10px] text-zinc-500">Take:</span>
            <button
              onClick={(e) => {
                e.stopPropagation()
                const idx = Math.max(0, (asset.activeTakeIndex ?? 0) - 1)
                setActiveTake(asset.id, idx)
              }}
              disabled={(asset.activeTakeIndex ?? 0) === 0}
              className="p-0.5 rounded hover:bg-zinc-600 text-zinc-400 hover:text-white disabled:text-zinc-600 disabled:hover:bg-transparent"
            >
              <ChevronLeft className="h-3 w-3" />
            </button>
            <span className="text-[10px] text-zinc-300 min-w-[28px] text-center">
              {(asset.activeTakeIndex ?? 0) + 1}/{asset.takes.length}
            </span>
            <button
              onClick={(e) => {
                e.stopPropagation()
                const idx = Math.min(asset.takes!.length - 1, (asset.activeTakeIndex ?? 0) + 1)
                setActiveTake(asset.id, idx)
              }}
              disabled={(asset.activeTakeIndex ?? 0) >= asset.takes.length - 1}
              className="p-0.5 rounded hover:bg-zinc-600 text-zinc-400 hover:text-white disabled:text-zinc-600 disabled:hover:bg-transparent"
            >
              <ChevronRight className="h-3 w-3" />
            </button>
          </div>
          <button
            onClick={() => {
              setTakesViewAssetId(asset.id)
              clearSelection()
              closeMenu()
            }}
            className="w-full text-left px-3 py-1.5 text-zinc-300 hover:bg-zinc-700 flex items-center gap-3"
          >
            <Layers className="h-3.5 w-3.5 text-zinc-500" />
            <span>View All Takes</span>
          </button>
          <button
            onClick={() => {
              if (!asset.takes) return
              const splitAssets = asset.takes.slice(1).map(take => createAssetFromTake(asset, take))
              const firstTake = asset.takes[0]
              if (splitAssets.length > 0) actions.addAssetsToEditor(splitAssets)
              actions.updateAsset(asset.id, {
                takes: [firstTake],
                activeTakeIndex: 0,
                path: firstTake.path,
                bigThumbnailPath: firstTake.bigThumbnailPath,
                smallThumbnailPath: firstTake.smallThumbnailPath,
                width: firstTake.width,
                height: firstTake.height,
              })
              closeMenu()
            }}
            className="w-full text-left px-3 py-1.5 text-zinc-300 hover:bg-zinc-700 flex items-center gap-3"
          >
            <GitMerge className="h-3.5 w-3.5 text-zinc-500 rotate-180" />
            <span>Ungroup Takes</span>
          </button>
          <button
            onClick={() => {
              const activeIdx = asset.activeTakeIndex ?? 0
              if (confirm(`Delete take ${activeIdx + 1}?`)) {
                actions.deleteAssetTake(asset.id, activeIdx)
              }
              closeMenu()
            }}
            className="w-full text-left px-3 py-1.5 text-red-400 hover:bg-red-900/30 flex items-center gap-3"
          >
            <Trash2 className="h-3.5 w-3.5" />
            <span>Delete Active Take</span>
          </button>
        </>
      )}

      <div className="h-px bg-zinc-700 my-1" />

      <div className="px-3 py-1 text-[10px] text-zinc-500 font-semibold uppercase tracking-wider">Label</div>
      <div className="px-3 py-1.5 flex items-center gap-1 flex-wrap">
        <button
          onClick={() => setColor(undefined)}
          className={`w-4 h-4 rounded-full border-2 flex items-center justify-center transition-all ${
            !asset.colorLabel ? 'border-white scale-110' : 'border-zinc-600 hover:border-zinc-400'
          }`}
          title="No label"
        >
          <X className="h-2 w-2 text-zinc-400" />
        </button>
        {COLOR_LABELS.map(cl => (
          <button
            key={cl.id}
            onClick={() => setColor(cl.id)}
            className={`w-4 h-4 rounded-full transition-all ${
              asset.colorLabel === cl.id ? 'ring-2 ring-white ring-offset-1 ring-offset-zinc-800 scale-110' : 'hover:scale-125'
            }`}
            style={{ backgroundColor: cl.color }}
            title={cl.label}
          />
        ))}
      </div>

      <div className="h-px bg-zinc-700 my-1" />

      <div className="px-3 py-1 text-[10px] text-zinc-500 font-semibold uppercase tracking-wider">Move to Bin</div>

      <button
        onClick={() => moveToBin(undefined)}
        className="w-full text-left px-3 py-1.5 text-zinc-300 hover:bg-zinc-700 flex items-center gap-3"
      >
        <X className="h-3.5 w-3.5 text-zinc-500" />
        <span>Remove from Bin</span>
      </button>

      {bins.map(bin => (
        <button
          key={bin}
          onClick={() => moveToBin(bin)}
          className="w-full text-left px-3 py-1.5 text-zinc-300 hover:bg-zinc-700 flex items-center gap-3"
        >
          <Folder className="h-3.5 w-3.5 text-zinc-500" />
          <span>{bin}</span>
        </button>
      ))}

      <button
        onClick={() => {
          const name = prompt('New bin name:')
          if (name?.trim()) moveToBin(name.trim())
        }}
        className="w-full text-left px-3 py-1.5 text-zinc-300 hover:bg-zinc-700 flex items-center gap-3"
      >
        <FolderPlus className="h-3.5 w-3.5 text-zinc-500" />
        <span>New Bin...</span>
      </button>

      {isMulti && (
        <>
          <div className="h-px bg-zinc-700 my-1" />
          <button
            onClick={() => {
              const selectedAssets = assets.filter(a => targetIds.includes(a.id))
              if (selectedAssets.length < 2) return
              const primary = selectedAssets[0]
              const newTakes = selectedAssets.map(a => ({
                path: a.path,
                bigThumbnailPath: a.bigThumbnailPath,
                smallThumbnailPath: a.smallThumbnailPath,
                width: a.width,
                height: a.height,
                createdAt: a.createdAt,
              }))
              actions.updateAsset(primary.id, {
                takes: newTakes,
                activeTakeIndex: 0,
                path: newTakes[0].path,
                bigThumbnailPath: newTakes[0].bigThumbnailPath,
                smallThumbnailPath: newTakes[0].smallThumbnailPath,
                width: newTakes[0].width,
                height: newTakes[0].height,
              })
              actions.deleteAssets(selectedAssets.slice(1).map(a => a.id))
              clearSelection()
              closeMenu()
            }}
            className="w-full text-left px-3 py-1.5 text-blue-300 hover:bg-zinc-700 flex items-center gap-3"
          >
            <GitMerge className="h-3.5 w-3.5" />
            <span>Group as Takes</span>
          </button>
          <button
            onClick={() => {
              clearSelection()
              closeMenu()
            }}
            className="w-full text-left px-3 py-1.5 text-zinc-300 hover:bg-zinc-700 flex items-center gap-3"
          >
            <X className="h-3.5 w-3.5 text-zinc-500" />
            <span>Clear Selection</span>
          </button>
        </>
      )}

      <div className="h-px bg-zinc-700 my-1" />

      <button
        onClick={deleteTargetAssets}
        className="w-full text-left px-3 py-1.5 text-red-400 hover:bg-zinc-700 flex items-center gap-3"
      >
        <Trash2 className="h-3.5 w-3.5" />
        <span>{isMulti ? `Delete ${targetIds.length} Assets` : 'Delete Asset'}</span>
      </button>
    </div>
  )
}
