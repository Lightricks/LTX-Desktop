import React, { type RefObject } from 'react'
import { Plus, Copy, Eye, Trash2 } from 'lucide-react'
import type { Asset, AssetTake } from '../../types/project'
import { useEditorActions } from './editor-store'

export interface TakeContextMenuProps {
  tcAsset: Asset
  take: AssetTake
  takeIndex: number
  takeContextMenu: { assetId: string; takeIndex: number; x: number; y: number }
  takeContextMenuRef: RefObject<HTMLDivElement>
  addClipToTimeline: (asset: Asset, trackIndex?: number, startTime?: number) => void
  createAssetFromTake: (asset: Asset, take: AssetTake) => Asset
  setTakeContextMenu: React.Dispatch<React.SetStateAction<{ assetId: string; takeIndex: number; x: number; y: number } | null>>
}

export function TakeContextMenu({
  tcAsset,
  take,
  takeIndex,
  takeContextMenu,
  takeContextMenuRef,
  addClipToTimeline,
  createAssetFromTake,
  setTakeContextMenu,
}: TakeContextMenuProps) {
  const actions = useEditorActions()
  const isActive = (tcAsset.activeTakeIndex ?? 0) === takeIndex

  return (
    <div
      ref={takeContextMenuRef}
      className="fixed bg-zinc-800 border border-zinc-700 rounded-xl shadow-2xl py-1.5 z-[60] min-w-[190px] text-xs"
      style={{ left: takeContextMenu.x, top: takeContextMenu.y }}
      onClick={(e) => e.stopPropagation()}
    >
      <div className="px-3 py-1 text-[10px] text-zinc-500 font-medium">
        Take {takeIndex + 1} of {tcAsset.takes!.length}
      </div>

      {!isActive && (
        <button
          onClick={() => {
            actions.setAssetActiveTake(tcAsset.id, takeIndex)
            setTakeContextMenu(null)
          }}
          className="w-full text-left px-3 py-1.5 text-zinc-300 hover:bg-zinc-700 flex items-center gap-3"
        >
          <Eye className="h-3.5 w-3.5 text-zinc-500" />
          <span>Set as Active Take</span>
        </button>
      )}

      <button
        onClick={() => {
          addClipToTimeline({
            ...tcAsset,
            path: take.path,
            bigThumbnailPath: take.bigThumbnailPath,
            smallThumbnailPath: take.smallThumbnailPath,
          }, 0)
          setTakeContextMenu(null)
        }}
        className="w-full text-left px-3 py-1.5 text-zinc-300 hover:bg-zinc-700 flex items-center gap-3"
      >
        <Plus className="h-3.5 w-3.5 text-zinc-500" />
        <span>Add to Timeline</span>
      </button>

      <div className="h-px bg-zinc-700 my-1" />

      <button
        onClick={() => {
          actions.addAssetToEditor(createAssetFromTake(tcAsset, take))
          setTakeContextMenu(null)
        }}
        className="w-full text-left px-3 py-1.5 text-blue-300 hover:bg-zinc-700 flex items-center gap-3"
      >
        <Copy className="h-3.5 w-3.5" />
        <span>Create New Asset from Take</span>
      </button>

      {tcAsset.takes!.length > 1 && (
        <>
          <div className="h-px bg-zinc-700 my-1" />
          <button
            onClick={() => {
              if (confirm(`Delete take ${takeIndex + 1}?`)) {
                actions.deleteAssetTake(tcAsset.id, takeIndex)
              }
              setTakeContextMenu(null)
            }}
            className="w-full text-left px-3 py-1.5 text-red-400 hover:bg-red-900/30 flex items-center gap-3"
          >
            <Trash2 className="h-3.5 w-3.5" />
            <span>Delete Take</span>
          </button>
        </>
      )}
    </div>
  )
}
