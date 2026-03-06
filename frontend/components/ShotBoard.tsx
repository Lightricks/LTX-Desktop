import { Plus, GitBranch, Star } from 'lucide-react'
import type { Shot, Asset } from '../types/project'

interface ShotBoardProps {
  shots: Shot[]
  assets: Asset[]
  onCreateShot: () => void
  onSelectShot: (id: string) => void
  selectedShotId: string | null
}

function getShotPrimaryAsset(shotsAssets: Asset[], shot: Shot): Asset | undefined {
  if (shot.referenceAssetId) {
    return shotsAssets.find((a) => a.id === shot.referenceAssetId)
  }
  const canonical = shot.variants.find((v) => v.id === shot.canonicalVariantId)
  if (canonical?.assetId) {
    return shotsAssets.find((a) => a.id === canonical.assetId)
  }
  const firstWithAsset = shot.variants.find((v) => v.assetId)
  if (firstWithAsset?.assetId) {
    return shotsAssets.find((a) => a.id === firstWithAsset.assetId)
  }
  return undefined
}

export function ShotBoard({ shots, assets, onCreateShot, onSelectShot, selectedShotId }: ShotBoardProps) {
  if (shots.length === 0) {
    return (
      <div className="flex items-center justify-between px-4 py-2 border-b border-zinc-800 bg-zinc-950/80">
        <div className="flex items-center gap-2 text-xs text-zinc-400">
          <GitBranch className="h-3.5 w-3.5 text-zinc-500" />
          <span>Storyboard is empty.</span>
          <span className="text-zinc-500">Create your first shot from a generation.</span>
        </div>
        <button
          onClick={onCreateShot}
          className="inline-flex items-center gap-1.5 rounded-md bg-zinc-800 hover:bg-zinc-700 px-2.5 py-1 text-[11px] font-medium text-zinc-100 transition-colors"
        >
          <Plus className="h-3 w-3" />
          New Shot
        </button>
      </div>
    )
  }

  return (
    <div className="flex items-center justify-between px-4 py-2 border-b border-zinc-800 bg-zinc-950/80">
      <div className="flex items-center gap-3 overflow-x-auto scrollbar-thin scrollbar-thumb-zinc-700 scrollbar-track-transparent">
        {shots
          .slice()
          .sort((a, b) => a.order - b.order || a.createdAt - b.createdAt)
          .map((shot, index) => {
            const isSelected = shot.id === selectedShotId
            const primaryAsset = getShotPrimaryAsset(assets, shot)
            const canonical = shot.variants.find((v) => v.id === shot.canonicalVariantId)
            return (
              <button
                key={shot.id}
                onClick={() => onSelectShot(shot.id)}
                className={`flex items-center gap-2 rounded-md border px-2.5 py-1.5 text-left text-[11px] transition-colors ${
                  isSelected
                    ? 'border-blue-500 bg-blue-500/10 text-zinc-50'
                    : 'border-zinc-800 bg-zinc-900/60 text-zinc-300 hover:border-zinc-600 hover:bg-zinc-900'
                }`}
              >
                <div className="flex flex-col">
                  <div className="flex items-center gap-1">
                    <span className="text-[10px] text-zinc-500">#{index + 1}</span>
                    <span className="font-semibold truncate max-w-[160px]">
                      {shot.name || `Shot ${index + 1}`}
                    </span>
                    {canonical && (
                      <Star className="h-3 w-3 text-amber-400" aria-hidden="true" />
                    )}
                  </div>
                  <div className="flex items-center gap-1 text-[10px] text-zinc-500">
                    <span>{shot.variants.length} variant{shot.variants.length !== 1 ? 's' : ''}</span>
                    {primaryAsset && (
                      <>
                        <span className="text-zinc-700">•</span>
                        <span className="truncate max-w-[120px]">{primaryAsset.prompt || primaryAsset.path}</span>
                      </>
                    )}
                  </div>
                </div>
              </button>
            )
          })}
      </div>
      <button
        onClick={onCreateShot}
        className="ml-3 inline-flex items-center gap-1.5 rounded-md bg-zinc-800 hover:bg-zinc-700 px-2.5 py-1 text-[11px] font-medium text-zinc-100 transition-colors flex-shrink-0"
      >
        <Plus className="h-3 w-3" />
        New Shot
      </button>
    </div>
  )
}

