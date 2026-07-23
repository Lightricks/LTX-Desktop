import { LoraInfoPopover } from './LoraInfoPopover'
import type { components } from '../generated/backend-openapi'

type Sections = NonNullable<components['schemas']['LoraCatalogItem']['instructions']>

export interface SelectedLoraItem {
  name: string
  sections?: Sections
  repoId?: string
  isCommunity?: boolean
}

// The "currently selected LoRA(s)" bar shown above the prompt bar — a name + info popover
// per item, inline. Used for both the single active IC-LoRA and the list of plain LoRAs.
export function SelectedLoraInfo({ items }: { items: SelectedLoraItem[] }) {
  if (items.length === 0) return null
  return (
    <div className="mb-2 flex flex-wrap items-center gap-x-4 gap-y-1 rounded-lg border border-zinc-800 bg-zinc-900/80 px-3 py-2 text-[11px] text-zinc-400">
      {items.map((it, i) => (
        <span key={i} className="inline-flex items-center gap-1">
          <span className="font-semibold text-zinc-300">{it.name}</span>
          <LoraInfoPopover
            sections={it.sections ?? []}
            name={it.name}
            repoId={it.repoId}
            isCommunity={it.isCommunity}
          />
        </span>
      ))}
    </div>
  )
}
