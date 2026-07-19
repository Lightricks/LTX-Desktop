import { useId, useRef } from 'react'
import { Info, ExternalLink } from 'lucide-react'
import type { components } from '../generated/backend-openapi'

type InstructionSection = NonNullable<components['schemas']['LoraCatalogItem']['instructions']>[number]

const COMMUNITY_DISCLAIMER =
  'This LoRA was created by a community member, not LTX. LTX does not endorse or take responsibility for community LoRAs. LTX has no control over the LoRA and any claims of any kind should be directed to the creator of the LoRA.'

// Info icon → popover with a LoRA/IC-LoRA's instruction sections + a link to its HF page
// (derived from repoId, no stored URL). Native popover API: top-layer (escapes the
// modal's overflow clipping and stacks above the <dialog>), with ESC + click-outside
// dismiss for free. The popover/popovertarget attrs aren't in @types/react 18.3's JSX
// yet, so they're declared via cast objects below — but statically, so the element is
// never rendered without `popover` (setting it post-mount let it flash inline in the row).
export function LoraInfoPopover({
  sections,
  name,
  repoId,
  isCommunity = true,
  maxHeight = '70vh',
}: {
  sections: InstructionSection[]
  name: string
  repoId?: string
  // Fail closed: missing affiliation → show disclaimer.
  isCommunity?: boolean
  // Caller-tunable cap so the popover scrolls instead of overflowing its container
  // (e.g. the library modal needs a lower value than it does elsewhere).
  maxHeight?: string
}) {
  const id = useId()
  const btnRef = useRef<HTMLButtonElement>(null)
  const popRef = useRef<HTMLDivElement>(null)
  const showCommunityDisclaimer = isCommunity

  if (sections.length === 0 && !repoId && !showCommunityDisclaimer) return null

  // Native Popover API attrs missing from @types/react 18.3's JSX; cast so they render.
  const popoverAttrs = { popover: 'auto' } as unknown as React.HTMLAttributes<HTMLDivElement>
  const triggerAttrs = { popovertarget: id } as unknown as React.ButtonHTMLAttributes<HTMLButtonElement>


  // The popover is viewport-fixed in the top layer; anchor it above the trigger on open.
  const anchor = () => {
    const btn = btnRef.current, pop = popRef.current
    if (!btn || !pop) return
    const r = btn.getBoundingClientRect()
    pop.style.margin = '0'
    pop.style.right = 'auto'
    pop.style.top = 'auto'
    pop.style.left = `${Math.max(8, r.left)}px`
    pop.style.bottom = `${window.innerHeight - r.top + 6}px`
  }

  return (
    <>
      <button
        ref={btnRef}
        type="button"
        {...triggerAttrs}
        onClick={anchor}
        aria-label={`${name} instructions`}
        className="rounded p-0.5 text-zinc-400 transition-colors hover:text-white"
      >
        <Info className="h-3.5 w-3.5" />
      </button>
      <div
        ref={popRef}
        id={id}
        {...popoverAttrs}
        aria-label={`${name} instructions`}
        style={{ maxHeight }}
        className="w-72 overflow-y-auto rounded-lg border border-zinc-700 bg-zinc-900 p-3 text-left text-white shadow-xl"
      >
        <div className="mb-1.5 text-xs font-semibold text-white">{name}</div>
        <div className="flex flex-col gap-2">
          {sections.map((s, i) => (
            <div key={i}>
              <div className="text-[10px] font-semibold uppercase tracking-wider text-zinc-500">{s.title}</div>
              {Array.isArray(s.body) ? (
                <ul className="mt-0.5 list-disc pl-4 text-[11px] text-zinc-300">
                  {s.body.map((b, j) => <li key={j}>{b}</li>)}
                </ul>
              ) : (
                <p className="mt-0.5 text-[11px] text-zinc-300">{s.body}</p>
              )}
            </div>
          ))}
        </div>
        {showCommunityDisclaimer && (
          <p className="mt-2.5 border-t border-zinc-800 pt-2 text-[10px] leading-relaxed text-zinc-500">
            {COMMUNITY_DISCLAIMER}
          </p>
        )}
        {repoId && (
          <button
            type="button"
            onClick={() => { void window.electronAPI.openHuggingFaceRepo({ repoId }) }}
            className="mt-2.5 inline-flex items-center gap-1 text-[11px] font-medium text-indigo-400 transition-colors hover:text-indigo-300"
          >
            <ExternalLink className="h-3 w-3" /> View on HuggingFace
          </button>
        )}
      </div>
    </>
  )
}
