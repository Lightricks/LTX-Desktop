import { useState } from 'react'
import { Sparkles, X, Search } from 'lucide-react'
import { EFFECT_DEFINITIONS } from '../../types/project'
import type { EffectType, TimelineClip } from '../../types/project-model'

interface EffectsBrowserProps {
  onClose: () => void
  selectedClip: TimelineClip | null
  addEffectToClip: (clipId: string, effectType: EffectType) => void
}

export function EffectsBrowser({ onClose, selectedClip, addEffectToClip }: EffectsBrowserProps) {
  const [effectsSearchQuery, setEffectsSearchQuery] = useState('')

  return (
    <div className="w-56 flex-shrink-0 bg-surface-secondary border-r border-separator-secondary flex flex-col overflow-hidden">
      {/* Header */}
      <div className="flex items-center gap-2 px-3 py-2.5 border-b border-separator-secondary bg-[color-mix(in_srgb,var(--semantic-bg-primary)_50%,transparent)]">
        <div className="w-5 h-5 rounded bg-blue-600/20 flex items-center justify-center">
          <Sparkles className="h-3 w-3 text-blue-400" />
        </div>
        <span className="text-[11px] font-semibold text-fg-primary flex-1">Effects</span>
        <button onClick={onClose} className="text-fg-tertiary hover:text-fg-secondary transition-colors">
          <X className="h-3.5 w-3.5" />
        </button>
      </div>
      {/* Search */}
      <div className="px-2.5 py-2 border-b border-separator-secondary">
        <div className="relative">
          <Search className="absolute left-2 top-1/2 -translate-y-1/2 h-3 w-3 text-fg-tertiary" />
          <input
            type="text"
            placeholder="Search effects..."
            value={effectsSearchQuery}
            onChange={(e) => setEffectsSearchQuery(e.target.value)}
            className="w-full pl-7 pr-2 py-1.5 bg-[color-mix(in_srgb,var(--semantic-bg-action-secondary-enabled)_70%,transparent)] rounded-md text-[11px] text-fg-primary placeholder-fg-tertiary outline-none border border-separator-secondary focus:border-blue-500/50 focus:bg-action transition-colors"
          />
        </div>
      </div>
      {/* Effect categories */}
      <div className="flex-1 overflow-y-auto py-1.5 custom-scrollbar">
        {(['filter', 'stylize', 'color-preset'] as const).map(category => {
          const categoryLabel = category === 'filter' ? 'Filters' : category === 'stylize' ? 'Stylize' : 'Color Presets'
          const categoryIcon = category === 'filter' ? 'filter' : category === 'stylize' ? 'stylize' : 'color'
          const effects = (Object.entries(EFFECT_DEFINITIONS) as [EffectType, typeof EFFECT_DEFINITIONS[EffectType]][])
            .filter(([_, def]) => def.category === category)
            .filter(([_, def]) => !effectsSearchQuery || def.name.toLowerCase().includes(effectsSearchQuery.toLowerCase()))
          if (effects.length === 0) return null
          return (
            <div key={category} className="mb-1">
              <div className="flex items-center gap-1.5 px-3 py-1.5 text-[9px] font-bold text-fg-tertiary uppercase tracking-[0.08em]">
                <div className={`w-1 h-1 rounded-full ${categoryIcon === 'filter' ? 'bg-blue-400' : categoryIcon === 'stylize' ? 'bg-warning' : 'bg-success'}`} />
                {categoryLabel}
              </div>
              <div className="px-2 space-y-px">
                {effects.map(([type, def]) => {
                  const lutGradient: Record<string, string> = {
                    'lut-cinematic': 'linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%)',
                    'lut-vintage': 'linear-gradient(135deg, #d4a373 0%, #e6ccb2 50%, #b5838d 100%)',
                    'lut-bw': 'linear-gradient(135deg, #111 0%, #666 50%, #ccc 100%)',
                    'lut-cool': 'linear-gradient(135deg, #4cc9f0 0%, #4895ef 50%, #4361ee 100%)',
                    'lut-warm': 'linear-gradient(135deg, #f77f00 0%, #fcbf49 50%, #eae2b7 100%)',
                    'lut-muted': 'linear-gradient(135deg, #6b7280 0%, #9ca3af 50%, #d1d5db 100%)',
                    'lut-vivid': 'linear-gradient(135deg, #ff006e 0%, #fb5607 50%, #ffbe0b 100%)',
                  }
                  const filterIcon: Record<string, string> = {
                    'blur': 'B', 'sharpen': 'S', 'glow': 'G', 'vignette': 'V', 'grain': 'N',
                  }
                  const filterColor: Record<string, string> = {
                    'blur': 'from-blue-500/20 to-blue-600/10 text-blue-400',
                    'sharpen': 'from-ext-teal/20 to-ext-teal/10 text-ext-teal',
                    'glow': 'from-warning/20 to-warning/10 text-fg-warning',
                    'vignette': 'from-blue-500/20 to-blue-600/10 text-blue-400',
                    'grain': 'from-stone-500/20 to-stone-600/10 text-stone-400',
                  }

                  return (
                    <button
                      key={type}
                      draggable
                      onDragStart={(e) => {
                        e.dataTransfer.setData('effectType', type)
                        e.dataTransfer.effectAllowed = 'copy'
                      }}
                      onDoubleClick={() => {
                        if (selectedClip) addEffectToClip(selectedClip.id, type)
                      }}
                      className="w-full flex items-center gap-2.5 px-2 py-1.5 rounded-md hover:bg-[color-mix(in_srgb,var(--semantic-bg-action-secondary-enabled)_80%,transparent)] transition-all cursor-grab active:cursor-grabbing group"
                      title={`${def.name} — drag onto clip or double-click to apply`}
                    >
                      {/* Icon/swatch */}
                      {category === 'color-preset' ? (
                        <div
                          className="w-7 h-7 rounded-md flex-shrink-0 ring-1 ring-white/10 group-hover:ring-white/20 transition-all"
                          style={{ background: lutGradient[type] || 'linear-gradient(135deg, #333, #555)' }}
                        />
                      ) : (
                        <div className={`w-7 h-7 rounded-md flex-shrink-0 bg-gradient-to-br ${filterColor[type] || 'from-action-hover to-action text-fg-secondary'} flex items-center justify-center ring-1 ring-white/5 group-hover:ring-white/15 transition-all`}>
                          <span className="text-[11px] font-black">{filterIcon[type] || 'F'}</span>
                        </div>
                      )}
                      {/* Label */}
                      <span className="text-[11px] text-fg-secondary group-hover:text-fg-primary transition-colors truncate">{def.name}</span>
                    </button>
                  )
                })}
              </div>
            </div>
          )
        })}
      </div>
      {/* Footer hint */}
      <div className="px-3 py-2 border-t border-separator-secondary bg-[color-mix(in_srgb,var(--semantic-bg-primary)_30%,transparent)]">
        <p className="text-[9px] text-fg-tertiary leading-relaxed">Drag onto a clip or double-click to apply to selection</p>
      </div>
    </div>
  )
}
