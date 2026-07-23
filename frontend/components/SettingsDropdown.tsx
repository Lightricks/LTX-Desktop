import React, { useState, useEffect, useRef } from 'react'
import { Tooltip } from './ui/tooltip'

// Generic settings dropdown used across the prompt bar's control rows.
export function SettingsDropdown({
  trigger,
  options,
  value,
  onChange,
  title,
  tooltip,
  triggerClassName,
}: {
  trigger: React.ReactNode
  options: { value: string; label: string; disabled?: boolean; tooltip?: string; icon?: React.ReactNode }[]
  value: string
  onChange: (value: string) => void
  title: string
  tooltip?: string
  // Extra classes on the trigger button — e.g. to visually attach it to an adjacent button
  // as a split-button (rounded-l-none, no left padding, etc).
  triggerClassName?: string
}) {
  const [isOpen, setIsOpen] = useState(false)
  const dropdownRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setIsOpen(false)
      }
    }
    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside)
    }
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [isOpen])

  const triggerButton = (
    <button
      onClick={() => setIsOpen(!isOpen)}
      className={`flex shrink-0 items-center gap-1 whitespace-nowrap px-2 py-1.5 rounded-md transition-colors ${isOpen ? 'bg-zinc-700 hover:bg-zinc-700' : 'hover:bg-zinc-800'} ${triggerClassName ?? ''}`}
    >
      {trigger}
    </button>
  )

  return (
    <div ref={dropdownRef} className="relative">
      {tooltip && !isOpen ? <Tooltip content={tooltip}>{triggerButton}</Tooltip> : triggerButton}

      {isOpen && (
        <div className="absolute bottom-full left-0 mb-2 bg-zinc-800 border border-zinc-700 rounded-md p-2 min-w-[160px] shadow-xl z-[9999]">
          <div className="text-[10px] text-zinc-500 uppercase tracking-wider mb-2">{title}</div>
          {/* Cap height + scroll so a long option list (e.g. many catalog / custom IC-LoRAs)
              doesn't clip off-screen — matches the LoRA picker's max-h-80. */}
          <div className="space-y-1 max-h-80 overflow-y-auto">
            {options.map(option => (
              <div key={option.value} className="relative group/option">
                <button
                  onClick={() => { if (!option.disabled) { onChange(option.value); setIsOpen(false) } }}
                  className={`w-full flex items-center justify-between px-2 py-2 rounded-md transition-colors text-left ${
                    option.disabled
                      ? 'cursor-not-allowed'
                      : value === option.value ? 'bg-white/20 hover:bg-white/25' : 'hover:bg-zinc-700'
                  }`}
                >
                  <span className={`flex items-center gap-2.5 text-sm ${
                    option.disabled
                      ? 'text-zinc-600'
                      : value === option.value ? 'text-white' : 'text-zinc-400'
                  }`}>
                    {option.icon && <span className="flex-shrink-0">{option.icon}</span>}
                    {option.label}
                  </span>
                  {value === option.value && !option.disabled && (
                    <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                    </svg>
                  )}
                </button>
                {option.disabled && option.tooltip && (
                  <div className="absolute left-full ml-2 top-1/2 -translate-y-1/2 px-2 py-1 bg-zinc-700 rounded text-xs text-zinc-300 whitespace-nowrap opacity-0 group-hover/option:opacity-100 pointer-events-none z-[10000] transition-opacity">
                    {option.tooltip}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
