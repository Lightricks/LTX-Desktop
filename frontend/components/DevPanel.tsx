import { useState, useEffect } from 'react'
import { X, Copy, Check } from 'lucide-react'
import { useDevFlags, DEV_FLAGS, type DevFlagKey } from '../contexts/DevFlagsContext'

// Master-detail feature-flag panel (Ctrl/Cmd+Shift+D). List on the left
// (green = enabled, ring = selected), detail + toggle on the right.
export function DevPanel() {
  const { flags, setFlag, isPanelOpen, setPanelOpen } = useDevFlags()
  const [selectedKey, setSelectedKey] = useState<DevFlagKey>(DEV_FLAGS[0]?.key)
  const [copied, setCopied] = useState(false)

  useEffect(() => {
    if (!isPanelOpen) return
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setPanelOpen(false) }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [isPanelOpen, setPanelOpen])

  if (!isPanelOpen) return null

  const selected = DEV_FLAGS.find(f => f.key === selectedKey) ?? DEV_FLAGS[0]
  const isOn = selected ? flags[selected.key] : false

  const copyKey = () => {
    if (!selected) return
    void navigator.clipboard?.writeText(selected.key).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 1200)
    })
  }

  return (
    <div
      className="fixed inset-0 z-[70] flex items-center justify-center bg-black/60 backdrop-blur-sm"
      onClick={() => setPanelOpen(false)}
    >
      <div
        className="flex h-[440px] w-full max-w-3xl flex-col overflow-hidden rounded-xl border border-zinc-700 bg-zinc-900 shadow-2xl"
        onClick={e => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between border-b border-zinc-800 px-5 py-3.5">
          <div>
            <h2 className="text-base font-semibold text-white">Dev Panel</h2>
            <p className="text-[11px] text-zinc-500">Local-only feature toggles, saved on this machine.</p>
          </div>
          <button
            onClick={() => setPanelOpen(false)}
            className="rounded-md p-1.5 text-zinc-400 transition-colors hover:bg-zinc-800 hover:text-white"
            title="Close (Esc)"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        {/* Body: list + detail */}
        <div className="flex min-h-0 flex-1">
          {/* List */}
          <div className="w-64 shrink-0 overflow-y-auto border-r border-zinc-800 p-2">
            {DEV_FLAGS.map(spec => {
              const on = flags[spec.key]
              const isSelected = spec.key === selectedKey
              return (
                <button
                  key={spec.key}
                  onClick={() => setSelectedKey(spec.key)}
                  className={`mb-1 w-full rounded-md px-3 py-2 text-left text-sm transition-colors ${
                    isSelected
                      ? 'text-white ring-1 ring-blue-500/60 ' + (on ? 'bg-emerald-900/40' : 'bg-zinc-800')
                      : on
                        ? 'bg-emerald-900/30 text-emerald-100 hover:bg-emerald-900/50'
                        : 'text-zinc-300 hover:bg-zinc-800'
                  }`}
                >
                  {spec.label}
                </button>
              )
            })}
          </div>

          {/* Detail */}
          <div className="min-w-0 flex-1 overflow-y-auto p-5">
            {selected && (
              <>
                <div className="flex items-start justify-between gap-3">
                  <h3 className="text-lg font-semibold text-white">{selected.label}</h3>
                  <button
                    onClick={copyKey}
                    className="flex shrink-0 items-center gap-1.5 font-mono text-[11px] text-zinc-500 transition-colors hover:text-zinc-300"
                    title="Copy flag key"
                  >
                    {selected.key}
                    {copied ? <Check className="h-3 w-3 text-green-400" /> : <Copy className="h-3 w-3" />}
                  </button>
                </div>

                <div className="mt-4 flex items-center gap-3">
                  <span className="text-xs font-medium uppercase tracking-wide text-zinc-400">Enabled</span>
                  <button
                    type="button"
                    role="switch"
                    aria-checked={isOn}
                    onClick={() => setFlag(selected.key, !isOn)}
                    className={`relative inline-flex h-6 w-11 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none ${
                      isOn ? 'bg-blue-500' : 'bg-zinc-600'
                    }`}
                  >
                    <span
                      className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out ${
                        isOn ? 'translate-x-5' : 'translate-x-0'
                      }`}
                    />
                  </button>
                </div>

                <div className="mt-6">
                  <p className="mb-1 text-xs font-medium text-zinc-400">Description</p>
                  <p className="text-sm leading-relaxed text-zinc-300">{selected.description}</p>
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
