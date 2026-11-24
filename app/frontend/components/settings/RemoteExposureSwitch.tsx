import type { ReactNode } from 'react'

export function RemoteExposureSwitch({
  enabled,
  onToggle,
  className,
  children,
}: {
  enabled: boolean
  onToggle: () => void
  className: string
  children: ReactNode
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={enabled}
      aria-label={enabled ? 'Remote on' : 'Remote off'}
      onClick={onToggle}
      className={className}
    >
      {children}
    </button>
  )
}
