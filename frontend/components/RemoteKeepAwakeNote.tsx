export function RemoteKeepAwakeNote({ compact }: { compact?: boolean }) {
  return (
    <p className={compact ? 'text-xs text-fg-tertiary leading-relaxed' : 'text-sm text-fg-secondary leading-relaxed'}>
      While Remote is starting or serving and this computer is plugged in, idle sleep is blocked so
      other devices can still reach it. On battery, closing the lid, or choosing Sleep still sleeps.
      The display can still turn off.
    </p>
  )
}
