export type MenuPlacement = 'above' | 'below'

const EDGE_PX = 8

export function fixedMenuPosition({
  trigger,
  placement,
  viewport,
  menuWidth,
  gap = 8,
}: {
  trigger: { left: number; right: number; top: number; bottom: number }
  placement: MenuPlacement
  viewport: { width: number; height: number }
  menuWidth?: number
  gap?: number
}): { left: number; top?: number; bottom?: number } {
  let left = trigger.left
  const measuredWidth =
    menuWidth != null && menuWidth < viewport.width - EDGE_PX * 2 ? menuWidth : undefined
  if (measuredWidth != null) {
    const maxLeft = viewport.width - measuredWidth - EDGE_PX
    if (left > maxLeft) left = trigger.right - measuredWidth
    left = Math.max(EDGE_PX, Math.min(left, maxLeft))
  } else {
    left = Math.max(EDGE_PX, left)
  }

  if (placement === 'below') {
    return { left, top: trigger.bottom + gap }
  }
  return { left, bottom: viewport.height - trigger.top + gap }
}
