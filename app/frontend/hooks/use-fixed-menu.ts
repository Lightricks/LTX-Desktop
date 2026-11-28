import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react'
import type { CSSProperties } from 'react'
import { fixedMenuPosition, type MenuPlacement } from '../lib/fixed-menu-position'

const HIDDEN_FIXED: CSSProperties = { position: 'fixed', zIndex: 9999, visibility: 'hidden' }

export function useFixedMenu(placement: MenuPlacement) {
  const [isOpen, setIsOpen] = useState(false)
  const triggerRef = useRef<HTMLDivElement>(null)
  const menuRef = useRef<HTMLDivElement>(null)
  const [style, setStyle] = useState<CSSProperties>(HIDDEN_FIXED)

  const updatePosition = useCallback(() => {
    const trigger = triggerRef.current?.getBoundingClientRect()
    if (!trigger) return
    const menuWidth = menuRef.current?.offsetWidth
    const pos = fixedMenuPosition({
      trigger,
      placement,
      viewport: { width: window.innerWidth, height: window.innerHeight },
      menuWidth: menuWidth || undefined,
    })
    setStyle({
      position: 'fixed',
      zIndex: 9999,
      visibility: 'visible',
      left: pos.left,
      ...(pos.top != null ? { top: pos.top } : { bottom: pos.bottom }),
    })
  }, [placement])

  useLayoutEffect(() => {
    if (!isOpen) {
      setStyle(HIDDEN_FIXED)
      return
    }
    updatePosition()
    window.addEventListener('resize', updatePosition)
    window.addEventListener('scroll', updatePosition, true)
    return () => {
      window.removeEventListener('resize', updatePosition)
      window.removeEventListener('scroll', updatePosition, true)
    }
  }, [isOpen, updatePosition])

  useEffect(() => {
    if (!isOpen) return
    const onPointerDown = (event: MouseEvent) => {
      const node = event.target as Node
      if (triggerRef.current?.contains(node) || menuRef.current?.contains(node)) return
      setIsOpen(false)
    }
    document.addEventListener('mousedown', onPointerDown)
    return () => document.removeEventListener('mousedown', onPointerDown)
  }, [isOpen])

  return { isOpen, setIsOpen, triggerRef, menuRef, style }
}
