import React, { useState, useRef, useCallback, useEffect } from 'react'
import ReactDOM from 'react-dom'
import { cn } from '@/lib/utils'

interface TooltipProps {
  content: React.ReactNode
  children: React.ReactNode
  /** Which side of the trigger the tooltip appears on. Default: 'top' */
  side?: 'top' | 'bottom' | 'left' | 'right'
  className?: string
}

const DELAY_MS = 500
const GAP_PX = 6

/**
 * Styled tooltip with 250ms show delay and instant hide.
 * Renders via a portal into document.body so it is never clipped
 * by overflow-hidden ancestors.
 */
export function Tooltip({ content, children, side = 'top', className }: TooltipProps) {
  const [visible, setVisible] = useState(false)
  const [style, setStyle] = useState<React.CSSProperties>({})
  const wrapperRef = useRef<HTMLDivElement>(null)
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const computeStyle = useCallback(() => {
    const rect = wrapperRef.current?.getBoundingClientRect()
    if (!rect) return {}
    switch (side) {
      case 'top':
        return { left: rect.left + rect.width / 2, top: rect.top - GAP_PX, transform: 'translate(-50%, -100%)' }
      case 'bottom':
        return { left: rect.left + rect.width / 2, top: rect.bottom + GAP_PX, transform: 'translate(-50%, 0)' }
      case 'left':
        return { left: rect.left - GAP_PX, top: rect.top + rect.height / 2, transform: 'translate(-100%, -50%)' }
      case 'right':
        return { left: rect.right + GAP_PX, top: rect.top + rect.height / 2, transform: 'translate(0, -50%)' }
    }
  }, [side])

  const handleMouseEnter = useCallback(() => {
    timerRef.current = setTimeout(() => {
      setStyle(computeStyle() ?? {})
      setVisible(true)
    }, DELAY_MS)
  }, [computeStyle])

  const handleMouseLeave = useCallback(() => {
    if (timerRef.current) { clearTimeout(timerRef.current); timerRef.current = null }
    setVisible(false)
  }, [])

  useEffect(() => () => { if (timerRef.current) clearTimeout(timerRef.current) }, [])

  return (
    <div
      ref={wrapperRef}
      className={cn('inline-flex', className)}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
    >
      {children}
      {visible && ReactDOM.createPortal(
        <div
          className="fixed z-[99999] px-2.5 py-1.5 bg-white text-zinc-800 text-xs font-medium rounded-md shadow-md whitespace-nowrap pointer-events-none select-none"
          style={style}
        >
          {content}
        </div>,
        document.body,
      )}
    </div>
  )
}
