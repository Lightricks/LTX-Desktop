import { useCallback, useEffect, useRef, useState } from 'react'

/** Within this distance (px) from the bottom, the view is "at the bottom" and sticks to it. */
const SNAP_THRESHOLD_PX = 16
/** Beyond this distance (px) from the bottom, the jump-to-bottom button shows. */
const JUMP_BUTTON_THRESHOLD_PX = 160

/** What a scroll position (distance in px from the bottom) means for the view. */
export function getScrollState(distanceFromBottom: number) {
  return {
    sticks: distanceFromBottom <= SNAP_THRESHOLD_PX,
    showJumpButton: distanceFromBottom > JUMP_BUTTON_THRESHOLD_PX,
  }
}

/**
 * Keeps a scroll container glued to its bottom while new content arrives.
 *
 * The scroll position alone decides whether the view sticks: scrolling up
 * past the snap threshold releases it, and reaching the bottom again (by hand
 * or with `scrollToBottom`) re-engages it. `enabled: false` pauses the
 * snapping, for example while the user browses search matches.
 */
export function useStickToBottom(enabled: boolean) {
  const containerRef = useRef<HTMLDivElement>(null)
  const contentRef = useRef<HTMLDivElement>(null)
  const stickRef = useRef(true)
  const enabledRef = useRef(enabled)
  const [showJumpButton, setShowJumpButton] = useState(false)
  // Mirrors `stickRef` for callers that must react when the user scrolls away or back.
  const [isStuck, setIsStuck] = useState(true)

  useEffect(() => {
    enabledRef.current = enabled
  }, [enabled])

  const distanceFromBottom = useCallback(() => {
    const el = containerRef.current
    return el ? el.scrollHeight - el.scrollTop - el.clientHeight : 0
  }, [])

  // The only place that decides whether the view sticks.
  const onScroll = useCallback(() => {
    const state = getScrollState(distanceFromBottom())
    stickRef.current = state.sticks
    setIsStuck(state.sticks)
    setShowJumpButton(state.showJumpButton)
  }, [distanceFromBottom])

  const scrollToBottom = useCallback(() => {
    const el = containerRef.current
    if (!el) return
    el.scrollTop = el.scrollHeight
    onScroll()
  }, [onScroll])

  // New content or a layout change (font size, wrap, resize) moves the bottom
  // without a scroll event. Follow it while stuck, otherwise refresh the button.
  useEffect(() => {
    const container = containerRef.current
    const content = contentRef.current
    if (!container || !content) return
    const observer = new ResizeObserver(() => {
      if (stickRef.current && enabledRef.current) {
        scrollToBottom()
      } else {
        setShowJumpButton(getScrollState(distanceFromBottom()).showJumpButton)
      }
    })
    observer.observe(container)
    observer.observe(content)
    return () => observer.disconnect()
  }, [scrollToBottom, distanceFromBottom])

  // Back to following the log (for example when a search ends).
  useEffect(() => {
    if (enabled) scrollToBottom()
  }, [enabled, scrollToBottom])

  return { containerRef, contentRef, onScroll, scrollToBottom, showJumpButton, isStuck }
}
