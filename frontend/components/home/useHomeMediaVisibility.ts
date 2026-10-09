import { useCallback, useEffect, useState } from 'react'
import { homeMediaObserverRoot } from './homeFeaturePreviewMedia'

const HOME_MEDIA_ROOT_MARGIN = '120px 0px'
const HOME_MEDIA_THRESHOLD = 0.01

/**
 * Viewport gate for Home media. Uses native IntersectionObserver so Desktop
 * does not take a react-intersection-observer dependency for this surface.
 */
export function useHomeMediaVisibility(): {
  rootRef: (node: HTMLElement | null) => void
  hasBeenVisible: boolean
  inView: boolean
} {
  const [element, setElement] = useState<HTMLElement | null>(null)
  const [inView, setInView] = useState(false)
  const [hasBeenVisible, setHasBeenVisible] = useState(false)
  const rootRef = useCallback((node: HTMLElement | null) => {
    setElement(node)
  }, [])

  useEffect(() => {
    if (typeof IntersectionObserver === 'undefined') {
      setInView(true)
      setHasBeenVisible(true)
      return
    }
    if (!element) return

    const observer = new IntersectionObserver(
      (entries) => {
        const entry = entries[0]
        if (!entry) return
        const isIntersecting = entry.isIntersecting
        setInView(isIntersecting)
        if (isIntersecting) setHasBeenVisible(true)
      },
      {
        root: homeMediaObserverRoot(element),
        rootMargin: HOME_MEDIA_ROOT_MARGIN,
        threshold: HOME_MEDIA_THRESHOLD,
      },
    )
    observer.observe(element)
    return () => observer.disconnect()
  }, [element])

  return { rootRef, hasBeenVisible, inView }
}
