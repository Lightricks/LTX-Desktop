import { useEffect, useRef, useState } from 'react'
import { flushSync } from 'react-dom'
import { Button } from '@ds/Button/Button'
import { Text } from '@ds/Text/Text'
import { InstallProgressBody } from '../install/InstallProgress'
import type { DownloadProgressSnapshot } from '@/lib/download-progress'
import { usePrefersReducedMotion } from '../home/usePrefersReducedMotion'
import { useThemeColorScheme } from '@/ds/styles/themes/useTheme'
import { EndpageLockup, ENDPAGE_SETTLED_MS } from './EndpageLockup'
import endpageAudioUrl from './ltx-endpage.m4a'
import lockupStyles from './EndpageLockup.module.scss'
import styles from './StartupLoader.module.scss'

// The bar waits for the lockup to land, so it never fades in over the motion.
const PROGRESS_AFTER_MS = ENDPAGE_SETTLED_MS
const AUDIO_FADE_MS = 240
// Longest the picture waits for the soundtrack to become playable. Past this
// the splash starts regardless, silent or slightly late, rather than stalling.
const AUDIO_WAIT_CAP_MS = 1000
const WAITING_FILL_PERCENT = 90
const READY_FILL_PERCENT = 100

type StartupLoaderProps = {
  ready?: boolean
  onFinished?: () => void
  /**
   * Play the endpage mix. `null` while the caller is still deciding (the
   * splash holds frame 0 for that beat, capped like the audio wait below, so
   * a first launch never starts muted just because the check was slow).
   */
  sound?: boolean | null
  /**
   * Present while Python downloads: the splash shows the letters, then the
   * shared progress body with a real bar. Absent = plain boot splash with a
   * waiting fill. The two modes share the lockup and the exit fade, nothing
   * else — the boot animation machine below never runs in download mode.
   */
  python?: {
    snapshot: DownloadProgressSnapshot
    error: string | null
    onRetry: () => void
  } | null
}

export function StartupLoader({
  ready = false,
  onFinished,
  sound = true,
  python = null,
}: StartupLoaderProps) {
  const mode = python != null ? 'download' : 'boot'
  const [settled, setSettled] = useState(false)
  const [showProgress, setShowProgress] = useState(false)
  const [fillFinishing, setFillFinishing] = useState(false)
  const [exiting, setExiting] = useState(false)
  const [started, setStarted] = useState(false)
  const reducedMotion = usePrefersReducedMotion()
  const colorScheme = useThemeColorScheme()
  const isReady = ready

  const rootRef = useRef<HTMLDivElement>(null)
  const fillNodeRef = useRef<HTMLDivElement>(null)
  const fillAmountRef = useRef(0)
  const readyRef = useRef(ready)
  const onFinishedRef = useRef(onFinished)
  const exitStartedRef = useRef(false)

  readyRef.current = isReady
  onFinishedRef.current = onFinished
  const startedRef = useRef(false)
  startedRef.current = started

  useEffect(() => {
    if (!started) return
    const settleId = window.setTimeout(() => setSettled(true), ENDPAGE_SETTLED_MS)
    return () => window.clearTimeout(settleId)
  }, [started])

  // The mix is cut to start with frame 0, so the two only line up if they begin
  // in the same tick. Fetching and decoding the audio takes time the animation
  // does not, so the lockup holds frame 0 until the track can play through;
  // starting both on mount let the picture run ahead of the mix on a cold load.
  // Autoplay is blocked until the window has seen a gesture; a silent splash is
  // the documented fallback, so the picture starts either way.
  useEffect(() => {
    if (reducedMotion) {
      setStarted(true)
      return
    }
    if (sound === null) {
      const undecidedId = window.setTimeout(() => setStarted(true), AUDIO_WAIT_CAP_MS)
      return () => window.clearTimeout(undecidedId)
    }
    // Already playing means the cap above fired first: stay silent rather than
    // dropping the mix in partway through the animation.
    if (!sound || startedRef.current) {
      setStarted(true)
      return
    }
    const audio = new Audio()
    audio.preload = 'auto'
    audio.src = endpageAudioUrl

    let launched = false
    let capId = 0
    // `playing`, not the play() call: decoding finishes before the output device
    // is actually running, and that warm-up only happens on the session's first
    // sound — which is why later replays looked in sync and the first never did.
    const launch = () => {
      if (launched) return
      launched = true
      window.clearTimeout(capId)
      audio.removeEventListener('canplaythrough', requestPlay)
      audio.removeEventListener('playing', launch)
      setStarted(true)
    }
    const requestPlay = () => {
      void audio.play().catch(launch)
    }
    capId = window.setTimeout(launch, AUDIO_WAIT_CAP_MS)
    audio.addEventListener('canplaythrough', requestPlay, { once: true })
    audio.addEventListener('playing', launch, { once: true })
    audio.load()

    return () => {
      window.clearTimeout(capId)
      audio.removeEventListener('canplaythrough', requestPlay)
      audio.removeEventListener('playing', launch)
      const startVolume = audio.volume
      const startedAt = performance.now()
      const fade = window.setInterval(() => {
        const progress = (performance.now() - startedAt) / AUDIO_FADE_MS
        if (progress >= 1) {
          window.clearInterval(fade)
          audio.pause()
          return
        }
        audio.volume = startVolume * (1 - progress)
      }, 20)
    }
  }, [reducedMotion, sound])

  useEffect(() => {
    if (!started || isReady) return
    const progressId = window.setTimeout(() => {
      if (!readyRef.current && !exitStartedRef.current) {
        setShowProgress(true)
      }
    }, PROGRESS_AFTER_MS)
    return () => window.clearTimeout(progressId)
  }, [started, isReady])

  useEffect(() => {
    if (!showProgress || isReady) return

    if (reducedMotion) {
      setFillWidth(fillNodeRef.current, fillAmountRef, WAITING_FILL_PERCENT)
      return
    }

    const tauMs = readCssTimeMs(rootRef.current, '--splash-wait-duration')
    const startedAt = performance.now()
    const from = fillAmountRef.current
    let raf = 0
    const tick = (now: number) => {
      const tau = tauMs > 0 ? tauMs : 1
      const next =
        WAITING_FILL_PERCENT - (WAITING_FILL_PERCENT - from) * Math.exp(-(now - startedAt) / tau)
      setFillWidth(fillNodeRef.current, fillAmountRef, next)
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [showProgress, isReady, reducedMotion])

  useEffect(() => {
    if (!isReady || exitStartedRef.current) return
    exitStartedRef.current = true

    let cancelled = false
    let completed = false

    const finish = () => {
      if (cancelled || completed) return
      completed = true
      onFinishedRef.current?.()
    }

    if (reducedMotion) {
      finish()
      return () => {
        cancelled = true
        if (!completed) exitStartedRef.current = false
      }
    }

    const run = async () => {
      const fillEl = fillNodeRef.current
      const rootEl = rootRef.current

      if (showProgress && fillEl) {
        const currentPct = readFillPercent(fillEl, fillAmountRef.current)
        setFillWidth(fillEl, fillAmountRef, currentPct)
        flushSync(() => setFillFinishing(true))
        await nextPaint()
        if (cancelled) return
        setFillWidth(fillEl, fillAmountRef, READY_FILL_PERCENT)
        const finishMs = readCssTimeMs(rootEl, '--splash-finish-duration')
        await waitForTransition(fillEl, 'width', finishMs)
        if (cancelled) return
      }

      if (!rootEl) {
        finish()
        return
      }

      flushSync(() => setExiting(true))
      const fadeMs = readCssTimeMs(rootEl, '--splash-fade-duration')
      await waitForTransition(rootEl, 'opacity', fadeMs)
      if (cancelled) return
      finish()
    }

    void run()
    return () => {
      cancelled = true
      if (!completed) exitStartedRef.current = false
    }
  }, [mode, isReady, showProgress, reducedMotion])

  return (
    <div
      ref={rootRef}
      className={`${styles.root}${exiting ? ` ${styles.rootExiting}` : ''}`}
    >
      {/* The `ltx-io` scope is load-bearing (semantic tokens only resolve under
          `.ltx-io[data-theme]`), but it must NOT sit on the root: its base
          `height: 100%` beats `.root`'s `100vh` in load order and collapses the
          splash, whose children are all absolute. The scheme follows the app
          theme so the lockup stays legible against glass that does the same. */}
      <div className={`ltx-io ${styles.themeScope}`} data-theme={colorScheme}>
        <div className={styles.dragRegion} aria-hidden />
        <div className={lockupStyles.shell}>
          <div
            className={styles.lockup}
            role="status"
            aria-live="polite"
            aria-busy={!isReady && python?.error == null}
            aria-label={python?.snapshot.statusLabel ?? 'LTX Desktop'}
          >
            <EndpageLockup paused={!started} />
          </div>
          {/* Download mode: letters first, then the shared progress body in the
              slot position. The boot track/fill nodes above stay boot-only. */}
          {python != null && settled ? (
              <div className={`${styles.progressSlot} ${styles.progressSlotVisible} ${styles.progressSlotDownload}`}>
                <InstallProgressBody
                  snapshot={python.snapshot}
                  error={
                    python.error ? (
                      <>
                        <Text as="p" variant="body" size="md" align="center">
                          {python.error}
                        </Text>
                        <div className={styles.downloadActions}>
                          <Button
                            appearance="neutral"
                            hierarchy="primary"
                            size="md"
                            label="Retry"
                            onClick={python.onRetry}
                          />
                        </div>
                      </>
                    ) : undefined
                  }
                />
              </div>
          ) : (
            <div
              className={`${styles.progressSlot} ${showProgress ? styles.progressSlotVisible : ''}`}
              role="progressbar"
              aria-hidden={!showProgress}
              aria-label="Starting"
            >
              <div className={styles.progressTrack}>
                <div
                  ref={fillNodeRef}
                  className={`${styles.progressFill}${fillFinishing ? ` ${styles.progressFillFinish}` : ''}`}
                />
              </div>
            </div>
          )}
        </div>
        <Text as="p" variant="body" size="md" className={styles.copyright}>
          © 2026 Lightricks
        </Text>
      </div>
    </div>
  )
}

function setFillWidth(
  el: HTMLDivElement | null,
  fillAmountRef: { current: number },
  percent: number,
): void {
  fillAmountRef.current = percent
  if (el) el.style.width = `${percent}%`
}

function readFillPercent(el: HTMLDivElement, fallback: number): number {
  const track = el.parentElement
  if (!track) return fallback
  const trackWidth = track.getBoundingClientRect().width
  if (trackWidth <= 0) return fallback
  return (el.getBoundingClientRect().width / trackWidth) * 100
}

function readCssTimeMs(el: HTMLElement | null, variable: string): number {
  if (!el) return 0
  return parseCssTimeMs(getComputedStyle(el).getPropertyValue(variable))
}

function parseCssTimeMs(value: string): number {
  const token = value.trim().split(',')[0]?.trim() ?? ''
  if (token.endsWith('ms')) {
    const n = Number.parseFloat(token)
    return Number.isFinite(n) ? n : 0
  }
  if (token.endsWith('s')) {
    const n = Number.parseFloat(token)
    return Number.isFinite(n) ? n * 1000 : 0
  }
  const n = Number.parseFloat(token)
  return Number.isFinite(n) ? n : 0
}

function nextPaint(): Promise<void> {
  return new Promise((resolve) => {
    requestAnimationFrame(() => {
      requestAnimationFrame(() => resolve())
    })
  })
}

function waitForTransition(el: HTMLElement, property: string, durationMs: number): Promise<void> {
  if (durationMs <= 0) return Promise.resolve()
  return new Promise((resolve) => {
    let settled = false
    const settle = () => {
      if (settled) return
      settled = true
      el.removeEventListener('transitionend', onEnd)
      window.clearTimeout(timer)
      resolve()
    }
    const onEnd = (event: TransitionEvent) => {
      if (event.target !== el || event.propertyName !== property) return
      settle()
    }
    el.addEventListener('transitionend', onEnd)
    const timer = window.setTimeout(settle, durationMs)
  })
}
