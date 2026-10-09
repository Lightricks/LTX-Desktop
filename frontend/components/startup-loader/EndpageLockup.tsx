import { useMemo, type CSSProperties } from 'react'
import Lottie, { type Options as LottieOptions } from 'react-lottie'

import animationData from './ltx-endpage.json'
import styles from './EndpageLockup.module.scss'

/** Frame the camera pull-back lands on; everything after it is a static hold. */
export const ENDPAGE_SETTLED_FRAME = 90
export const ENDPAGE_SETTLED_MS = Math.round((ENDPAGE_SETTLED_FRAME / 30) * 1000)

/**
 * react-lottie forwards unknown options straight to lottie-web but its types
 * only cover the ones it documents, and `initialSegment` is how a frame gets
 * rendered without playing.
 */
function lottieOptions(options: LottieOptions & { initialSegment: [number, number] }): LottieOptions {
  return options
}

type EndpageLockupProps = {
  /** Hold the settled frame instead of playing in — the install screen's lockup. */
  still?: boolean
  /** Hold frame 0 until the caller is ready, so picture and soundtrack start together. */
  paused?: boolean
  className?: string
  style?: CSSProperties
}

/**
 * The LTX-2 endpage animation, cropped to the window like the master render
 * (16:9 artwork, covered rather than letterboxed so the intro close-up runs off
 * the window edges instead of cutting inside it).
 */
export function EndpageLockup({ still = false, paused = false, className, style }: EndpageLockupProps) {
  // react-lottie replays on every parent render, so the element identity has to
  // stay stable while progress ticks around it.
  const player = useMemo(
    () => (
      <Lottie
        options={lottieOptions({
          loop: false,
          autoplay: !still && !paused,
          animationData,
          rendererSettings: { preserveAspectRatio: 'xMidYMid slice' },
          // A one-frame segment renders the settled frame without playing.
          initialSegment: still
            ? [ENDPAGE_SETTLED_FRAME, ENDPAGE_SETTLED_FRAME + 1]
            : [0, ENDPAGE_SETTLED_FRAME + 1],
        })}
        isStopped={!still && paused}
        height="100%"
        width="100%"
        isClickToPauseDisabled
        ariaRole="presentation"
        ariaLabel="LTX Desktop"
      />
    ),
    [still, paused],
  )

  return (
    <div className={`${styles.stage}${className ? ` ${className}` : ''}`} style={style} aria-hidden>
      <div className={styles.frame}>{player}</div>
    </div>
  )
}
