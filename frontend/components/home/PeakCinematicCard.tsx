import { Fragment, useMemo } from 'react'
import { Text } from '@/ds/Text/Text'
import styles from './PeakCinematicCard.module.scss'
import { PeakCinematicPlaylist } from './PeakCinematicPlaylist'
import { type PeakCinematicExample } from './peakCinematic'
import { useHomeMediaVisibility } from './useHomeMediaVisibility'

/**
 * Full-bleed cinematic peak — muted playlist fills the card; title + blurb
 * overlay bottom-left. Clips play one after another and loop.
 */
export function PeakCinematicCard({
  title,
  description,
  examples,
}: {
  title: string
  description: string
  examples: PeakCinematicExample[]
}) {
  const playableExamples = useMemo(
    () => examples.filter((example) => Boolean(example.videoUrl)),
    [examples],
  )
  const hasMedia = playableExamples.length > 0
  const { rootRef, hasBeenVisible, inView } = useHomeMediaVisibility()

  const overlay = (
    <div className={styles.content}>
      <div className={styles.textPair}>
        <Text as="p" variant="display" size="xl" className={styles.eyebrow}>
          {title}
        </Text>
        <Text as="h1" variant="display" size="xl" className={styles.title}>
          {description.split('\n').map((line, index, lines) => (
            <Fragment key={`${line}-${index}`}>
              {line}
              {index < lines.length - 1 ? (
                <>
                  <br className={styles.titleBreak} />
                  <span className={styles.titleJoin}> </span>
                </>
              ) : null}
            </Fragment>
          ))}
        </Text>
      </div>
    </div>
  )

  return (
    <article ref={rootRef} className={styles.card} aria-label={title}>
      {hasMedia ? (
        <PeakCinematicPlaylist
          examples={playableExamples}
          hasBeenVisible={hasBeenVisible}
          inView={inView}
        />
      ) : null}
      {overlay}
    </article>
  )
}
