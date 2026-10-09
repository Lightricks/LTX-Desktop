/**
 * LTX-2.5 cinematic showreel on the home page.
 * Copy from the LTX-2.5 Messaging Doc.
 */
export const PEAK_CINEMATIC_TITLE = 'LTX-2.5'
// Newline is intentional: mobile shows two lines; desktop collapses it to one.
export const PEAK_CINEMATIC_DESCRIPTION = 'The world model\nthe world builds on.'

/** One clip in a sequential cinematic playlist. */
export type PeakCinematicExample = {
  id: string
  videoUrl?: string
  posterUrl?: string
}
