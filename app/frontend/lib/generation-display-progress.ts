/**
 * How the GenSpace circle/bar should fill from a progress poll.
 * Step-based inference fill was proposed by Sergio Gil (@jimeneztion).
 */

/** In-flight jobs never show 100% — decode/encode still runs after the last denoise step. */
export const MAX_IN_FLIGHT_PROGRESS_PERCENT = 95

export function displayPolledProgress(args: {
  phase: string
  progress: number
  totalSteps: number | null
  elapsedInferenceS: number
  estimatedInferenceS: number
}): number {
  if (args.phase === 'complete') {
    return MAX_IN_FLIGHT_PROGRESS_PERCENT
  }
  // Local jobs report real denoise totals. Trust the backend percentage so the
  // existing bar tracks work done instead of a guessed duration.
  if (args.phase === 'inference' && (args.totalSteps ?? 0) > 1) {
    return Math.min(MAX_IN_FLIGHT_PROGRESS_PERCENT, args.progress)
  }
  if (args.phase === 'inference') {
    const inferenceProgress = Math.min(
      args.elapsedInferenceS / args.estimatedInferenceS,
      MAX_IN_FLIGHT_PROGRESS_PERCENT / 100,
    )
    return Math.min(
      MAX_IN_FLIGHT_PROGRESS_PERCENT,
      15 + Math.floor(inferenceProgress * 80),
    )
  }
  return Math.min(MAX_IN_FLIGHT_PROGRESS_PERCENT, args.progress)
}
