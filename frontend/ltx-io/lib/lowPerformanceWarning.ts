/** Short edge from which a low performance machine runs a generation slowly. */
const LOW_PERFORMANCE_MIN_SHORT_EDGE = 720;

export const LOW_PERFORMANCE_RESOLUTION_WARNING =
  "This resolution is likely to take a long time on your Mac. Use 540p or lower for a quicker result.";

/** The warning for a resolution with this short edge, or none on a normal machine. */
export function lowPerformanceResolutionWarning(
  shortEdge: number,
  lowPerformanceMachine: boolean | undefined,
): string | undefined {
  return lowPerformanceMachine === true && shortEdge >= LOW_PERFORMANCE_MIN_SHORT_EDGE
    ? LOW_PERFORMANCE_RESOLUTION_WARNING
    : undefined;
}
