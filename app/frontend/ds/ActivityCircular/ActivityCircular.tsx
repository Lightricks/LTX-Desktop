import clsx from "clsx";

import styles from "./ActivityCircular.module.scss";

const clamp = (value: number, min: number, max: number) =>
  Math.min(Math.max(value, min), max);

// Constants for SVG dimensions and calculations
const VIEWBOX_SIZE = 18; // The viewbox is intentionally fixed to 18x18 to make the stroke grow proportionally with the size

const STROKE_WIDTH = 2;
const SPINNER_PROGRESS_PERCENT = 25;

const CENTER = VIEWBOX_SIZE / 2;
const RADIUS = VIEWBOX_SIZE / 2 - STROKE_WIDTH;

export const ACTIVITY_CIRCULAR_APPEARANCE_MAP = {
  default: "default",
  "over-background": "over-background",
} as const;

export type Appearance = keyof typeof ACTIVITY_CIRCULAR_APPEARANCE_MAP;

export type ActivityCircularProps = {
  /** The size of the activity indicator */
  size?: number;
  /** The appearance of the activity indicator */
  appearance?: Appearance;
  /** Progress value between 0 and 100. If not provided, the component acts as a spinner */
  progress?: number;
  /** Test ID for testing purposes */
  "data-testid"?: string;
};

/** This is a circular activity indicator that shows progress from 0 to 100, or acts as a spinner when progress is not provided. */
export function ActivityCircular(props: ActivityCircularProps) {
  const {
    size = 16,
    appearance = "default",
    progress,
    "data-testid": testId = "activity-circular",
  } = props;

  const isProgressUndefined = progress === undefined;

  // Calculate stroke dash array and offset for progress
  const circumference = 2 * Math.PI * RADIUS;
  // If DOES NOT HAVE PROGRESS, show 25% of the circle like a spinner
  const displayProgress = isProgressUndefined
    ? SPINNER_PROGRESS_PERCENT
    : clamp(progress, 0, 100);
  const progressOffset = circumference - (displayProgress / 100) * circumference;

  return (
    <svg
      className={clsx(
        styles.activityCircular,
        isProgressUndefined && styles.spinner,
        appearance === "over-background" && styles.activityCircularOverBackground,
      )}
      xmlns="http://www.w3.org/2000/svg"
      width={size}
      height={size}
      viewBox={`0 0 ${VIEWBOX_SIZE} ${VIEWBOX_SIZE}`}
      fill="none"
      role={isProgressUndefined ? "status" : "progressbar"}
      aria-label="Loading"
      aria-valuenow={isProgressUndefined ? undefined : displayProgress}
      aria-valuemin={isProgressUndefined ? undefined : 0}
      aria-valuemax={isProgressUndefined ? undefined : 100}
      aria-busy={isProgressUndefined ? true : undefined}
      data-testid={testId}
    >
      <circle
        className={styles.activityCircularCircle}
        cx={CENTER}
        cy={CENTER}
        r={RADIUS}
        strokeWidth={STROKE_WIDTH}
        fill="none"
      />
      <circle
        className={styles.activityCircularCircleLoadingPath}
        cx={CENTER}
        cy={CENTER}
        r={RADIUS}
        strokeWidth={STROKE_WIDTH}
        fill="none"
        strokeDasharray={circumference}
        strokeDashoffset={progressOffset}
        strokeLinecap="round"
        transform={`rotate(-90 ${CENTER} ${CENTER})`}
      />
    </svg>
  );
}
