import { withBetBackstop } from "../../lib/budget-risk.ts";

/** Queued-job codes explained by fixed copy (no runtime inputs). */
export const CUDA_OOM = "CUDA_OOM";
export const JOB_TOO_LARGE = "JOB_TOO_LARGE";
export const IMAGE_BLOCKED = "IMAGE_BLOCKED_BY_SAFETY_CHECKER";

export const CUDA_OOM_MESSAGE =
  "The GPU ran out of memory during this generation. Close other heavy apps (like games, browsers, or video editors) and retry, or lower the resolution or duration.";
// Cause-neutral by design: the record carries the code only, so this one
// line covers GPU-memory drift, RAM shortfall, unsupported machines, and
// over-size sources alike (threading the underlying cause through is a
// filed follow-up). No close-apps advice: on Macs the pool is a fixed cap,
// so closing apps never unlocks cells there.
export const JOB_TOO_LARGE_MESSAGE =
  "This job no longer fits on this machine. Lower the resolution or duration and retry.";

export const IMAGE_BLOCKED_MESSAGE =
  "The image provider's safety checker blocked this prompt. Change the prompt and try again.";

const STATIC_FAILURE_MESSAGES: Record<string, string> = {
  [CUDA_OOM]: CUDA_OOM_MESSAGE,
  [JOB_TOO_LARGE]: JOB_TOO_LARGE_MESSAGE,
  [IMAGE_BLOCKED]: IMAGE_BLOCKED_MESSAGE,
};

export type StaticFailureNotice = {
  body: string;
  /** Never set — no Settings shortcut fixes VRAM. Present for the shared render path. */
  openSettingsLabel?: undefined;
};

/**
 * Explain a static execute-time failure on an Explore result: a mid-run OOM
 * or a budget miss while queued. Null for anything else — conditional
 * failures (API key, embedding) have their own matchers.
 */
export function staticFailureNotice(input: {
  errorCode: string | null | undefined;
  /** The job carried a create-time risk snapshot. */
  ranOnBet?: boolean;
}): StaticFailureNotice | null {
  if (input.errorCode == null) return null;
  const body = STATIC_FAILURE_MESSAGES[input.errorCode];
  if (body === undefined) return null;
  return { body: withBetBackstop(body, input.ranOnBet === true) };
}
