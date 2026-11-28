import {
  MOTION_PATH_DEFAULT_EASING,
  type MotionPath,
  type MotionPathEasing,
  clampMotionPathHolds,
} from "../../../lib/motionPathSampler.ts";

/** The timing of one track. The clip length is shared, so it is not here. */
export type TrackPace = {
  easing: MotionPathEasing;
  startHoldSec: number;
  endHoldSec: number;
};

/**
 * The timing of the tracks in the editor. The clip length is always shared. The pace is
 * shared too while `samePace` is true. Then `ownPaces`, keyed by track id, is not used.
 * The holds of every pace stay under the clip length.
 */
export type TrackPaceState = {
  durationSec: number;
  samePace: boolean;
  sharedPace: TrackPace;
  ownPaces: Record<number, TrackPace>;
};

function paceOfPath(path: MotionPath | undefined): TrackPace {
  return {
    easing: path?.easing ?? MOTION_PATH_DEFAULT_EASING,
    startHoldSec: path?.startHoldSec ?? 0,
    endHoldSec: path?.endHoldSec ?? 0,
  };
}

function isSamePace(a: TrackPace, b: TrackPace): boolean {
  return (
    a.easing === b.easing && a.startHoldSec === b.startHoldSec && a.endHoldSec === b.endHoldSec
  );
}

function clampPace(pace: TrackPace, durationSec: number): TrackPace {
  return { ...pace, ...clampMotionPathHolds({ durationSec, ...pace }) };
}

/** The tracks of a saved run start at their own pace, with ids 0, 1, 2 in order. */
export function initialTrackPaceState(paths: MotionPath[], durationSec: number): TrackPaceState {
  const first = paceOfPath(paths[0]);
  return {
    durationSec,
    samePace: paths.every((path) => isSamePace(paceOfPath(path), first)),
    sharedPace: first,
    ownPaces: Object.fromEntries(paths.map((path, index) => [index, paceOfPath(path)])),
  };
}

export function paceOfTrack(state: TrackPaceState, trackId: number): TrackPace {
  return state.samePace ? state.sharedPace : (state.ownPaces[trackId] ?? state.sharedPace);
}

export function withDuration(state: TrackPaceState, durationSec: number): TrackPaceState {
  return {
    ...state,
    durationSec,
    sharedPace: clampPace(state.sharedPace, durationSec),
    ownPaces: Object.fromEntries(
      Object.entries(state.ownPaces).map(([id, pace]) => [id, clampPace(pace, durationSec)]),
    ),
  };
}

/** Change the shared pace, or the pace of this track when the paces are individual. */
export function withPaceUpdate(
  state: TrackPaceState,
  trackId: number,
  partial: Partial<TrackPace>,
): TrackPaceState {
  const next = clampPace({ ...paceOfTrack(state, trackId), ...partial }, state.durationSec);
  return state.samePace
    ? { ...state, sharedPace: next }
    : { ...state, ownPaces: { ...state.ownPaces, [trackId]: next } };
}

/**
 * Turning it on: the pace of the track on screen becomes the shared pace.
 * Turning it off: every track starts from the shared pace, not from an older individual one.
 */
export function withSamePace(
  state: TrackPaceState,
  samePace: boolean,
  currentTrackId: number,
  trackIds: readonly number[],
): TrackPaceState {
  if (samePace === state.samePace) return state;
  if (samePace) {
    return { ...state, samePace, sharedPace: paceOfTrack(state, currentTrackId) };
  }
  return {
    ...state,
    samePace,
    ownPaces: {
      ...state.ownPaces,
      ...Object.fromEntries(trackIds.map((id) => [id, state.sharedPace])),
    },
  };
}
