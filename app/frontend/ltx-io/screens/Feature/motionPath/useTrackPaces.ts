import { useCallback, useState } from "react";

import type { MotionPath } from "../../../lib/motionPathSampler";
import {
  type TrackPace,
  initialTrackPaceState,
  paceOfTrack,
  withDuration,
  withPaceUpdate,
  withSamePace,
} from "./trackPaceState";

export type TrackPaces = {
  durationSec: number;
  /** True when every track uses one pace. */
  samePace: boolean;
  paceOf: (trackId: number) => TrackPace;
  setDurationSec: (durationSec: number) => void;
  updatePace: (trackId: number, partial: Partial<TrackPace>) => void;
  /** `currentTrackId` is the track on screen. */
  setSamePace: (samePace: boolean, currentTrackId: number, trackIds: readonly number[]) => void;
};

/** The timing of the tracks in the editor. The rules are in `trackPaceState`. */
export function useTrackPaces(paths: MotionPath[], initialDurationSec: number): TrackPaces {
  const [state, setState] = useState(() => initialTrackPaceState(paths, initialDurationSec));

  const paceOf = useCallback((trackId: number) => paceOfTrack(state, trackId), [state]);
  const setDurationSec = useCallback(
    (durationSec: number) => setState((current) => withDuration(current, durationSec)),
    [],
  );
  const updatePace = useCallback(
    (trackId: number, partial: Partial<TrackPace>) =>
      setState((current) => withPaceUpdate(current, trackId, partial)),
    [],
  );
  const setSamePace = useCallback(
    (samePace: boolean, currentTrackId: number, trackIds: readonly number[]) =>
      setState((current) => withSamePace(current, samePace, currentTrackId, trackIds)),
    [],
  );

  return { durationSec: state.durationSec, samePace: state.samePace, paceOf, setDurationSec, updatePace, setSamePace };
}
