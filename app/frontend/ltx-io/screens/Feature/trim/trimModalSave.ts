import type { CloseReason } from "../../../components/shared/Modal/modalTypes.ts";
import type { MediaTrimRequest } from "../fields/prepareDurationLimitedMediaImport.ts";

export type TrimModalSave = {
  /** What the modal's Done button runs. Rejects so the modal can report it. */
  save: (startSec: number, endSec: number) => Promise<void>;
  /** True while a save is in flight, so dismissal stays blocked. */
  isSaving: () => boolean;
};

/**
 * Binds the modal's Done button to the import request's own trim-then-accept.
 * There is deliberately no second copy of that logic here: whatever
 * `prepareDurationLimitedMediaImport` built is what Done runs.
 *
 * A rejection propagates and the modal stays open, so the selection survives a
 * retry and the failure is reported inside the modal rather than behind it.
 */
export function createTrimModalSave(
  request: Pick<MediaTrimRequest, "onSave">,
  hide: (reason: CloseReason) => void,
): TrimModalSave {
  let inFlight: Promise<void> | null = null;

  return {
    isSaving: () => inFlight !== null,
    // A repeat Done shares the first save's outcome instead of resolving early.
    save: (startSec, endSec) => {
      inFlight ??= (async () => {
        try {
          await request.onSave(startSec, endSec);
        } finally {
          inFlight = null;
        }
        hide("apply");
      })();
      return inFlight;
    },
  };
}
