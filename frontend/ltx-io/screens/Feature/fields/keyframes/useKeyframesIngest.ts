import { randomUUID } from "../../../../../lib/random-uuid.ts";
import type { FeatureKeyframe } from "../../types";
import { useImageAssetsIngest } from "../useImageAssetsIngest";
import { addKeyframes } from "./keyframesModel";

/**
 * Add images to a list of stills: drop files on the timeline, drop them anywhere else, or pick
 * them from the library. Files dropped on the timeline go near the frame under the pointer.
 * All the other images spread over the free stretches, from `preferredFrame` when it is free.
 * `edit` changes the list, and `onAdded` gets the still that was added last.
 */
export function useKeyframesIngest({
  label,
  maxCount,
  count,
  lastFrame,
  preferredFrame,
  edit,
  onAdded,
}: {
  label: string;
  maxCount: number;
  count: number;
  lastFrame: number;
  preferredFrame: number;
  edit: (update: (current: FeatureKeyframe[]) => FeatureKeyframe[]) => {
    current: FeatureKeyframe[];
    next: FeatureKeyframe[];
  };
  onAdded: (keyframe: FeatureKeyframe) => void;
}) {
  return useImageAssetsIngest<number>({
    label,
    maxCount,
    count,
    // `droppedFrame` is the frame under the pointer when files were dropped on the timeline.
    onAdd: (assets, droppedFrame) => {
      const { current, next } = edit((keyframes) =>
        addKeyframes(keyframes, assets, {
          lastFrame,
          preferredFrame: droppedFrame ?? preferredFrame,
          placement: droppedFrame === undefined ? "spread" : "near",
          maxCount,
          createId: randomUUID,
        }),
      );
      const added = next.at(-1);
      if (added && next.length > current.length) onAdded(added);
    },
  });
}
