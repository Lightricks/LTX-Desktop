import { nudgeKeyframe } from "../../../../../lib/keyframe-controls.ts";
import {
  DEFAULT_KEYFRAME_STRENGTH,
  nudgeKeyframeStrength,
} from "../../../../../lib/keyframe-strength.ts";
import { findNearestFreeFrameIndex, pickFreeFrameIndex } from "../../../../../lib/keyframe-timeline.ts";
import type { AssetRef, FeatureKeyframe } from "../../types.ts";

/**
 * The side of a marker on the timeline, in pixels, for a clip with `count` stills. More stills
 * need smaller markers to overlap less. The strip is as high as the biggest size, so a smaller
 * marker changes the thumbnail only.
 */
export function markerSizeForCount(count: number): number {
  if (count >= 9) return 28;
  if (count >= 6) return 36;
  return 44;
}

export type KeyframePatch = Partial<Pick<FeatureKeyframe, "frameIndex" | "strength">>;

/**
 * Put new stills on the timeline. A still with no free frame left is dropped, as is any past
 * `maxCount`. With `placement: "spread"` a batch spreads over the widest free stretches.
 * With `placement: "near"` each still takes the free frame nearest `preferredFrame`,
 * so a still dropped on the timeline stays where the user dropped it.
 */
export function addKeyframes(
  keyframes: readonly FeatureKeyframe[],
  assets: readonly AssetRef[],
  options: {
    lastFrame: number;
    preferredFrame: number;
    placement: "spread" | "near";
    maxCount: number;
    createId: () => string;
  },
): FeatureKeyframe[] {
  const next = [...keyframes];
  for (const asset of assets) {
    if (next.length >= options.maxCount) break;
    // The two helpers take the last frame and the wanted frame in opposite orders.
    const frameIndex =
      options.placement === "near"
        ? findNearestFreeFrameIndex(next, options.preferredFrame, options.lastFrame)
        : pickFreeFrameIndex(next, options.lastFrame, options.preferredFrame);
    if (frameIndex === null) break;
    next.push({
      id: options.createId(),
      asset,
      frameIndex,
      strength: DEFAULT_KEYFRAME_STRENGTH,
    });
  }
  return next;
}

/**
 * Move the stills so the gap between each neighbor pair is the same, the first on frame 0 and
 * the last on `lastFrame`. The time order and every id stay. A frame is a whole number, so a gap
 * can differ by one frame. Fewer than two stills, or too few frames for them, change nothing.
 */
export function spreadKeyframes(
  keyframes: readonly FeatureKeyframe[],
  lastFrame: number,
): FeatureKeyframe[] {
  const ordered = byFrame(keyframes);
  if (ordered.length < 2 || lastFrame < ordered.length - 1) return [...keyframes];
  const frameById = new Map(
    ordered.map((keyframe, index) => [
      keyframe.id,
      Math.round((index * lastFrame) / (ordered.length - 1)),
    ]),
  );
  return keyframes.map((keyframe) => ({
    ...keyframe,
    frameIndex: frameById.get(keyframe.id) ?? keyframe.frameIndex,
  }));
}

/**
 * Two stills trade places on the timeline: each takes the frame of the other. Each keeps its
 * image and its strength. An unknown id, or the same id twice, changes nothing.
 */
export function swapKeyframeFrames(
  keyframes: readonly FeatureKeyframe[],
  idA: string,
  idB: string,
): FeatureKeyframe[] {
  const first = keyframes.find((keyframe) => keyframe.id === idA);
  const second = keyframes.find((keyframe) => keyframe.id === idB);
  if (!first || !second || first.id === second.id) return [...keyframes];
  return keyframes.map((keyframe) => {
    if (keyframe.id === first.id) return { ...keyframe, frameIndex: second.frameIndex };
    if (keyframe.id === second.id) return { ...keyframe, frameIndex: first.frameIndex };
    return keyframe;
  });
}

/**
 * Give the edits of the user to `current`. `base` is the list the user started from, and `draft`
 * is the list after the edits. A still in `base` that is not in `draft` was removed. A still in
 * `draft` that is not in `base` was added. Any other still in `draft` gets its frame and strength.
 * `current` keeps its images and its order, and a still that is in `current` only, such as an
 * image that finished loading meanwhile, stays as it is and keeps its frame. A still that
 * wants a taken frame goes to the nearest free frame, so no two stills share a frame. A still
 * added past `maxCount` is dropped.
 */
export function applyKeyframeLayout(
  current: readonly FeatureKeyframe[],
  base: readonly FeatureKeyframe[],
  draft: readonly FeatureKeyframe[],
  options: { lastFrame: number; maxCount: number },
): FeatureKeyframe[] {
  const { lastFrame, maxCount } = options;
  const draftById = new Map(draft.map((keyframe) => [keyframe.id, keyframe]));
  const baseIds = new Set(base.map((keyframe) => keyframe.id));
  const currentIds = new Set(current.map((keyframe) => keyframe.id));
  const kept = current.filter((keyframe) => draftById.has(keyframe.id) || !baseIds.has(keyframe.id));
  const taken: { frameIndex: number }[] = kept.filter((keyframe) => !draftById.has(keyframe.id));
  const placed = kept.map((keyframe) => {
    const edited = draftById.get(keyframe.id);
    if (!edited) return keyframe;
    const frameIndex =
      findNearestFreeFrameIndex(taken, edited.frameIndex, lastFrame) ?? keyframe.frameIndex;
    taken.push({ frameIndex });
    return { ...keyframe, frameIndex, strength: edited.strength };
  });
  const added: FeatureKeyframe[] = [];
  for (const keyframe of draft) {
    if (baseIds.has(keyframe.id) || currentIds.has(keyframe.id)) continue;
    if (placed.length + added.length >= maxCount) break;
    const frameIndex = findNearestFreeFrameIndex(taken, keyframe.frameIndex, lastFrame);
    if (frameIndex === null) break;
    taken.push({ frameIndex });
    added.push({ ...keyframe, frameIndex });
  }
  return [...placed, ...added];
}

/**
 * The free frame nearest `targetFrame` for the still `id`. Its own frame does not count as taken.
 * Null when no frame is free.
 */
export function freeFrameFor(
  keyframes: readonly FeatureKeyframe[],
  id: string,
  targetFrame: number,
  lastFrame: number,
): number | null {
  const others = keyframes.filter((keyframe) => keyframe.id !== id);
  return findNearestFreeFrameIndex(others, targetFrame, lastFrame);
}

/** The keys that edit a marker: left and right move it by one frame, up and down set its strength. */
export const KEYFRAME_EDIT_KEYS: readonly string[] = ["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"];

/** The change a key press makes to a still, or null when the key changes nothing. */
export function keyframePatchForKey(
  keyframes: readonly FeatureKeyframe[],
  keyframe: FeatureKeyframe,
  key: string,
  lastFrame: number,
): KeyframePatch | null {
  if (key === "ArrowLeft" || key === "ArrowRight") {
    const frameIndex = nudgeKeyframe(keyframes, keyframe.id, key === "ArrowLeft" ? -1 : 1, lastFrame);
    return frameIndex === null ? null : { frameIndex };
  }
  if (key === "ArrowUp" || key === "ArrowDown") {
    return { strength: nudgeKeyframeStrength(keyframe.strength, key === "ArrowUp" ? 1 : -1) };
  }
  return null;
}

export function updateKeyframe(
  keyframes: readonly FeatureKeyframe[],
  id: string,
  patch: KeyframePatch | { asset: AssetRef },
): FeatureKeyframe[] {
  return keyframes.map((keyframe) =>
    keyframe.id === id ? { ...keyframe, ...patch } : keyframe,
  );
}

export function removeKeyframe(
  keyframes: readonly FeatureKeyframe[],
  id: string,
): FeatureKeyframe[] {
  return keyframes.filter((keyframe) => keyframe.id !== id);
}

/** The stills in time order. The list keeps the order the user added them in. */
export function byFrame(
  keyframes: readonly FeatureKeyframe[],
): FeatureKeyframe[] {
  return [...keyframes].sort((a, b) => a.frameIndex - b.frameIndex);
}

/**
 * The place of the still `id` in the preview. `orderedIds` are the stills in time order, and
 * `positions` are as `orderLightboxAssets` gives them. Null when the still is not in the list,
 * or has nothing to show.
 */
export function stillPosition(
  orderedIds: readonly string[],
  positions: readonly (number | null)[],
  id: string,
): number | null {
  const index = orderedIds.indexOf(id);
  return index === -1 ? null : (positions[index] ?? null);
}

/** The id of the still at a place of the preview, the reverse of `stillPosition`. */
export function stillIdAtPosition(
  orderedIds: readonly string[],
  positions: readonly (number | null)[],
  position: number,
): string | undefined {
  const index = positions.indexOf(position);
  return index === -1 ? undefined : orderedIds[index];
}

/**
 * The preview items for the stills in `assetIds`, one for each still that has a loaded asset.
 * An asset used by two stills is listed twice, so each still keeps its own place in the preview.
 * `positions[i]` is the place of still `i` in `items`. It is null while that asset is not loaded,
 * and then the still has nothing to show.
 */
export function orderLightboxAssets<T>(
  assetIds: readonly string[],
  loaded: ReadonlyMap<string, T>,
): { items: T[]; positions: (number | null)[] } {
  const items: T[] = [];
  const positions = assetIds.map((assetId) => {
    const asset = loaded.get(assetId);
    if (!asset) return null;
    items.push(asset);
    return items.length - 1;
  });
  return { items, positions };
}
