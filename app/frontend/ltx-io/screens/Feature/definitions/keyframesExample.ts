import type { KeyframesSeedKind } from "../../../../../shared/explore-seed-filenames.ts";
import { DEFAULT_KEYFRAME_STRENGTH } from "../../../../lib/keyframe-strength.ts";
import type { AssetRef, FeatureKeyframe } from "../types.ts";

/** The packaged stills of the Keyframes example, one for each season. */
export type KeyframesExampleSeed = Readonly<Record<KeyframesSeedKind, AssetRef>>;

export const KEYFRAMES_EXAMPLE_PROMPT =
  "A continuous, fast natural time-lapse from a fixed camera. A traveler stands motionless on a cliff, looking across the same mountain valley. Clouds race across the sky and sunlight and shadows sweep over the landscape. Summer changes through visible physical processes: leaves gradually turn brown and gold, then fall as branches become bare; winter snow blows in and steadily piles up across the trees, valley, and cliff; spring sunlight returns, snow visibly melts and runs into streams, and fresh buds open into new leaves. The valley grows lush and green, returning to bright summer sunshine. Make the changes happen continuously and organically in the scene, like real time passing—no fades, dissolves, cuts, or sudden scene swaps. Keep the traveler, clothing, backpack, pose, viewpoint, and composition unchanged throughout.";

export const KEYFRAMES_EXAMPLE_NEGATIVE_PROMPT =
  "Crossfades, dissolves, fade transitions, cuts, jump cuts, instant seasonal swaps, scenery popping in or out, frozen clouds, static foliage, sudden snow appearing, snow disappearing instantly, abrupt changes in lighting, camera movement, zoom, pan, tilt, shake, traveler walking, traveler turning, changing pose, changing clothes, changing backpack, duplicate traveler, distorted landscape, flicker, jitter, unnatural morphing";

/** The clip the example stills belong to: 6 s at 24 fps ends on frame 144. */
export const KEYFRAMES_EXAMPLE_SETTINGS = {
  resolution: "720p",
  duration: 6,
  fps: 24,
} as const;

/** Summer opens and closes the clip, so one packaged still fills two slots. */
const KEYFRAMES_EXAMPLE_SLOTS: readonly { seed: KeyframesSeedKind; frameIndex: number }[] = [
  { seed: "keyframes-summer-image", frameIndex: 0 },
  { seed: "keyframes-autumn-image", frameIndex: 36 },
  { seed: "keyframes-winter-image", frameIndex: 72 },
  { seed: "keyframes-spring-image", frameIndex: 108 },
  { seed: "keyframes-summer-image", frameIndex: 144 },
];

export function exampleKeyframes(seed: KeyframesExampleSeed): FeatureKeyframe[] {
  return KEYFRAMES_EXAMPLE_SLOTS.map((slot, index) => ({
    id: `example-keyframe-${index}`,
    asset: seed[slot.seed],
    frameIndex: slot.frameIndex,
    strength: DEFAULT_KEYFRAME_STRENGTH,
  }));
}
