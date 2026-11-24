import { useEffect, useRef } from "react";

import type { FeatureKeyframe } from "../../types";

/**
 * One `edit(update)` for a list of stills that is changed by async work. An add can end after the
 * render that started it, and two can end before the next render. Each edit must start from the
 * list as the last edit left it, not from the `value` of its own render. It returns the list
 * before and after the edit.
 */
export function useKeyframeEdit(
  value: readonly FeatureKeyframe[],
  onChange: (next: FeatureKeyframe[]) => void,
): (update: (current: FeatureKeyframe[]) => FeatureKeyframe[]) => {
  current: FeatureKeyframe[];
  next: FeatureKeyframe[];
} {
  const latest = useRef<FeatureKeyframe[]>([...value]);
  useEffect(() => {
    latest.current = [...value];
  }, [value]);
  return (update) => {
    const current = latest.current;
    const next = update(current);
    latest.current = next;
    onChange(next);
    return { current, next };
  };
}
