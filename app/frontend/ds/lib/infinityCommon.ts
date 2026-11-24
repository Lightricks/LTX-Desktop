// Small geometry helpers used by the design system, kept local so `frontend/ds/`
// has no external workspace dependency.

export interface Size {
  width: number;
  height: number;
}

export function aspectRatioToNumber(aspectRatio: string): number {
  const [width, height] = aspectRatio.split(":").map(Number);
  if (!width || !height) {
    throw new Error("Invalid aspect ratio");
  }
  return width / height;
}
