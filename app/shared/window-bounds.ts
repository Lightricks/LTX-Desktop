/** Installer default is 933×600. The app opens larger at 16:9. */
export const INSTALL_WINDOW_BOUNDS = {
  width: 933,
  height: 600,
  minWidth: 800,
  minHeight: 600,
} as const

export const APP_WINDOW_BOUNDS = {
  width: 1440,
  height: 810,
  minWidth: 1024,
  minHeight: 576,
} as const

export interface Bounds {
  x: number
  y: number
  width: number
  height: number
}

const SCREEN_MARGIN = 24

/** Pure geometry: shrink a window to fit inside a work area. */
export function fitToWorkArea(
  width: number,
  height: number,
  areaWidth: number,
  areaHeight: number,
): { width: number; height: number } {
  const maxW = Math.max(1, areaWidth - SCREEN_MARGIN)
  const maxH = Math.max(1, areaHeight - SCREEN_MARGIN)
  const scale = Math.min(1, maxW / width, maxH / height)
  return {
    width: Math.round(width * scale),
    height: Math.round(height * scale),
  }
}

/** Pure geometry: center a window of the given size inside a work area. */
export function centeredBounds(
  workArea: Bounds,
  width: number,
  height: number,
): Bounds {
  const fitted = fitToWorkArea(width, height, workArea.width, workArea.height)
  return {
    x: Math.round(workArea.x + (workArea.width - fitted.width) / 2),
    y: Math.round(workArea.y + (workArea.height - fitted.height) / 2),
    width: fitted.width,
    height: fitted.height,
  }
}
