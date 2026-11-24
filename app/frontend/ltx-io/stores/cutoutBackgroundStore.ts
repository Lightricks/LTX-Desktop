import { create } from "zustand";

export type CutoutBackgroundKind = "checkerboard" | "color";

export const DEFAULT_CUTOUT_COLOR = "#000000";

const HEX_COLOR = /^#[0-9a-f]{6}$/i;

export function isHexColor(value: string): boolean {
  return HEX_COLOR.test(value);
}

/**
 * Press on the color swatch. A swatch that is not selected becomes selected.
 * A selected swatch opens the color picker.
 */
export function pressColorSwatch(current: CutoutBackgroundKind): {
  background: CutoutBackgroundKind;
  openPicker: boolean;
} {
  return { background: "color", openPicker: current === "color" };
}

type CutoutBackgroundStore = {
  background: CutoutBackgroundKind;
  color: string;
  selectCheckerboard: () => void;
  selectColor: () => void;
  /** Sets the color and selects it. An invalid value is ignored. */
  setColor: (color: string) => void;
};

/**
 * Session-wide backdrop for AlphaGen cutouts. Every result keeps the last
 * choice of the user until the app closes.
 */
export const useCutoutBackgroundStore = create<CutoutBackgroundStore>((set) => ({
  background: "checkerboard",
  color: DEFAULT_CUTOUT_COLOR,
  selectCheckerboard: () => set({ background: "checkerboard" }),
  selectColor: () => set({ background: "color" }),
  setColor: (color) => {
    if (!isHexColor(color)) return;
    set({ background: "color", color: color.toLowerCase() });
  },
}));
