import { clsx } from "clsx";
import { type KeyboardEvent, useCallback, useRef, useState } from "react";

import { Tooltip } from "@ds/Tooltip/Tooltip";

import {
  pressColorSwatch,
  useCutoutBackgroundStore,
} from "../../../../stores/cutoutBackgroundStore";

import styles from "./CutoutBackgroundStrip.module.scss";
import { CutoutColorPicker } from "./CutoutColorPicker";

/**
 * Picks the backdrop the cutout plays over. The first swatch is the
 * transparency grid. The second is one color: a press selects it, and a press
 * on the selected color opens the picker, which changes the backdrop live.
 */
export function CutoutBackgroundStrip() {
  const stripRef = useRef<HTMLDivElement>(null);
  const checkerboardRef = useRef<HTMLButtonElement>(null);
  const colorRef = useRef<HTMLButtonElement>(null);
  const [pickerOpen, setPickerOpen] = useState(false);
  const closePicker = useCallback(() => setPickerOpen(false), []);
  const background = useCutoutBackgroundStore((state) => state.background);
  const color = useCutoutBackgroundStore((state) => state.color);
  const selectCheckerboard = useCutoutBackgroundStore(
    (state) => state.selectCheckerboard,
  );
  const selectColor = useCutoutBackgroundStore((state) => state.selectColor);
  const setColor = useCutoutBackgroundStore((state) => state.setColor);

  const onColorPress = () => {
    const press = pressColorSwatch(background);
    selectColor();
    if (press.openPicker) setPickerOpen((open) => !open);
  };

  const onCheckerboardPress = () => {
    selectCheckerboard();
    closePicker();
  };

  // A radio group is one Tab stop. The arrow keys move the selection and the focus.
  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (!["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(event.key)) return;
    event.preventDefault();
    if (background === "checkerboard") {
      selectColor();
      colorRef.current?.focus();
    } else {
      onCheckerboardPress();
      checkerboardRef.current?.focus();
    }
  };

  return (
    <div
      ref={stripRef}
      className={styles.strip}
      role="radiogroup"
      aria-label="Cutout background"
      onKeyDown={onKeyDown}
      data-testid="cutout-background-strip"
    >
      <Tooltip content="Transparency grid" side="bottom">
        <span className={styles.slot}>
          <button
            ref={checkerboardRef}
            type="button"
            role="radio"
            aria-checked={background === "checkerboard"}
            tabIndex={background === "checkerboard" ? 0 : -1}
            aria-label="Transparency grid"
            className={clsx(
              styles.swatch,
              styles.checkerboard,
              background === "checkerboard" && styles.selected,
            )}
            onClick={onCheckerboardPress}
          />
        </span>
      </Tooltip>
      <Tooltip
        content={background === "color" ? "Change color" : "Color"}
        side="bottom"
      >
        <span className={styles.slot}>
          <button
            ref={colorRef}
            type="button"
            role="radio"
            aria-checked={background === "color"}
            tabIndex={background === "color" ? 0 : -1}
            aria-label="Background color"
            className={clsx(styles.swatch, background === "color" && styles.selected)}
            style={{ backgroundColor: color }}
            onClick={onColorPress}
          />
        </span>
      </Tooltip>
      {pickerOpen ? (
        <CutoutColorPicker
          anchorRef={stripRef}
          returnFocusRef={colorRef}
          color={color}
          onChange={setColor}
          onClose={closePicker}
        />
      ) : null}
    </div>
  );
}
