import {
  type KeyboardEvent,
  type PointerEvent,
  type RefObject,
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from "react";
import { createPortal } from "react-dom";

import { useThemedPortalContainer } from "@ds/styles/themes/useTheme";

import { hexToHsv, hsvToHex, type Hsv } from "../../../../lib/colorMath";
import { isHexColor } from "../../../../stores/cutoutBackgroundStore";

import styles from "./CutoutColorPicker.module.scss";
import { placeBesideAnchor } from "./placeBesideAnchor";

const KEY_STEP = 0.01;
const KEY_STEP_LARGE = 0.1;

const clamp01 = (value: number): number => Math.min(1, Math.max(0, value));

/**
 * A color picker that opens beside the strip, on the left, so it covers the form
 * and not the preview. The native picker cannot be placed: on macOS it is a system
 * panel. Every change goes to `onChange` at once, so the backdrop follows live.
 */
export function CutoutColorPicker({
  anchorRef,
  returnFocusRef,
  color,
  onChange,
  onClose,
}: {
  anchorRef: RefObject<HTMLElement | null>;
  /** Gets the focus back when a key press closes the picker. */
  returnFocusRef: RefObject<HTMLElement | null>;
  color: string;
  onChange: (color: string) => void;
  onClose: () => void;
}) {
  const themedContainer = useThemedPortalContainer();
  const popoverRef = useRef<HTMLDivElement>(null);
  const [position, setPosition] = useState<{ left: number; top: number } | null>(null);
  const [hsv, setHsv] = useState<Hsv>(() => hexToHsv(color));
  const [hexText, setHexText] = useState(() => color.slice(1));
  // Fullscreen shows only that element, so the picker goes inside it.
  const fullscreen = document.fullscreenElement;
  const container = fullscreen instanceof HTMLElement ? fullscreen : themedContainer;

  useLayoutEffect(() => {
    const popover = popoverRef.current;
    const anchor = anchorRef.current;
    if (!popover || !anchor) return;
    setPosition(
      placeBesideAnchor(
        anchor.getBoundingClientRect(),
        { width: popover.offsetWidth, height: popover.offsetHeight },
        { width: window.innerWidth, height: window.innerHeight },
      ),
    );
  }, [anchorRef, container]);

  // A hidden element cannot take focus, so this waits for the position.
  const placed = position != null;
  useEffect(() => {
    if (placed) popoverRef.current?.querySelector<HTMLElement>("[role=slider]")?.focus();
  }, [placed]);

  const closeWithFocus = useCallback(() => {
    returnFocusRef.current?.focus();
    onClose();
  }, [onClose, returnFocusRef]);

  useEffect(() => {
    const onPointerDown = (event: globalThis.PointerEvent) => {
      const target = event.target;
      if (!(target instanceof Node)) return;
      if (popoverRef.current?.contains(target) || anchorRef.current?.contains(target)) return;
      onClose();
    };
    const onKeyDown = (event: globalThis.KeyboardEvent) => {
      if (event.key === "Escape") closeWithFocus();
    };
    document.addEventListener("pointerdown", onPointerDown);
    document.addEventListener("keydown", onKeyDown);
    window.addEventListener("resize", onClose);
    document.addEventListener("fullscreenchange", onClose);
    return () => {
      document.removeEventListener("pointerdown", onPointerDown);
      document.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("resize", onClose);
      document.removeEventListener("fullscreenchange", onClose);
    };
  }, [anchorRef, closeWithFocus, onClose]);

  const apply = (next: Hsv) => {
    const hex = hsvToHex(next);
    setHsv(next);
    setHexText(hex.slice(1));
    onChange(hex);
  };

  const pickFromPointer = (event: PointerEvent<HTMLDivElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    apply({
      h: hsv.h,
      s: clamp01((event.clientX - rect.left) / rect.width),
      v: 1 - clamp01((event.clientY - rect.top) / rect.height),
    });
  };

  const onAreaKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    const step = event.shiftKey ? KEY_STEP_LARGE : KEY_STEP;
    const moves: Record<string, Hsv | undefined> = {
      ArrowLeft: { ...hsv, s: clamp01(hsv.s - step) },
      ArrowRight: { ...hsv, s: clamp01(hsv.s + step) },
      ArrowUp: { ...hsv, v: clamp01(hsv.v + step) },
      ArrowDown: { ...hsv, v: clamp01(hsv.v - step) },
    };
    const next = moves[event.key];
    if (next == null) return;
    event.preventDefault();
    apply(next);
  };

  const onHexChange = (text: string) => {
    const digits = text.replace(/[^0-9a-f]/gi, "").slice(0, 6);
    setHexText(digits);
    const hex = `#${digits}`;
    if (!isHexColor(hex)) return;
    setHsv(hexToHsv(hex));
    onChange(hex.toLowerCase());
  };

  if (container == null) return null;
  return createPortal(
    <div
      ref={popoverRef}
      className={styles.popover}
      style={{
        left: position?.left ?? 0,
        top: position?.top ?? 0,
        visibility: position == null ? "hidden" : "visible",
      }}
      // React events cross a portal. The result frame must not see these clicks.
      onClick={(event) => event.stopPropagation()}
      onDoubleClick={(event) => event.stopPropagation()}
      onKeyDown={(event) => {
        // Keys in the picker must not reach the strip, which moves its selection on arrows.
        event.stopPropagation();
        // The popover sits at the end of the page. Tab at its edges goes back to the swatch.
        if (event.key !== "Tab") return;
        const stops = event.currentTarget.querySelectorAll<HTMLElement>("[role=slider], input");
        const edge = stops[event.shiftKey ? 0 : stops.length - 1];
        if (document.activeElement !== edge) return;
        event.preventDefault();
        closeWithFocus();
      }}
      role="dialog"
      aria-label="Background color"
      data-testid="cutout-color-picker"
    >
      <div
        className={styles.area}
        style={{ backgroundColor: `hsl(${hsv.h} 100% 50%)` }}
        role="slider"
        tabIndex={0}
        aria-label="Saturation and brightness"
        aria-valuetext={`Saturation ${Math.round(hsv.s * 100)}%, brightness ${Math.round(hsv.v * 100)}%`}
        aria-valuenow={Math.round(hsv.s * 100)}
        onPointerDown={(event) => {
          event.currentTarget.setPointerCapture(event.pointerId);
          pickFromPointer(event);
        }}
        onPointerMove={(event) => {
          if (event.currentTarget.hasPointerCapture(event.pointerId)) pickFromPointer(event);
        }}
        onKeyDown={onAreaKeyDown}
      >
        <span
          className={styles.areaThumb}
          style={{ left: `${hsv.s * 100}%`, top: `${(1 - hsv.v) * 100}%` }}
        />
      </div>
      <input
        type="range"
        className={styles.hue}
        min={0}
        max={359}
        step={1}
        value={Math.round(hsv.h) % 360}
        aria-label="Hue"
        onChange={(event) => apply({ ...hsv, h: Number(event.target.value) })}
      />
      <label className={styles.hexRow}>
        <span className={styles.hexLabel}>#</span>
        <input
          className={styles.hexInput}
          value={hexText}
          maxLength={6}
          spellCheck={false}
          aria-label="Hex color"
          onChange={(event) => onHexChange(event.target.value)}
          onBlur={() => setHexText(hsvToHex(hsv).slice(1))}
        />
      </label>
    </div>,
    container,
  );
}
