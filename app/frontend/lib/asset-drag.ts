import { useRef, type DragEvent, type PointerEvent } from "react";

export function setAssetDragData(transfer: DataTransfer, assetId: string): void {
  transfer.setData("assetId", assetId);
  transfer.effectAllowed = "copy";
}

// Native drag would steal pointer gestures (e.g. timeline scrubbing) from media controls.
const MEDIA_CONTROLS_SELECTOR = "[data-lightbox-controls], audio, input";

export function useAssetDragSource(assetId: string | null) {
  const fromControls = useRef(false);
  return {
    draggable: assetId !== null,
    onPointerDownCapture: (event: PointerEvent<HTMLElement>) => {
      fromControls.current =
        event.target instanceof Element &&
        event.target.closest(MEDIA_CONTROLS_SELECTOR) !== null;
    },
    onDragStart: (event: DragEvent<HTMLElement>) => {
      if (assetId === null || fromControls.current) {
        event.preventDefault();
        return;
      }
      setAssetDragData(event.dataTransfer, assetId);
    },
  };
}
