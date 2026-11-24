import {
  useEffect,
  useRef,
  useState,
  type FocusEvent,
  type PointerEvent,
  type RefObject,
} from "react";
import {
  HOME_SIDEBAR_COLLAPSE_DELAY_MS,
  applyHomeSidebarPinChange,
  createHomeSidebarCollapseDelay,
  isHomeSidebarOpen,
} from "./home-sidebar-expansion";

function isKeyboardFocus(target: EventTarget | null): boolean {
  return target instanceof Element && target.matches(":focus-visible");
}

function nodeIsInside(
  node: EventTarget | null,
  root: Node | null | undefined,
): boolean {
  return node instanceof Node && root != null && root.contains(node);
}

export type HomeSidebarOpenApi = {
  pinned: boolean;
  open: boolean;
  /** Attached to the aside; used to detect moves between hotspot and panel. */
  panelRef: RefObject<HTMLElement>;
  /** Attached to the pin toggle; used to preserve keyboard focus on unpin. */
  toggleRef: RefObject<HTMLButtonElement>;
  setPinned: (nextPinned: boolean) => void;
  setMenuOpen: (open: boolean) => void;
  /**
   * Spread onto BOTH the peek hotspot and the aside. Moving between them
   * never collapses (relatedTarget stays inside the group); leaving both —
   * including into portaled menus or native chrome — schedules collapse after
   * the delay. menuOpen/focus keep `open` true meanwhile, which is what makes
   * the old rect-geometry check unnecessary.
   */
  hoverProps: {
    onPointerEnter: () => void;
    onPointerLeave: (event: PointerEvent<HTMLElement>) => void;
  };
  focusProps: {
    onFocus: (event: FocusEvent<HTMLElement>) => void;
    onBlur: (event: FocusEvent<HTMLElement>) => void;
  };
};

/**
 * Single owner of the rail's transient-open state machine. The component never
 * sees hovered/focused/menuOpen individually — only pinned/open plus props.
 */
export function useHomeSidebarOpen(input: {
  defaultPinned: boolean;
}): HomeSidebarOpenApi {
  const [pinned, setPinnedState] = useState(input.defaultPinned);
  const [hovered, setHovered] = useState(false);
  const [focused, setFocused] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  const panelRef = useRef<HTMLElement>(null);
  const toggleRef = useRef<HTMLButtonElement>(null);
  const collapseDelayRef = useRef(
    createHomeSidebarCollapseDelay({
      delayMs: HOME_SIDEBAR_COLLAPSE_DELAY_MS,
      setTimeoutFn: (fn, ms) => window.setTimeout(fn, ms),
      clearTimeoutFn: (id) => window.clearTimeout(id),
    }),
  );

  useEffect(() => {
    const delay = collapseDelayRef.current;
    return () => delay.cancel();
  }, []);

  const cancelCollapse = () => collapseDelayRef.current.cancel();

  const handlePointerEnter = () => {
    cancelCollapse();
    setHovered(true);
  };

  const handlePointerLeave = (event: PointerEvent<HTMLElement>) => {
    const next = event.relatedTarget;
    if (nodeIsInside(next, event.currentTarget)) return;
    if (nodeIsInside(next, panelRef.current)) return;
    collapseDelayRef.current.schedule(() => {
      setHovered(false);
    });
  };

  const handleFocus = (event: FocusEvent<HTMLElement>) => {
    if (event.target === toggleRef.current) {
      cancelCollapse();
    }
    if (!isKeyboardFocus(event.target)) {
      setFocused(false);
      return;
    }
    cancelCollapse();
    setFocused(true);
  };

  const handleBlur = (event: FocusEvent<HTMLElement>) => {
    const next = event.relatedTarget;
    if (nodeIsInside(next, panelRef.current)) return;
    collapseDelayRef.current.schedule(() => {
      setFocused(false);
    });
  };

  const setPinned = (nextPinned: boolean) => {
    collapseDelayRef.current.cancel();
    const next = applyHomeSidebarPinChange(
      { pinned, hovered, focused, menuOpen },
      nextPinned,
    );
    setPinnedState(next.pinned);
    setHovered(next.hovered);
    setFocused(
      next.pinned ? next.focused : isKeyboardFocus(toggleRef.current),
    );
  };

  return {
    pinned,
    open: isHomeSidebarOpen({ pinned, hovered, focused, menuOpen }),
    panelRef,
    toggleRef,
    setPinned,
    setMenuOpen,
    hoverProps: {
      onPointerEnter: handlePointerEnter,
      onPointerLeave: handlePointerLeave,
    },
    focusProps: {
      onFocus: handleFocus,
      onBlur: handleBlur,
    },
  };
}
