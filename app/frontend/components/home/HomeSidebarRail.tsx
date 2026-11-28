import { clsx } from "clsx";
import type { MouseEvent, PointerEvent } from "react";
import { createPortal } from "react-dom";
import { Button } from "@/ds/Button/Button";
import ExpandSidebarIcon from "@/ds/assets/Icons/PinSidebar.svg?react";
import CollapseSidebarIcon from "@/ds/assets/Icons/UnpinSidebar.svg?react";
import {
  HOME_SIDEBAR_EXPANDED_WIDTH_PX,
  homeSidebarContentOffsetPx,
  homeSidebarPinLabel,
} from "./home-sidebar-expansion";
import type {
  HomeSidebarActions,
  HomeSidebarItemPlan,
} from "./home-sidebar-items";
import { HomeLogoButton } from "./HomeLogoButton";
import { HomeSidebarMenu } from "./HomeSidebarMenu";
import { useHomeSidebarOpen } from "./useHomeSidebarOpen";
import styles from "./DesktopHomeSidebar.module.scss";

export function HomeSidebarRail({
  plan,
  actions,
  portalContainer,
}: {
  plan: HomeSidebarItemPlan;
  actions: HomeSidebarActions;
  portalContainer?: HTMLElement | null;
}) {
  const sidebar = useHomeSidebarOpen({ defaultPinned: plan.defaultPinned });
  const { pinned, open } = sidebar;
  const offsetPx = homeSidebarContentOffsetPx(pinned);
  const showCollapsedChrome = !pinned && !open;

  const handleTogglePointerDown = (event: PointerEvent<HTMLButtonElement>) => {
    if (event.button !== 0) return;
    event.stopPropagation();
  };

  const handleToggleClick = (event: MouseEvent<HTMLButtonElement>) => {
    event.stopPropagation();
    // The toggle lives in two hosts (portaled chrome when closed, brand when
    // open — genuinely different positions), so activating it remounts the
    // focused button. Restore focus so keyboard users don't restart from body.
    const hadFocus = document.activeElement === sidebar.toggleRef.current;
    sidebar.setPinned(!pinned);
    if (hadFocus) {
      requestAnimationFrame(() => sidebar.toggleRef.current?.focus());
    }
  };

  const toggleButton = (
    <Button
      ref={sidebar.toggleRef}
      isIconOnly
      hierarchy="plain"
      leftIcon={pinned ? <CollapseSidebarIcon /> : <ExpandSidebarIcon />}
      aria-label={homeSidebarPinLabel(pinned)}
      aria-pressed={pinned}
      onPointerDown={handleTogglePointerDown}
      onClick={handleToggleClick}
      className={clsx(styles.pinButton, "window-no-drag")}
    />
  );

  const collapsedChrome = (
    <div className={styles.collapsedChrome}>
      <div
        className={styles.peekHotspot}
        aria-hidden="true"
        onPointerEnter={sidebar.hoverProps.onPointerEnter}
        onPointerLeave={sidebar.hoverProps.onPointerLeave}
      />
      <div className={clsx(styles.chromeToggle, "window-no-drag")}>
        {toggleButton}
      </div>
    </div>
  );

  return (
    <>
      <div
        className={styles.offset}
        style={{
          width: offsetPx,
          flexBasis: offsetPx,
        }}
        aria-hidden="true"
      />
      {showCollapsedChrome && portalContainer
        ? createPortal(collapsedChrome, portalContainer)
        : showCollapsedChrome
          ? collapsedChrome
          : null}
      <aside
        ref={sidebar.panelRef}
        id="home-sidebar"
        className={clsx(
          styles.sidebar,
          open && styles.sidebarOpen,
          open && !pinned && styles.sidebarOverlay,
        )}
        style={{ width: HOME_SIDEBAR_EXPANDED_WIDTH_PX }}
        data-pinned={pinned ? "true" : "false"}
        aria-hidden={open ? undefined : true}
        onPointerEnter={sidebar.hoverProps.onPointerEnter}
        onPointerLeave={sidebar.hoverProps.onPointerLeave}
        onFocus={sidebar.focusProps.onFocus}
        onBlur={sidebar.focusProps.onBlur}
      >
        <div className={clsx(styles.brand, "window-no-drag")}>
          <HomeLogoButton
            actions={actions}
            buttonClassName={clsx(styles.logoButton, "window-no-drag")}
            logoClassName={styles.logo}
          />
          {showCollapsedChrome ? null : toggleButton}
        </div>
        <nav aria-label="Home" className={styles.menu}>
          <HomeSidebarMenu plan={plan} actions={actions} />
        </nav>
      </aside>
    </>
  );
}
