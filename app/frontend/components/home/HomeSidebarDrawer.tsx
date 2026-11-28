import { clsx } from "clsx";
import { useEffect, useRef, useState } from "react";
import { Menu } from "lucide-react";
import { useLocation } from "react-router";
import { Button } from "@/ds/Button/Button";
import CloseIcon from "@/ds/assets/Icons/Close/Normal.svg?react";
import type {
  HomeSidebarActions,
  HomeSidebarItemPlan,
} from "./home-sidebar-items";
import { HomeSidebarMenu } from "./HomeSidebarMenu";
import { HomeLogoButton } from "./HomeLogoButton";
import styles from "./DesktopHomeSidebar.module.scss";

export function HomeSidebarDrawer({
  plan,
  actions,
}: {
  plan: HomeSidebarItemPlan;
  actions: HomeSidebarActions;
}) {
  const [drawerOpen, setDrawerOpen] = useState(false);
  const menuButtonRef = useRef<HTMLButtonElement>(null);
  const closeButtonRef = useRef<HTMLButtonElement>(null);
  const hadDrawerOpenRef = useRef(false);
  const location = useLocation();

  useEffect(() => {
    setDrawerOpen(false);
  }, [location.pathname]);

  useEffect(() => {
    if (drawerOpen) {
      hadDrawerOpenRef.current = true;
      closeButtonRef.current?.focus();
      return;
    }
    if (hadDrawerOpenRef.current) {
      hadDrawerOpenRef.current = false;
      menuButtonRef.current?.focus();
    }
  }, [drawerOpen]);

  useEffect(() => {
    if (!drawerOpen) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      event.preventDefault();
      setDrawerOpen(false);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [drawerOpen]);

  return (
    <>
      <header className={styles.mobileBar}>
        <Button
          ref={menuButtonRef}
          isIconOnly
          hierarchy="plain"
          leftIcon={<Menu className={styles.barIcon} />}
          aria-label="Open menu"
          aria-expanded={drawerOpen}
          aria-controls="home-sidebar"
          onClick={() => setDrawerOpen(true)}
          className={styles.barButton}
        />
        <HomeLogoButton
          actions={actions}
          buttonClassName={styles.logoButton}
          logoClassName={styles.barLogo}
        />
      </header>
      {drawerOpen ? (
        <button
          type="button"
          className={styles.backdrop}
          aria-label="Close menu"
          onClick={() => setDrawerOpen(false)}
        />
      ) : null}
      <aside
        id="home-sidebar"
        className={clsx(
          styles.sidebar,
          styles.drawer,
          drawerOpen && styles.drawerOpen,
        )}
        aria-hidden={drawerOpen ? undefined : true}
        aria-modal={drawerOpen ? true : undefined}
      >
        <div className={styles.brand}>
          <HomeLogoButton
            actions={actions}
            buttonClassName={styles.logoButton}
            logoClassName={styles.logo}
          />
          <Button
            ref={closeButtonRef}
            isIconOnly
            hierarchy="plain"
            leftIcon={<CloseIcon />}
            aria-label="Close menu"
            onClick={() => setDrawerOpen(false)}
            className={styles.pinButton}
          />
        </div>
        <nav aria-label="Home" className={styles.menu}>
          <HomeSidebarMenu
            plan={plan}
            actions={actions}
            onNavigateComplete={() => setDrawerOpen(false)}
          />
        </nav>
      </aside>
    </>
  );
}
