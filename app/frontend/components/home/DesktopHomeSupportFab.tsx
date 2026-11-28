import { DesktopHomeSupportMenu } from "./DesktopHomeSupportMenu";
import styles from "./DesktopHomeSupportFab.module.scss";

export function DesktopHomeSupportFab({
  portalContainer,
  besidePanel = false,
}: {
  portalContainer?: HTMLElement | null;
  /** The docked Assistant panel covers the corner, so sit left of it. */
  besidePanel?: boolean;
}) {
  return (
    <div className={styles.root} data-beside-panel={besidePanel ? "" : undefined}>
      <DesktopHomeSupportMenu portalContainer={portalContainer} variant="fab" />
    </div>
  );
}
