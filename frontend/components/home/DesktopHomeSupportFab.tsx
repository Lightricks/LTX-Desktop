import { DesktopHomeSupportMenu } from "./DesktopHomeSupportMenu";
import styles from "./DesktopHomeSupportFab.module.scss";

export function DesktopHomeSupportFab({
  portalContainer,
}: {
  portalContainer?: HTMLElement | null;
}) {
  return (
    <div className={styles.root}>
      <DesktopHomeSupportMenu portalContainer={portalContainer} variant="fab" />
    </div>
  );
}
