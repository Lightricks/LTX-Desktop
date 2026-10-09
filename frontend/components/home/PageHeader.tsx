import { clsx } from "clsx";
import type { ReactNode } from "react";

import { HistoryNavButtons } from "./HistoryNavButtons";
import { HomeHeaderActions } from "./HomeHeaderActions";
import styles from "./PageHeader.module.scss";

type PageHeaderProps = {
  queueControl: ReactNode;
  leading?: ReactNode;
  beforeActions?: ReactNode;
  showHistoryNav?: boolean;
  /** When the shell `<main>` already applies `--shell-page-inset`. */
  insetFromShell?: boolean;
};

export function PageHeader({
  queueControl,
  leading,
  beforeActions,
  showHistoryNav = false,
  insetFromShell = false,
}: PageHeaderProps) {
  return (
    <header
      className={clsx(styles.header, insetFromShell && styles.headerInsetFromShell)}
    >
      <div className={styles.leading}>
        {showHistoryNav ? <HistoryNavButtons /> : null}
        {leading}
      </div>
      <div className={styles.trailing}>
        {beforeActions}
        <HomeHeaderActions queueControl={queueControl} />
      </div>
    </header>
  );
}
