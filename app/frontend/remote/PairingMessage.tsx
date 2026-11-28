import type { ReactNode } from "react";
import styles from "./RemoteShell.module.scss";

export type PairingHealth = "checking" | "ok" | "missing" | "failed";

export function PairingMessage({
  title,
  body,
  children,
}: {
  title: string;
  body: string;
  children?: ReactNode;
}) {
  return (
    <main className={styles.home}>
      <h1 className={styles.homeTitle}>{title}</h1>
      <p className={styles.homeCopy}>{body}</p>
      {children}
    </main>
  );
}

export function PairingStatus({
  health,
}: {
  health: Exclude<PairingHealth, "ok">;
}) {
  if (health === "checking") {
    return (
      <PairingMessage title="Connecting…" body="Connecting to Desktop…" />
    );
  }
  if (health === "missing") {
    return (
      <PairingMessage
        title="Missing pairing token"
        body="Open this page from the QR code or link shown in LTX Desktop."
      />
    );
  }
  return (
    <PairingMessage
      title="Could not pair"
      body="Could not reach Desktop, or this link is no longer valid. Keep LTX Desktop open, stay on this Wi-Fi, and open the current QR code or link."
    />
  );
}
