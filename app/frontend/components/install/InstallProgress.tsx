import type { ReactNode } from "react";

import { Text } from "@ds/Text/Text";
import { EndpageLockup } from "@/components/startup-loader/EndpageLockup";
import lockupStyles from "@/components/startup-loader/EndpageLockup.module.scss";

import {
  downloadProgressDetail,
  type DownloadProgressSnapshot,
} from "@/lib/download-progress";

import styles from "./InstallProgress.module.scss";

export type InstallProgressBodyProps = {
  snapshot: DownloadProgressSnapshot;
  /** Replaces the track + meta when present. */
  error?: ReactNode;
};

/**
 * The one centered-lockup progress body: narrow track, status · percent,
 * detail line. The splash download arm renders this directly (its Lottie
 * lockup is already on screen); full-screen installs use InstallProgress.
 */
export function InstallProgressBody({
  snapshot,
  error,
}: InstallProgressBodyProps) {
  if (error) {
    return <div className={styles.error} role="alert">{error}</div>;
  }
  const clampedPercent = Math.max(0, Math.min(100, Math.round(snapshot.percent)));
  const detail = downloadProgressDetail(snapshot);
  return (
    <div
      role="progressbar"
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={clampedPercent}
      aria-label={snapshot.statusLabel}
    >
      <div className={styles.track}>
        <div className={styles.fill} style={{ width: `${clampedPercent}%` }} />
      </div>
      <div className={styles.meta}>
        <Text as="p" variant="body" size="md" align="center" className={styles.status}>
          {snapshot.statusLabel}
          <span className={styles.percent}>{clampedPercent}%</span>
        </Text>
        <Text as="p" variant="body" size="md" align="center" className={styles.detail}>
          {detail ?? " "}
        </Text>
      </div>
    </div>
  );
}

export function InstallProgress(props: InstallProgressBodyProps) {
  return (
    <div className={styles.root}>
      <div className={lockupStyles.shell}>
        {/* The settled frame of the splash animation, so the lockup never moves
            between the two screens. */}
        <EndpageLockup still />
        <div className={styles.bodyUnderLockup}>
          <InstallProgressBody {...props} />
        </div>
      </div>
    </div>
  );
}
