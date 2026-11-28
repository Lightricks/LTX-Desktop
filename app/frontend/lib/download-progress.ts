import { formatBytes, formatTimeRemaining } from "./format.ts";

/**
 * The one download-progress shape. Produced by usePythonSetup and the model
 * download queue; consumed by InstallProgress(/Body). Bytes always flow
 * through unmodified — presentation branches on phase, never on doctored
 * numbers.
 */
export interface DownloadProgressSnapshot {
  statusLabel: string;
  percent: number;
  downloadedBytes?: number;
  totalBytes?: number;
  speedBytesPerSec?: number;
}

/** "1.2 GB / 3.1 GB · ETA: 4m" — null when there is nothing to show. */
export function downloadProgressDetail(
  s: DownloadProgressSnapshot,
): string | null {
  const parts: string[] = [];
  const downloaded = s.downloadedBytes ?? 0;
  const total = s.totalBytes ?? 0;
  if (total > 0) {
    parts.push(`${formatBytes(downloaded)} / ${formatBytes(total)}`);
  }
  const remaining = total - downloaded;
  if (
    s.speedBytesPerSec !== undefined &&
    s.speedBytesPerSec > 0 &&
    remaining > 0
  ) {
    parts.push(`ETA: ${formatTimeRemaining(remaining / s.speedBytesPerSec)}`);
  }
  return parts.length > 0 ? parts.join(" · ") : null;
}
