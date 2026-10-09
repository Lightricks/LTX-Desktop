import type { ReactNode } from "react";

export type DesktopHomeOutletContext = {
  pageHeaderQueueControl: ReactNode | null;
  /** Set by a shell that mounts back/forward. Absent context leaves the arrows off. */
  showHistoryNav?: boolean;
};

/** Remote and other hosts render these screens with no outlet context. */
export function pageHeaderQueueControlFromOutlet(
  context: DesktopHomeOutletContext | null | undefined,
): ReactNode | null {
  return context?.pageHeaderQueueControl ?? null;
}

export function showHistoryNavFromOutlet(
  context: DesktopHomeOutletContext | null | undefined,
): boolean {
  return context?.showHistoryNav === true;
}
