import { useEffect, useState } from "react";
import { createPortal } from "react-dom";

import { useThemedPortalContainer } from "@ds/styles/themes/useTheme";

import { Toast } from "./Toast";
import toastHostStyles from "./toastHost.module.scss";
import {
  type ActiveToast,
  dismissToast,
  subscribeToasts,
} from "./toastService";

const SLIDE_OUT_MS = 70;

export function LtxioToastHost() {
  const [toast, setToast] = useState<ActiveToast | null>(null);
  const [leaving, setLeaving] = useState(false);
  const portalTarget = useThemedPortalContainer();

  useEffect(() => {
    let leaveTimer: number | null = null;
    const unsubscribe = subscribeToasts((next) => {
      if (leaveTimer != null) {
        window.clearTimeout(leaveTimer);
        leaveTimer = null;
      }
      if (next == null) {
        setLeaving(true);
        leaveTimer = window.setTimeout(() => {
          setToast(null);
          setLeaving(false);
          leaveTimer = null;
        }, SLIDE_OUT_MS);
        return;
      }
      setLeaving(false);
      setToast(next);
    });
    return () => {
      unsubscribe();
      if (leaveTimer != null) window.clearTimeout(leaveTimer);
    };
  }, []);

  // The live region has to exist before the message appears. Screen readers
  // often skip aria-live on an element created in the same commit as the text.
  if (portalTarget == null) return null;

  return createPortal(
    <div
      className={toastHostStyles.toast_container}
      role="status"
      aria-live="polite"
    >
      {toast != null ? (
        <div className={leaving ? toastHostStyles.slideOut : toastHostStyles.slideIn}>
          <Toast
            message={toast.message}
            toastType={toast.toastType}
            action={toast.action}
            closeable={toast.closeable}
            className={toast.className}
            onBodyClick={toast.onBodyClick}
            hideIcon={toast.hideIcon}
            closeToast={dismissToast}
          />
        </div>
      ) : null}
    </div>,
    portalTarget,
  );
}
