import { ShieldCheck, X } from "lucide-react";

import { useAppSettings } from "../../contexts/AppSettingsContext";

import styles from "./CautiousModePill.module.scss";

/** Header chip while Cautious mode hides risky options, plus the last failed toggle. */
export function CautiousModePill() {
  const {
    settings,
    setBudgetMode,
    budgetModeSaving,
    settingsSyncError,
    clearSettingsSyncError,
  } = useAppSettings();
  const showPill = settings.budgetMode === "cautious";
  if (!showPill && settingsSyncError == null) return null;
  return (
    <span className={styles.pill}>
      {showPill ? <ShieldCheck className={styles.icon} aria-hidden /> : null}
      {showPill ? (
        <span role="status">Cautious mode is on. Jobs that need too much memory or are untested are blocked.</span>
      ) : null}
      {settingsSyncError ? <span>{settingsSyncError}</span> : null}
      {showPill ? (
        <button
          type="button"
          className={styles.action}
          disabled={budgetModeSaving}
          onClick={() => void setBudgetMode("brave")}
        >
          Turn off
        </button>
      ) : null}
      {settingsSyncError ? (
        <button
          type="button"
          className={styles.action}
          aria-label="Dismiss error"
          onClick={clearSettingsSyncError}
        >
          <X className={styles.icon} aria-hidden />
        </button>
      ) : null}
    </span>
  );
}
