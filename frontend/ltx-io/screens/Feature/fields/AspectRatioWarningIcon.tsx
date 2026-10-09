import { AlertTriangle } from "lucide-react";

import { Tooltip, TooltipProvider } from "@ds/Tooltip/Tooltip";

import styles from "./AspectRatioWarningIcon.module.scss";

/** Soft caveat for ratios outside the distilled training pair. */
export function AspectRatioWarningIcon({ warning }: { warning: string }) {
  return (
    <TooltipProvider delay={70}>
      <Tooltip content={warning} side="top" maxWidth={280}>
        <span
          className={styles.warningIcon}
          aria-label={warning}
          onClick={(event) => event.stopPropagation()}
        >
          <AlertTriangle aria-hidden />
        </span>
      </Tooltip>
    </TooltipProvider>
  );
}
