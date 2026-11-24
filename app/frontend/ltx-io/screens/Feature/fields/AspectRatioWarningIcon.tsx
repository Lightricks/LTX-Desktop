import { AlertTriangle } from "lucide-react";

import { Tooltip, TooltipProvider } from "@ds/Tooltip/Tooltip";

import styles from "./AspectRatioWarningIcon.module.scss";

/** Gray for a quality caveat, yellow for a budget bet. */
export function AspectRatioWarningIcon({
  warning,
  tone = "caveat",
}: {
  warning: string;
  tone?: "caveat" | "risk";
}) {
  return (
    <TooltipProvider delay={70}>
      <Tooltip content={warning} side="top" maxWidth={280}>
        <span
          className={tone === "risk" ? styles.riskIcon : styles.warningIcon}
          aria-label={warning}
          onClick={(event) => event.stopPropagation()}
        >
          <AlertTriangle aria-hidden />
        </span>
      </Tooltip>
    </TooltipProvider>
  );
}
