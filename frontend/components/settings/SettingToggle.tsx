import type { LucideIcon } from "lucide-react";
import { useId, type ReactNode } from "react";

import { Text } from "@/ds/Text/Text";

import {
  SettingsHint,
  SettingsSwitch,
} from "./SettingsContent";
import styles from "./SettingsContent.module.scss";

/** A labelled on/off setting row: optional icon, title, description, and switch. */
export function SettingToggle({
  title,
  description,
  enabled,
  onToggle,
  hideTopBorder = false,
  leadingIcon: LeadingIcon,
  nested = false,
  rowLayout = "default",
  disabled = false,
}: {
  title: string;
  description?: ReactNode;
  enabled: boolean;
  onToggle: () => void;
  hideTopBorder?: boolean;
  leadingIcon?: LucideIcon;
  nested?: boolean;
  /** Compact bordered row — title, description, switch (General tab style). */
  rowLayout?: "default" | "preference";
  disabled?: boolean;
}) {
  const titleId = useId();
  const descriptionId = useId();
  const switchId = useId();

  if (rowLayout === "preference") {
    return (
      <div className={styles.preferenceRow}>
        <div className={styles.preferenceRowCopy}>
          <label htmlFor={switchId} id={titleId} className={styles.preferenceRowTitle}>
            {title}
          </label>
          {description ? (
            <p id={descriptionId} className={styles.preferenceRowHint}>
              {description}
            </p>
          ) : null}
        </div>
        <SettingsSwitch
          id={switchId}
          enabled={enabled}
          onToggle={onToggle}
          disabled={disabled}
          ariaLabelledBy={titleId}
          ariaDescribedBy={description ? descriptionId : undefined}
        />
      </div>
    );
  }

  return (
    <div
      className={
        hideTopBorder
          ? nested
            ? styles.stack
            : styles.subsection
          : styles.subsectionWithDivider
      }
    >
      <div className="flex items-start justify-between gap-4">
        <div className="min-w-0 flex-1">
          <div className={styles.subsectionHeading}>
            {LeadingIcon ? (
              <LeadingIcon className={styles.subsectionIcon} aria-hidden />
            ) : null}
            <label htmlFor={switchId} id={titleId} className="cursor-pointer">
              <Text variant="label" size={nested ? "xs" : "sm"}>
                {title}
              </Text>
            </label>
          </div>
          {description ? (
            <SettingsHint>
              <span id={descriptionId}>{description}</span>
            </SettingsHint>
          ) : null}
        </div>

        <SettingsSwitch
          id={switchId}
          enabled={enabled}
          onToggle={onToggle}
          disabled={disabled}
          ariaLabelledBy={titleId}
          ariaDescribedBy={description ? descriptionId : undefined}
        />
      </div>
    </div>
  );
}
