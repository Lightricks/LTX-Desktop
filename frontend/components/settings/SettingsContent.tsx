import { clsx } from "clsx";
import { Check, type LucideIcon } from "lucide-react";
import type { ReactNode, Ref, RefObject } from "react";

import { Badge } from "@/ds/Badge/Badge";
import { Button as DsButton } from "@/ds/Button/Button";
import { Text } from "@/ds/Text/Text";

import styles from "./SettingsContent.module.scss";

export function SettingsStack({
  children,
  loose = false,
  className,
}: {
  children: ReactNode;
  loose?: boolean;
  className?: string;
}) {
  return (
    <div className={clsx(loose ? styles.stackLoose : styles.stack, className)}>
      {children}
    </div>
  );
}

export function SettingsSubsection({
  title,
  description,
  icon: Icon,
  badge,
  headerTrailing,
  showDivider = false,
  sectionRef,
  className,
  descriptionClassName,
  descriptionSize = "xs",
  bare = false,
  children,
}: {
  title: string;
  description?: ReactNode;
  icon?: LucideIcon;
  badge?: ReactNode;
  headerTrailing?: ReactNode;
  showDivider?: boolean;
  sectionRef?: RefObject<HTMLElement | null>;
  className?: string;
  descriptionClassName?: string;
  descriptionSize?: "xs" | "sm";
  /** Skip default subsection spacing when using API Keys layout classes. */
  bare?: boolean;
  children?: ReactNode;
}) {
  return (
    <section
      ref={sectionRef as Ref<HTMLElement> | undefined}
      className={clsx(
        !bare && (showDivider ? styles.subsectionWithDivider : styles.subsection),
        description && styles.subsectionFlushDescription,
        className,
      )}
    >
      <div className={styles.subsectionHeaderRow}>
        <div className={styles.subsectionHeading}>
          {Icon ? <Icon className={styles.subsectionIcon} aria-hidden /> : null}
          <Text as="h3" variant="heading" size="xs">
            {title}
          </Text>
          {badge}
        </div>
        {headerTrailing ? (
          <div className={styles.subsectionHeaderTrailing}>{headerTrailing}</div>
        ) : null}
      </div>
      {description ? (
        <Text
          as="p"
          variant="body"
          size={descriptionSize}
          className={clsx(styles.textTertiary, descriptionClassName)}
        >
          {description}
        </Text>
      ) : null}
      {children}
    </section>
  );
}

export function SettingsPanel({
  children,
  inset = false,
  className,
}: {
  children: ReactNode;
  inset?: boolean;
  className?: string;
}) {
  return (
    <div className={clsx(inset ? styles.panelInset : styles.panel, className)}>
      {children}
    </div>
  );
}

export function SettingsReadonlyField({
  value,
  emptyLabel = "Not set",
}: {
  value: string;
  emptyLabel?: string;
}) {
  return (
    <div className={styles.readonlyField}>
      {value ? (
        value
      ) : (
        <Text as="span" variant="body" size="sm" className={styles.textTertiary}>
          {emptyLabel}
        </Text>
      )}
    </div>
  );
}

export function SettingsFieldRow({
  children,
  className,
}: {
  children: ReactNode;
  className?: string;
}) {
  return <div className={clsx(styles.fieldRow, className)}>{children}</div>;
}

export function SettingsInputRow({
  children,
  className,
}: {
  children: ReactNode;
  className?: string;
}) {
  return <div className={clsx(styles.inputRow, className)}>{children}</div>;
}

export function SettingsSelect({
  className,
  ...props
}: React.SelectHTMLAttributes<HTMLSelectElement>) {
  return <select {...props} className={clsx(styles.select, className)} />;
}

export function SettingsNumberInput({
  className,
  ...props
}: React.InputHTMLAttributes<HTMLInputElement>) {
  return <input {...props} className={clsx(styles.numberInput, className)} />;
}

export function SettingsRadioMark({ selected }: { selected: boolean }) {
  return (
    <input
      type="radio"
      readOnly
      checked={selected}
      tabIndex={-1}
      className={styles.settingsRadio}
      aria-hidden
    />
  );
}

export function SettingsOptionCard({
  selected,
  disabled = false,
  onClick,
  children,
  footer,
  className,
  innerClassName,
  footerClassName,
  activateOnPointerDown = false,
  applySelectedClass = true,
}: {
  selected: boolean;
  disabled?: boolean;
  onClick?: () => void;
  children: ReactNode;
  footer?: ReactNode;
  className?: string;
  innerClassName?: string;
  footerClassName?: string;
  /** Fire on pointer down instead of click for instant selection (e.g. text encoding cards). */
  activateOnPointerDown?: boolean;
  /** When false, `selected` is ignored for the default blue option-card border class. */
  applySelectedClass?: boolean;
}) {
  const interactive = !disabled && onClick;

  const activate = () => {
    onClick?.();
  };

  return (
    <div
      role={interactive ? "button" : undefined}
      tabIndex={interactive ? 0 : undefined}
      onClick={
        interactive && !activateOnPointerDown ? activate : undefined
      }
      onPointerDown={
        interactive && activateOnPointerDown
          ? (event) => {
              if (event.button !== 0) return;
              event.preventDefault();
              activate();
            }
          : undefined
      }
      onKeyDown={
        interactive
          ? (event) => {
              if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                activate();
              }
            }
          : undefined
      }
      className={clsx(
        styles.optionCard,
        selected && applySelectedClass && styles.optionCardSelected,
        disabled && styles.optionCardDisabled,
        className,
      )}
    >
      <div className={clsx(styles.optionCardInner, innerClassName)}>{children}</div>
      {footer ? (
        <div className={clsx(styles.optionCardFooter, footerClassName)}>{footer}</div>
      ) : null}
    </div>
  );
}

export function SettingsOptionTitle({
  title,
  icon: Icon,
  badge,
  description,
}: {
  title: string;
  icon?: LucideIcon;
  badge?: ReactNode;
  description?: ReactNode;
}) {
  return (
    <div className="min-w-0 flex-1">
      <div className={styles.optionTitleRow}>
        {Icon ? <Icon className={styles.subsectionIcon} aria-hidden /> : null}
        <Text as="span" variant="label" size="sm">
          {title}
        </Text>
        {badge}
      </div>
      {description ? (
        <Text as="p" variant="body" size="xs" className={styles.textSecondary}>
          {description}
        </Text>
      ) : null}
    </div>
  );
}

export function SettingsSwitch({
  enabled,
  onToggle,
  ariaLabel,
  ariaLabelledBy,
  ariaDescribedBy,
  id,
  disabled = false,
}: {
  enabled: boolean;
  onToggle: () => void;
  ariaLabel?: string;
  ariaLabelledBy?: string;
  ariaDescribedBy?: string;
  id?: string;
  disabled?: boolean;
}) {
  return (
    <button
      id={id}
      disabled={disabled}
      type="button"
      role="switch"
      aria-checked={enabled}
      aria-label={ariaLabel}
      aria-labelledby={ariaLabelledBy}
      aria-describedby={ariaDescribedBy}
      onClick={onToggle}
      className={clsx(styles.switch, enabled && styles.switchOn)}
    >
      <span
        className={clsx(styles.switchThumb, enabled && styles.switchThumbOn)}
      />
    </button>
  );
}

export function SettingsHint({
  children,
  variant = "default",
}: {
  children: ReactNode;
  variant?: "default" | "warning";
}) {
  return (
    <Text
      as="p"
      variant="body"
      size="xs"
      className={variant === "warning" ? styles.hintWarning : styles.hint}
    >
      {children}
    </Text>
  );
}

export function SettingsCallout({ children }: { children: ReactNode }) {
  return <div className={styles.callout}>{children}</div>;
}

export function SettingsLinkButton({
  children,
  onClick,
  type = "button",
}: {
  children: ReactNode;
  onClick?: () => void;
  type?: "button" | "submit";
}) {
  return (
    <button type={type} className={styles.linkButton} onClick={onClick}>
      {children}
    </button>
  );
}

export function SettingsKeyStatus({
  configured,
  configuredLabel = "Key configured",
  missingLabel = "Not configured",
  required = false,
  labelCaps = false,
}: {
  configured: boolean;
  configuredLabel?: string;
  missingLabel?: string;
  required?: boolean;
  /** Uppercase label styling (API Keys header badges). */
  labelCaps?: boolean;
}) {
  const textVariant = labelCaps ? "labelCaps" : "label";

  if (configured) {
    return (
      <Badge
        appearance="success"
        size="sm"
        text={configuredLabel}
        textVariant={textVariant}
        className={styles.apiKeysStatusBadge}
        leftIcon={
          <Check
            aria-hidden
            strokeWidth={2.75}
            className={styles.apiKeysStatusCheck}
          />
        }
      />
    );
  }

  return (
    <Badge
      appearance={required ? "warning" : "neutral"}
      size="sm"
      text={required ? "Required" : missingLabel}
      textVariant={textVariant}
      className={styles.apiKeysStatusBadge}
    />
  );
}

export function SettingsSaveButton({
  label,
  disabled,
  onClick,
}: {
  label: string;
  disabled?: boolean;
  onClick: () => void;
}) {
  return (
    <DsButton
      appearance="brand"
      hierarchy="primary"
      size="md"
      label={label}
      disabled={disabled}
      onClick={onClick}
    />
  );
}

export function SettingsNestedGroup({ children }: { children: ReactNode }) {
  return <div className={styles.nestedGroup}>{children}</div>;
}

export function SettingsProgressBar({ percent }: { percent: number }) {
  return (
    <div className={styles.progressTrack}>
      <div className={styles.progressFill} style={{ width: `${percent}%` }} />
    </div>
  );
}
