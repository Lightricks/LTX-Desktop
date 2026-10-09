import { clsx } from "clsx";

import { Button } from "@ds/Button/Button";
import AcceptIcon from "@ds/assets/Icons/Accept/Fill.svg?react";
import Warning from "@ds/assets/Icons/Attention/Fill.svg?react";
import CancelIcon from "@ds/assets/Icons/Cancel/Fill.svg?react";
import CloseIcon from "@ds/assets/Icons/Close/Normal.svg?react";
import InfoIcon from "@ds/assets/Icons/Info/Fill.svg?react";

import styles from "./Toast.module.scss";

type ToastType = "neutral" | "danger" | "warning" | "success" | "brand";

export type ToastActionProps = {
  label: string;
  onClick: (() => void) | (() => Promise<unknown>);
};

export type ToastOwnProps = {
  message: string;
  toastType: ToastType;
  action?: ToastActionProps;
  closeable?: boolean;
  className?: string;
  onBodyClick?: () => void;
  hideIcon?: boolean;
};

export type ToastProps = ToastOwnProps & {
  closeToast?: () => void;
};

export function ToastIcon({ toastType }: { toastType: ToastType }) {
  switch (toastType) {
    case "neutral":
      return <InfoIcon className={clsx(styles.icon, styles.neutral)} />;
    case "warning":
      return <Warning className={clsx(styles.icon, styles.warning)} />;
    case "danger":
      return <CancelIcon className={clsx(styles.icon, styles.danger)} />;
    case "success":
      return <AcceptIcon className={clsx(styles.icon, styles.success)} />;
    case "brand":
      return <InfoIcon className={clsx(styles.icon, styles.brand)} />;
  }
}

export function ToastAction({
  label,
  onClick,
  closeToast,
}: ToastActionProps & { closeToast?: () => void }) {
  return (
    <Button
      appearance="neutral"
      hierarchy="plain"
      size="md"
      label={label}
      className={styles.action}
      onClick={(event) => {
        event.stopPropagation();
        void onClick();
        closeToast?.();
      }}
    />
  );
}

export function Toast(props: ToastProps) {
  const { message, toastType, className, closeable, action, closeToast, onBodyClick, hideIcon } =
    props;

  return (
    <div
      className={clsx(styles.toast, className, "dark-only-theme-style")}
      onClick={onBodyClick}
    >
      <div className={styles.topRow}>
        {hideIcon ? null : <ToastIcon toastType={toastType} />}
        <div className={clsx(styles.message, !!action && styles.with_action)}>
          {message}
        </div>
        {!!action && (
          <ToastAction
            label={action.label}
            onClick={action.onClick}
            closeToast={closeToast}
          />
        )}
        {!!closeable && (
          <Button
            appearance="neutral"
            hierarchy="plain"
            size="md"
            isIconOnly
            leftIcon={<CloseIcon />}
            aria-label="Close"
            className={styles.close}
            onClick={(event) => {
              event.stopPropagation();
              closeToast?.();
            }}
          />
        )}
      </div>
    </div>
  );
}
