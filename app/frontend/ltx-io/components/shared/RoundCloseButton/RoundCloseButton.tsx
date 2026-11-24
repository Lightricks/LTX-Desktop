import { clsx } from "clsx";

import CloseIcon from "@ds/assets/Icons/Close/Normal.svg?react";

import styles from "./RoundCloseButton.module.scss";

export function RoundCloseButton({
  onClick,
  isDisabled,
  className,
}: {
  onClick?: () => void;
  isDisabled?: boolean;
  className?: string;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={clsx(styles.closeButton, className)}
      disabled={isDisabled}
      aria-label="Close"
    >
      <CloseIcon />
    </button>
  );
}
