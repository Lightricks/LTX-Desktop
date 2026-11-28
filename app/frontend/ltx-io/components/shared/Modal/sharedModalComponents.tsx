import { RefObject } from "react";

import { Button } from "@ds/Button/Button";

import styles from "./sharedModalComponents.module.scss";

export function ModalTitle({ title }: { title: string }) {
  return <div className={styles.title}>{title}</div>;
}

export function ModalFooter({ children }: { children: React.ReactNode }) {
  return <div className={styles.bottom_bar}>{children}</div>;
}

export function ModalBottomBar({
  onCancelPressed,
  onDonePressed,
  isCancelDisabled,
  isDoneButtonDisabled,
  doneLabel,
  cancelLabel,
  cancelButtonClassName,
  doneButtonClassName,
  variant = "neutral",
  showCancelButton = true,
  doneButtonRef,
  isInProgress = false,
}: {
  onCancelPressed: () => void;
  onDonePressed: () => void;
  isCancelDisabled?: boolean;
  isDoneButtonDisabled?: boolean;
  doneLabel?: string;
  cancelLabel?: string;
  cancelButtonClassName?: string;
  doneButtonClassName?: string;
  variant?: "brand" | "neutral";
  showCancelButton?: boolean;
  doneButtonRef?: RefObject<HTMLButtonElement>;
  isInProgress?: boolean;
}) {
  return (
    <ModalFooter>
      {showCancelButton && (
        <Button
          size="md"
          appearance="neutral"
          hierarchy="plain"
          label={cancelLabel || "Cancel"}
          className={cancelButtonClassName}
          onClick={onCancelPressed}
          disabled={isCancelDisabled}
        />
      )}
      <Button
        size="md"
        appearance={variant}
        hierarchy="primary"
        label={doneLabel || "Done"}
        isLoading={isInProgress}
        className={doneButtonClassName}
        disabled={isDoneButtonDisabled}
        onClick={onDonePressed}
        ref={doneButtonRef}
      />
    </ModalFooter>
  );
}
