import { clsx } from "clsx";
import { motion } from "framer-motion";
import React, { useEffect, useRef } from "react";
import ReactDOM from "react-dom";

import { useModalContext } from "@/ltx-io/components/shared/Modal/ModalContext.tsx";
import { CloseReason } from "@/ltx-io/components/shared/Modal/modalTypes";
import {
  backdropAnimationProps,
  modalAnimationProps,
} from "@ds/lib/popoverAnimationOptions";
import { useKeyboardShortcut } from "@ds/lib/useKeyboardShortcut";

import styles from "./Modal.module.scss";
import { ModalVariant } from "./modalTypes";

export type ModalProps = {
  onClose: (reason: CloseReason) => void;
  children: React.ReactNode;
  className?: string;
  backdropClassName?: string;
  disablePointerClickOutside?: boolean;
  shouldPreventDismiss?: () => boolean;
  shouldBeTopmost?: boolean;
  variant?: ModalVariant;
  noPadding?: boolean;
  /** If set, the close button will be hidden for the specified duration in milliseconds */
  disableCloseButtonUntilMs?: number;
};

const FOCUSABLE_SELECTOR =
  'a[href],button:not([disabled]),textarea:not([disabled]),input:not([disabled]),select:not([disabled]),[tabindex]:not([tabindex="-1"])';

function getFocusableElements(container: HTMLElement): HTMLElement[] {
  return Array.from(container.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR)).filter(
    (element) =>
      !element.hasAttribute("disabled") && element.getAttribute("aria-hidden") !== "true",
  );
}

export function Modal({
  onClose,
  children,
  className,
  backdropClassName,
  disablePointerClickOutside,
  shouldPreventDismiss,
  shouldBeTopmost = false,
  variant = "secondary",
  noPadding = false,
}: ModalProps) {
  const dialogRef = useRef<HTMLDivElement>(null);
  const previousFocusedElementRef = useRef<HTMLElement | null>(null);

  const handleBackdropClick = (event: React.MouseEvent<HTMLDivElement>) => {
    event.stopPropagation();
    if (
      event.target === event.currentTarget &&
      !disablePointerClickOutside &&
      !shouldPreventDismiss?.()
    ) {
      onClose("click_outside");
    }
  };

  const modalContext = useModalContext();

  useKeyboardShortcut({
    shouldHandle: (e) => {
      if (e.key !== "Escape") return false;

      if (shouldPreventDismiss?.()) return false;

      // If there's an active confirmation and this modal's onClose is NOT the hideConfirmation function,
      // then this is a regular modal and should defer to the confirmation
      if (
        modalContext.currentConfirmationName &&
        onClose !== modalContext.hideConfirmation
      ) {
        return false;
      }

      return true;
    },
    onDown: (event) => {
      event.preventDefault();
      event.stopPropagation();
      onClose("escape_key");
    },
  });

  useEffect(() => {
    const dialog = dialogRef.current;
    if (!dialog) {
      return;
    }

    previousFocusedElementRef.current = document.activeElement as HTMLElement | null;

    const focusFirstInsideDialog = () => {
      const focusableElements = getFocusableElements(dialog);
      const target = focusableElements[0] ?? dialog;
      target.focus();
    };

    focusFirstInsideDialog();

    const handleFocusIn = (event: FocusEvent) => {
      const target = event.target;
      if (!(target instanceof Node)) {
        return;
      }
      if (!dialog.contains(target)) {
        // Allow focus outside this dialog when it belongs to:
        // - Radix UI portaled content (popper/portal) triggered from within this modal
        // - Another dialog stacked on top of this one (e.g. a confirmation modal)
        // Without these exemptions, stacked modals fight over focus in an infinite loop.
        if (
          target instanceof Element &&
          target.closest(
            "[data-radix-popper-content-wrapper], [data-radix-portal], [role='dialog']",
          )
        ) {
          return;
        }
        focusFirstInsideDialog();
      }
    };

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key !== "Tab") {
        return;
      }

      const focusableElements = getFocusableElements(dialog);
      if (focusableElements.length === 0) {
        event.preventDefault();
        dialog.focus();
        return;
      }

      const firstElement = focusableElements[0];
      const lastElement = focusableElements[focusableElements.length - 1];
      const activeElement = document.activeElement;

      if (!event.shiftKey && activeElement === lastElement) {
        event.preventDefault();
        firstElement.focus();
      }

      if (event.shiftKey && activeElement === firstElement) {
        event.preventDefault();
        lastElement.focus();
      }
    };

    document.addEventListener("focusin", handleFocusIn);
    document.addEventListener("keydown", handleKeyDown);

    return () => {
      document.removeEventListener("focusin", handleFocusIn);
      document.removeEventListener("keydown", handleKeyDown);
      previousFocusedElementRef.current?.focus();
    };
  }, []);

  const modalElement = (
    <motion.div
      className={clsx(
        styles.modal_backdrop,
        backdropClassName,
        shouldBeTopmost && styles.modal_backdrop_topmost,
      )}
      onMouseDown={handleBackdropClick}
      {...backdropAnimationProps}
    >
      <motion.div
        ref={dialogRef}
        className={styles.modal_content_wrapper}
        onClick={(e: React.MouseEvent<HTMLDivElement, MouseEvent>) => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
        tabIndex={-1}
        {...modalAnimationProps}
      >
        <div className={styles.modal_scrollable_container}>
          <div
            className={clsx(
              styles.modal_children_container,
              styles[`modal_children_container_${variant}`],
              noPadding && styles.no_padding,
              className,
            )}
          >
            {children}
          </div>
        </div>
      </motion.div>
    </motion.div>
  );

  if (shouldBeTopmost) {
    return ReactDOM.createPortal(
      modalElement,
      document.getElementById("topmost-modal-root") as HTMLElement,
    );
  } else {
    return modalElement;
  }
}
