import { Button } from "@ds/Button/Button";
import { X } from "lucide-react";
import { type ReactNode, useCallback, useRef } from "react";

import { useModalContext } from "./ModalContext";
import styles from "./useCardModal.module.scss";

export type CardModalClose = {
  /** Close after the user used the result. */
  apply: () => void;
  cancel: () => void;
  /** Return true to intercept the close button and Escape key. */
  setShouldPreventDismiss: (fn: (() => boolean) | undefined) => void;
};

/**
 * Opens a padding-free card that only its own buttons close. The content decides what each
 * button does, so a stray click outside never loses work in progress. `closeButton` adds a
 * close button in the top corner, the same as the Assets lightbox, for content that has none
 * of its own.
 */
export function useCardModal(
  modalName: string,
  { closeButton = false }: { closeButton?: boolean } = {},
): (render: (close: CardModalClose) => ReactNode) => void {
  const { showModal, hideModal } = useModalContext();
  const shouldPreventDismissRef = useRef<(() => boolean) | undefined>(undefined);

  return useCallback(
    (render) => {
      shouldPreventDismissRef.current = undefined;
      const close: CardModalClose = {
        apply: () => hideModal("apply"),
        cancel: () => hideModal("cancel"),
        setShouldPreventDismiss: (fn) => {
          shouldPreventDismissRef.current = fn;
        },
      };
      const requestCancel = () => {
        if (shouldPreventDismissRef.current?.()) return;
        close.cancel();
      };
      const content = render(close);
      showModal({
        content: closeButton ? (
          <div className={styles.card}>
            <Button
              className={styles.closeButton}
              appearance="overlay"
              hierarchy="secondary"
              size="md"
              isIconOnly
              leftIcon={<X />}
              aria-label="Close"
              onClick={requestCancel}
            />
            {content}
          </div>
        ) : (
          content
        ),
        modalName,
        variant: "secondary",
        noPadding: true,
        disablePointerClickOutside: true,
        shouldPreventDismiss: () => shouldPreventDismissRef.current?.() ?? false,
      });
    },
    [closeButton, hideModal, modalName, showModal],
  );
}
