import { clsx } from "clsx";
import { AnimatePresence } from "framer-motion";
import { ReactNode, useCallback, useState } from "react";

import {
  CloseReason,
  ShowConfirmationProps,
} from "@/ltx-io/components/shared/Modal/ModalContext.tsx";

import { Modal } from "./Modal";
import styles from "./Modal.module.scss";

type UseConfirmationReturn = {
  Confirmation: JSX.Element | null;
  showConfirmation: (params: ShowConfirmationProps) => void;
  hideConfirmation: (reason: CloseReason) => void;
  confirmationName: string;
};

export function useConfirmation(): UseConfirmationReturn {
  const [isVisible, setIsVisible] = useState(false);
  const [content, setContent] = useState<ReactNode>(null);
  const [name, setName] = useState<string>("");
  const [shouldBeTopmost, setShouldBeTopmost] = useState<boolean>(false);
  const [confirmationClassName, setConfirmationClassName] = useState<
    string | undefined
  >();

  const showConfirmation = useCallback(
    ({
      content,
      confirmationName,
      confirmationClassName,
      shouldBeTopmost = false,
    }: ShowConfirmationProps) => {
      setShouldBeTopmost(shouldBeTopmost);
      setIsVisible(true);
      setContent(content);
      setName(confirmationName);
      setConfirmationClassName(confirmationClassName);
    },
    [],
  );

  const hideConfirmation = useCallback((_reason: CloseReason) => {
    setIsVisible(false);
    setContent(null);
    setName("");
    setConfirmationClassName(undefined);
    setShouldBeTopmost(false);
  }, []);

  const ConfirmationComponent = (
    <AnimatePresence>
      {isVisible && (
        <Modal
          className={clsx(styles.confirmation_children_container, confirmationClassName)}
          onClose={hideConfirmation}
          shouldBeTopmost={shouldBeTopmost}
        >
          {content}
        </Modal>
      )}
    </AnimatePresence>
  );

  return {
    Confirmation: ConfirmationComponent,
    showConfirmation,
    hideConfirmation,
    confirmationName: name,
  };
}
