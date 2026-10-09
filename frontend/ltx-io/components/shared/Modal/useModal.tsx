import { AnimatePresence } from "framer-motion";
import { ReactNode, useCallback, useRef, useState } from "react";

import { CloseReason } from "@/ltx-io/components/shared/Modal/ModalContext.tsx";
import { ShowModalParams } from "@/ltx-io/components/shared/Modal/modalTypes.ts";

import { Modal } from "./Modal";
import { ModalVariant } from "./modalTypes";

type UseModalReturn = {
  Modal: JSX.Element | null;
  showModal: (params: ShowModalParams) => void;
  hideModal: (reason: CloseReason) => void;
  modalName: string;
};

export function useModal(): UseModalReturn {
  const [isVisible, setIsVisible] = useState(false);
  const [content, setContent] = useState<ReactNode>(null);
  const [name, setName] = useState<string>("");
  const [modalClassName, setModalClassName] = useState<string | undefined>();
  const [variant, setVariant] = useState<ModalVariant>("secondary");
  const [disablePointerClickOutside, setDisablePointerClickOutside] =
    useState<boolean>(false);
  const [noPadding, setNoPadding] = useState<boolean>(false);
  const [disableCloseButtonUntilMs, setDisableCloseButtonUntilMs] = useState<
    number | undefined
  >(undefined);
  const [shouldBeTopmost, setShouldBeTopmost] = useState(false);

  const onCloseCallback = useRef<((reason: CloseReason) => void) | undefined>();
  const afterCloseCallback = useRef<VoidFunction | undefined>(undefined);
  // Held in a ref rather than state so it is read at dismissal time — the
  // answer depends on whatever the modal's content is doing right now.
  const shouldPreventDismissRef = useRef<(() => boolean) | undefined>(undefined);

  const resolveShouldPreventDismiss = useCallback(
    () => shouldPreventDismissRef.current?.() ?? false,
    [],
  );

  const showModal = useCallback(
    ({
      content,
      modalName,
      modalClassName,
      variant = "secondary",
      disablePointerClickOutside = false,
      shouldPreventDismiss,
      noPadding = false,
      disableCloseButtonUntilMs,
      shouldBeTopmost = false,
      onClose,
    }: ShowModalParams) => {
      onCloseCallback.current = onClose;
      shouldPreventDismissRef.current = shouldPreventDismiss;

      setIsVisible(true);
      setContent(content);
      setName(modalName);
      setModalClassName(modalClassName);
      setVariant(variant);
      setDisablePointerClickOutside(disablePointerClickOutside);
      setNoPadding(noPadding);
      setDisableCloseButtonUntilMs(disableCloseButtonUntilMs);
      setShouldBeTopmost(shouldBeTopmost);
    },
    [],
  );

  const hideModal = useCallback((reason: CloseReason) => {
    setIsVisible(false);

    setContent(null);
    setName("");
    setModalClassName(undefined);
    setVariant("secondary");
    setDisablePointerClickOutside(false);
    setNoPadding(false);
    setDisableCloseButtonUntilMs(undefined);
    setShouldBeTopmost(false);

    shouldPreventDismissRef.current = undefined;

    onCloseCallback.current?.(reason);
    onCloseCallback.current = undefined;

    if (afterCloseCallback.current) {
      afterCloseCallback.current();
      afterCloseCallback.current = undefined;
    }
  }, []);

  const ModalComponent = (
    <AnimatePresence>
      {isVisible && (
        <Modal
          className={modalClassName}
          variant={variant}
          onClose={hideModal}
          disablePointerClickOutside={disablePointerClickOutside}
          shouldPreventDismiss={resolveShouldPreventDismiss}
          noPadding={noPadding}
          disableCloseButtonUntilMs={disableCloseButtonUntilMs}
          shouldBeTopmost={shouldBeTopmost}
        >
          {content}
        </Modal>
      )}
    </AnimatePresence>
  );

  return {
    Modal: ModalComponent,
    showModal,
    hideModal,
    modalName: name,
  };
}
