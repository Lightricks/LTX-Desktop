import { ReactNode } from "react";

import { useConfirmation } from "@/ltx-io/components/shared/Modal/useConfirmation.tsx";

import { ModalContext } from "./ModalContext";
import { useModal } from "./useModal";

type ModalProviderProps = {
  children: ReactNode;
};

export function ModalProvider({ children }: ModalProviderProps) {
  const { Modal, showModal, hideModal, modalName } = useModal();
  const { Confirmation, showConfirmation, hideConfirmation, confirmationName } =
    useConfirmation();

  return (
    <ModalContext.Provider
      value={{
        modal: Modal,
        confirmation: Confirmation,
        showModal,
        hideModal,
        currentModalName: modalName,
        showConfirmation,
        hideConfirmation,
        currentConfirmationName: confirmationName,
      }}
    >
      {children}
      {Modal}
      {Confirmation}
    </ModalContext.Provider>
  );
}
