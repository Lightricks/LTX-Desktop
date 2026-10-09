import { ReactNode, createContext, useContext } from "react";

import {
  CloseReason,
  ShowModalParams,
} from "@/ltx-io/components/shared/Modal/modalTypes.ts";

// Re-export for backward compatibility
export type { CloseReason };

export type ShowConfirmationProps = {
  content: ReactNode;
  confirmationName: string;
  confirmationClassName?: string;
  shouldBeTopmost?: boolean;
};

export type ModalContextType = {
  modal: ReactNode;
  confirmation: ReactNode;

  showModal: (params: ShowModalParams) => void;
  hideModal: (reason: CloseReason) => void;

  showConfirmation: (params: ShowConfirmationProps) => void;
  hideConfirmation: (reason: CloseReason) => void;

  currentModalName: string;
  currentConfirmationName: string;
};

export const ModalContext = createContext<ModalContextType | undefined>(undefined);

export const useModalContext = () => {
  const context = useContext(ModalContext);
  if (!context) {
    throw new Error("useModalContext must be used within a ModalProvider");
  }
  return context;
};
