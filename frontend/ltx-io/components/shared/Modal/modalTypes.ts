import { ReactNode } from "react";

export type CloseReason =
  | "close"
  | "cancel"
  | "escape_key"
  | "click_outside"
  | "scroll_away"
  | "auto_dismiss"
  | "error"
  | "apply"
  | "confirm"
  | "done"
  | "user_selection"
  | "paywall_interruption"
  | "custom_voice_modal"
  | "delete_custom_voice"
  | "deleted";

export type ElementModalSource =
  | "tile"
  | "image_ref"
  | "elements_page"
  | "mentions_menu"
  | "moodboard";

export type ModalSource =
  | "premium_picture_model"
  | "premium_resolution"
  | "lightbox"
  | "moodboard_context_menu"
  | ElementModalSource;

export type ModalVariant = "primary" | "secondary" | "transparent";

export interface ShowModalParams {
  content: ReactNode;
  modalName: string;
  modalClassName?: string;
  variant?: ModalVariant;
  disablePointerClickOutside?: boolean;
  shouldPreventDismiss?: () => boolean;
  noPadding?: boolean;
  disableCloseButtonUntilMs?: number;
  shouldBeTopmost?: boolean;
  onClose?: (reason: CloseReason) => void;
}
