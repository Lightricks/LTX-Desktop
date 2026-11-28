import type { ToastOwnProps } from "./Toast";

export const DEFAULT_AUTOCLOSE_MS = 3000;

export type ActiveToast = ToastOwnProps & {
  id: number;
  autoCloseAfter?: number;
};

type Listener = (toast: ActiveToast | null) => void;

let listener: Listener | null = null;
let nextId = 1;
let hideTimer: number | null = null;

export function subscribeToasts(fn: Listener) {
  listener = fn;
  return () => {
    if (listener === fn) listener = null;
  };
}

export function showSuccessToast(
  message: string,
  options?: Partial<ToastOwnProps> & { autoCloseAfter?: number },
) {
  return createToast({
    message,
    toastType: "success",
    closeable: true,
    autoCloseAfter: DEFAULT_AUTOCLOSE_MS,
    ...options,
  });
}

export function createToast(
  props: ToastOwnProps & {
    autoCloseAfter?: number;
  },
) {
  const id = nextId++;
  listener?.({ ...props, id });
  if (hideTimer != null) window.clearTimeout(hideTimer);
  hideTimer = null;
  if (props.autoCloseAfter) {
    hideTimer = window.setTimeout(() => dismissToast(), props.autoCloseAfter);
  }
  return id;
}

export function dismissToast() {
  if (hideTimer != null) {
    window.clearTimeout(hideTimer);
    hideTimer = null;
  }
  listener?.(null);
}
