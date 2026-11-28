// framer-motion presets for popover/modal animations.

export const DEFAULT_POPOVER_OPTIONS = {
  auto: true,
  overflowContainer: true,
  triggerOffset: 5,
  containerOffset: 10,
};

export const animationProps = {
  initial: { opacity: 0, scale: 0.97, y: 5 },
  animate: { opacity: 1, scale: 1, y: 0 },
  exit: { opacity: 0, scale: 0.97, y: 5 },
  transition: { duration: 0.1 },
};

export const modalAnimationProps = {
  initial: { opacity: 0, scale: 0.96 },
  animate: { opacity: 1, scale: 1 },
  exit: { opacity: 0, scale: 0.96 },
  transition: { duration: 0.1 },
};

export const backdropAnimationProps = {
  initial: { opacity: 0 },
  animate: { opacity: 1 },
  exit: { opacity: 0 },
  transition: { duration: 0.15 },
};
