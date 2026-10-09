import { Tooltip as BaseTooltip } from "@base-ui/react/tooltip";
import { clsx } from "clsx";
import { type ComponentProps, isValidElement } from "react";

import { ThemeControlClass } from "@ds/styles/themes/useThemeVariables";
import {
  useThemeColorScheme,
  useThemeRootElement,
} from "@ds/styles/themes/useTheme";
import { logMessage } from "@ds/lib/handleErrors";

import styles from "./Tooltip.module.scss";
import { TooltipArrowIcon } from "./TooltipArrowIcon";

type PositionerProps = ComponentProps<typeof BaseTooltip.Positioner>;
export type TooltipSide = PositionerProps["side"];
export type TooltipAlign = PositionerProps["align"];

/**
 * Props for the Tooltip component.
 *
 * Built on Base UI Tooltip primitives. Pass any ReactNode as `content`;
 * the container applies `text_style_body_lg` by default which custom
 * content can override.
 */
export interface TooltipProps {
  /** Bold heading text displayed above content */
  title?: string;
  /** Theme class for styling — defaults to inverted (dark) theme */
  themeClassName?: ThemeControlClass;
  /** Tooltip body — accepts any ReactNode */
  content?: React.ReactNode;
  /** Delay before showing (ms). Defaults to {@link TOOLTIP_DEFAULT_OPEN_DELAY_MS}. */
  delay?: number;
  /** Delay before hiding (ms). Omit to inherit from Provider. */
  closeDelay?: number;
  /** Whether to hide the arrow indicator */
  hideArrow?: boolean;
  /** Which side of the trigger to place the tooltip */
  side?: TooltipSide;
  /** Alignment along the side axis */
  align?: TooltipAlign;
  /** Whether the tooltip is disabled (renders only children) */
  disabled?: boolean;
  /** Controlled open state */
  open?: boolean;
  /** Callback when open state changes (for controlled tooltips) */
  onOpenChange?: (open: boolean) => void;
  /** Distance from the trigger element (pixels) */
  sideOffset?: number;
  /** Offset along the alignment axis (pixels) */
  alignOffset?: number;
  /** Maximum width of the tooltip (pixels) */
  maxWidth?: number;
  /** Additional CSS class name for the tooltip content */
  className?: string;
  /** Portal target; defaults to the `.ltx-io` theme root. Use `document.body` to escape local stacking contexts. */
  portalContainer?: HTMLElement | null;
  /** The trigger element */
  children: React.ReactNode;
}

/**
 * Provides global settings (delay, closeDelay) for all Tooltip components in the tree.
 * Useful in Storybook stories and tests to override the default open delay.
 *
 * @example
 * ```tsx
 * <TooltipProvider delay={0}>
 *   <MyComponent />
 * </TooltipProvider>
 * ```
 */
export const TooltipProvider = BaseTooltip.Provider;

/** Max width for simple (text-only) tooltips */
const SIMPLE_TOOLTIP_MAX_WIDTH = 200;

/** Default open delay (ms). Short so hover hints feel responsive. */
export const TOOLTIP_DEFAULT_OPEN_DELAY_MS = 70;

function isForwardRefComponent(type: unknown): boolean {
  if (typeof type !== "object" || type === null) return false;
  const record = type as Record<string, unknown>;
  const sym = record.$$typeof;
  // React.forwardRef
  if (sym === Symbol.for("react.forward_ref")) return true;
  // React.memo(React.forwardRef(...))
  const inner = record.type as Record<string, unknown> | undefined;
  if (
    sym === Symbol.for("react.memo") &&
    inner?.$$typeof === Symbol.for("react.forward_ref")
  )
    return true;
  return false;
}

/**
 * A tooltip component built on Base UI.
 *
 * @component
 * @example
 * ```tsx
 * <Tooltip content="Hello" side="top">
 *   <button>Hover me</button>
 * </Tooltip>
 * ```
 */
export function Tooltip({
  title,
  content = null,
  side = "top",
  align = "center",
  delay = TOOLTIP_DEFAULT_OPEN_DELAY_MS,
  closeDelay,
  hideArrow,
  disabled,
  open,
  onOpenChange,
  sideOffset = 10,
  alignOffset,
  maxWidth: maxWidthProp,
  className,
  portalContainer,
  children,
  themeClassName = "inverted-theme-style",
}: TooltipProps) {
  const isControlled = open !== undefined;

  const rootElement = useThemeRootElement();
  const colorScheme = useThemeColorScheme();
  const portalTarget = portalContainer ?? rootElement ?? undefined;
  const portalingToBody =
    typeof document !== "undefined" && portalTarget === document.body;

  const maxWidth = maxWidthProp ?? SIMPLE_TOOLTIP_MAX_WIDTH;
  const tooltipClassName = clsx(themeClassName, styles.tooltip, className);

  // Legacy: React 18 requires forwardRef for components to accept refs.
  // React 19 removes this requirement, making this wrapper unnecessary.
  // Many consumers currently pass components without forwardRef, so we keep
  // the fallback wrapper until the React 19 upgrade.
  const canReceiveRef =
    isValidElement(children) &&
    (typeof children.type === "string" || isForwardRefComponent(children.type));

  if (process.env.NODE_ENV === "development" && !canReceiveRef) {
    logMessage(
      "Tooltip: child component does not forward refs. " +
        "Wrapping in a <span> for ref forwarding. Prefer using forwardRef on the child.",
      "warning",
    );
  }

  const referenceChild = canReceiveRef ? (
    children
  ) : (
    <span className={styles.referenceWrapper}>{children}</span>
  );

  const positioner = (
    <BaseTooltip.Positioner
      side={side}
      align={align}
      sideOffset={sideOffset}
      alignOffset={alignOffset}
      collisionPadding={8}
      className={clsx(
        styles.positioner,
        portalingToBody && styles.positionerAboveOverlays,
      )}
    >
      <BaseTooltip.Popup className={tooltipClassName} style={{ maxWidth }}>
        <div className={styles.tooltipContent}>
          {title && <div className={styles.tooltipTitle}>{title}</div>}
          {content && <div className={styles.tooltipBody}>{content}</div>}
        </div>
        {!hideArrow && (
          <BaseTooltip.Arrow className={styles.tooltipArrow}>
            <TooltipArrowIcon className={styles.tooltipArrowSvg} />
          </BaseTooltip.Arrow>
        )}
      </BaseTooltip.Popup>
    </BaseTooltip.Positioner>
  );

  return (
    <BaseTooltip.Root
      disabled={disabled}
      open={isControlled ? open : undefined}
      onOpenChange={isControlled || onOpenChange ? onOpenChange : undefined}
    >
      <BaseTooltip.Trigger
        render={referenceChild}
        {...(delay !== undefined && { delay })}
        {...(closeDelay !== undefined && { closeDelay })}
      />
      <BaseTooltip.Portal container={portalTarget}>
        {portalingToBody ? (
          <div className="ltx-io" data-theme={colorScheme}>
            {positioner}
          </div>
        ) : (
          positioner
        )}
      </BaseTooltip.Portal>
    </BaseTooltip.Root>
  );
}
