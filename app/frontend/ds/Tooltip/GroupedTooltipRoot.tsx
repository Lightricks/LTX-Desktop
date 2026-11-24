import { Tooltip as BaseTooltip } from "@base-ui/react/tooltip";
import { clsx } from "clsx";
import { type ComponentProps, type ReactNode } from "react";

import { type ThemeControlClass } from "@ds/styles/themes/useThemeVariables";

import styles from "./Tooltip.module.scss";
import { TooltipArrowIcon } from "./TooltipArrowIcon";

type PositionerProps = ComponentProps<typeof BaseTooltip.Positioner>;
type GroupedTooltipHandle<TPayload> = ReturnType<
  typeof BaseTooltip.createHandle<TPayload>
>;

export interface GroupedTooltipRootProps<TPayload> {
  handle: GroupedTooltipHandle<TPayload>;
  renderContent: (payload: TPayload) => ReactNode;
  side?: PositionerProps["side"];
  align?: PositionerProps["align"];
  sideOffset?: number;
  alignOffset?: number;
  collisionPadding?: number;
  hideArrow?: boolean;
  themeClassName?: ThemeControlClass;
  className?: string;
  keepMounted?: boolean;
}

export function GroupedTooltipRoot<TPayload>({
  handle,
  renderContent,
  side = "top",
  align = "center",
  sideOffset = 14,
  alignOffset,
  collisionPadding = 8,
  hideArrow = false,
  themeClassName = "inverted-theme-style",
  className,
  keepMounted = true,
}: GroupedTooltipRootProps<TPayload>) {
  return (
    <BaseTooltip.Root handle={handle}>
      {({ payload }) => {
        if (payload === undefined || payload === null) {
          return null;
        }

        return (
          <BaseTooltip.Portal keepMounted={keepMounted}>
            <BaseTooltip.Positioner
              side={side}
              align={align}
              sideOffset={sideOffset}
              alignOffset={alignOffset}
              collisionPadding={collisionPadding}
              className={styles.positionTransition}
            >
              <BaseTooltip.Popup
                className={clsx(themeClassName, styles.tooltip, className)}
              >
                {!hideArrow && (
                  <BaseTooltip.Arrow className={styles.tooltipArrow}>
                    <TooltipArrowIcon className={styles.tooltipArrowSvg} />
                  </BaseTooltip.Arrow>
                )}
                <BaseTooltip.Viewport className={styles.tooltipViewport}>
                  <div className={styles.tooltipContent}>{renderContent(payload)}</div>
                </BaseTooltip.Viewport>
              </BaseTooltip.Popup>
            </BaseTooltip.Positioner>
          </BaseTooltip.Portal>
        );
      }}
    </BaseTooltip.Root>
  );
}
