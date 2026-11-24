import * as RadixSelect from "@radix-ui/react-select";
import { clsx } from "clsx";
import React, { forwardRef, useState } from "react";

import { ActivityIndicator } from "@ds/ActivityIndicator/ActivityIndicator";
import {
  UseInfiniteScrollOptions,
  useInfiniteScroll,
} from "@ds/lib/useInfiniteScroll";

import { PickerBox, PickerBoxBaseProps } from "../PickerBox/PickerBox";
import { ThumbnailProps } from "../Thumbnail/Thumbnail";

import styles from "./DropdownPicker.module.scss";
import { renderSupportedOptionWithAdditionalProps } from "./utils";

export interface PickerOption {
  /** Value of the option */
  value: string;
  /** Label to display in the trigger when an option is selected. It will never be used in the dropdown. */
  label: string;
  /** Render function to display any content for the option in the dropdown */
  render: React.ReactNode;
  /** Whether the option is disabled */
  isDisabled?: boolean;
  /** Whether the option can be selected. Defaults to true. */
  isSelectable?: boolean;
}

export interface ContentProps extends RadixSelect.SelectContentProps {
  /** Side offset from the trigger element. Default is 5 */
  sideOffset?: number;
  /** Offset from the trigger element. Default is 0 */
  alignOffset?: number;
  /** Padding to maintain when handling collisions. Default is 20 */
  collisionPadding?: number;
}

export interface DropdownPickerTriggerBaseProps extends PickerBoxBaseProps {
  /** Minimum width of the dropdown. Default is "160px" */
  minWidth?: string;
  /** Maximum width of the dropdown. Default is "auto" */
  maxWidth?: string;
}

interface DropdownPickerTriggerWithIconProps extends DropdownPickerTriggerBaseProps {
  /** Icon to display in the picker */
  icon?: React.ReactNode;
  thumbnail?: never;
}

interface DropdownPickerTriggerWithThumbnailProps extends DropdownPickerTriggerBaseProps {
  /** Thumbnail to display in the picker */
  thumbnail?: ThumbnailProps;
  icon?: never;
}

export type DropdownPickerTriggerProps =
  | DropdownPickerTriggerWithIconProps
  | DropdownPickerTriggerWithThumbnailProps;

export interface DropdownPickerProps extends Omit<
  UseInfiniteScrollOptions,
  "itemsCount" | "isContainerOpen" | "fillContainer"
> {
  /** Array of options to be rendered */
  options: PickerOption[];
  /** React node to use as trigger. If not provided, will use PickerBox */
  children?: React.ReactNode;
  /** Label for the picker */
  label?: string;
  /** Whether the dropdown is disabled */
  isDisabled?: boolean;
  /** Callback function to handle value change */
  onValueChange: (value: string) => void;
  /** Control the open state of the dropdown. If not provided, the dropdown will be uncontrolled */
  isOpen?: boolean;
  /** Minimum width of the dropdown. Default is "160px" */
  minWidth?: string;
  /** Maximum width of the dropdown. Default is "auto" */
  maxWidth?: string;
  /** Maximum height of the dropdown. Default is "auto" */
  maxHeight?: string;
  /** Value of the selected option */
  value?: string;
  /** Props to pass to the default PickerBox trigger */
  pickerBoxProps?: Partial<DropdownPickerTriggerProps>;
  /**
   * Additional RadixUI Props for the Root component
   * @see {@link RadixSelect.SelectProps}
   */
  rootProps?: RadixSelect.SelectProps;
  /**
   * Props for the Trigger component
   * @see {@link RadixSelect.SelectContentProps}
   */
  contentProps?: ContentProps;
  /** Callback function to handle open change */
  onOpenChange?: (open: boolean) => void;
  /** Callback when hovering over an item (value) or leaving items (null) */
  onItemHover?: (value: string | null) => void;
  /** Optional DOM node to contain the portaled picker content. */
  portalContainer?: HTMLElement | null;
}

export const DropdownPicker = forwardRef<
  HTMLButtonElement,
  DropdownPickerProps
>(function DropdownPicker(props, forwardedRef) {
  const {
    options,
    value,
    label = "Dropdown picker",
    minWidth = "var(--radix-select-trigger-width)",
    maxWidth = "auto",
    maxHeight = "auto",
    hasNextPage,
    isFetchingNextPage,
    isDisabled,
    isOpen: controlledIsOpen,
    onValueChange,
    onNextPage,
    onOpenChange,
    bottomThreshold = 150,
    rootProps = {},
    contentProps = {},
    pickerBoxProps = {
      appearance: "normal",
      spacing: "compact",
      minWidth: "160px",
      maxWidth: "auto",
      placeholder: "Select an option",
      value: props.label,
    },
    children,
    onItemHover,
    portalContainer,
  } = props;

  const {
    sideOffset = 4,
    alignOffset = 0,
    collisionPadding = 20,
    style: contentStyle,
    className: contentClassName,
    ...restContentProps
  } = contentProps;

  const [uncontrolledIsOpen, setUncontrolledIsOpen] = useState(false);
  const isControlled = controlledIsOpen !== undefined;
  const isOpen = isControlled ? controlledIsOpen : uncontrolledIsOpen;

  const { scrollRef, InfiniteScrollSentinel } = useInfiniteScroll({
    hasNextPage,
    onNextPage,
    bottomThreshold,
    itemsCount: options.length,
    isFetchingNextPage,
  });

  const handleOpenChange = (valueIsOpen: boolean) => {
    if (!isControlled) {
      setUncontrolledIsOpen(valueIsOpen);
    }
    onOpenChange?.(valueIsOpen);
  };

  // Prevent selection of an item if an inner button is clicked
  function handleItemClick(event: React.PointerEvent): void {
    if (!(event.target instanceof Element)) return;

    const isButton = event.target.closest("button");

    if (isButton) {
      event.preventDefault();
      event.stopPropagation();
    }
  }

  return (
    <RadixSelect.Root
      /*
       * We need to use "" as a default value to make sure that resetting the selected option works.
       * This works around a bug in RadixUI.Select where the value cannot be set to undefined after
       * it has been set to anything else. See https://github.com/radix-ui/primitives/issues/1569
       *
       * Unfortunately the alternative solution suggested in that post of setting key={value} breaks
       * our storybook tests, so we can't use it.
       */
      value={value ?? ""}
      open={isOpen}
      onValueChange={onValueChange}
      onOpenChange={handleOpenChange}
      disabled={isDisabled}
      aria-label={label}
      {...rootProps}
    >
      <RadixSelect.Trigger asChild ref={forwardedRef}>
        {children || (
          <PickerBox
            style={{
              minWidth: pickerBoxProps?.minWidth,
              maxWidth: pickerBoxProps?.maxWidth,
            }}
            isDisabled={isDisabled}
            {...pickerBoxProps}
            state={isOpen ? "active" : "enabled"}
          />
        )}
      </RadixSelect.Trigger>
      <RadixSelect.Portal container={portalContainer ?? undefined}>
        <RadixSelect.Content
          data-testid="dropdown-picker"
          className={clsx(styles.pickerContent, contentClassName)}
          sideOffset={sideOffset}
          alignOffset={alignOffset}
          collisionPadding={collisionPadding}
          style={{ minWidth, maxWidth, maxHeight, ...contentStyle }}
          position="popper"
          {...restContentProps}
        >
          <RadixSelect.Viewport className={styles.viewport} ref={scrollRef}>
            {options.map((option) => {
              const isSelectable = option.isSelectable ?? true;
              const renderedOption = renderSupportedOptionWithAdditionalProps({
                optionRender: option.render,
                isDisabled: option.isDisabled,
                isSelected: isSelectable && value === option.value,
                isSelectable,
              });

              if (!isSelectable) {
                return (
                  <div
                    className={styles.pickerItem}
                    key={option.value}
                    onPointerEnter={() => onItemHover?.(option.value)}
                    onPointerLeave={() => onItemHover?.(null)}
                  >
                    {renderedOption}
                  </div>
                );
              }

              return (
                <RadixSelect.Item
                  className={styles.pickerItem}
                  key={option.value}
                  value={option.value}
                  disabled={option.isDisabled}
                  onPointerDown={handleItemClick}
                  onPointerUp={handleItemClick}
                  onPointerEnter={() => onItemHover?.(option.value)}
                  onPointerLeave={() => onItemHover?.(null)}
                >
                  {renderedOption}
                </RadixSelect.Item>
              );
            })}
            <InfiniteScrollSentinel />
            {isFetchingNextPage && <LoadingItem />}
          </RadixSelect.Viewport>
        </RadixSelect.Content>
      </RadixSelect.Portal>
    </RadixSelect.Root>
  );
});

function LoadingItem() {
  return (
    <RadixSelect.Item className={styles.pickerItemLoading} value="loading">
      <ActivityIndicator size="sm" />
    </RadixSelect.Item>
  );
}
