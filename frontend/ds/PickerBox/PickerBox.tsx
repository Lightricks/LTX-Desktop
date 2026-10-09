import { clsx } from "clsx";
import { forwardRef } from "react";

import { Badge } from "@ds/Badge/Badge";
import ArrowDownIcon from "@ds/assets/Icons/Arrow/Down/Small.svg?react";

import { Text, TextSize, TextVariant } from "../Text/Text";
import { Thumbnail, ThumbnailProps } from "../Thumbnail/Thumbnail";

import styles from "./PickerBox.module.scss";

export const PICKER_BOX_SPACING = ["tight", "compact", "comfortable"] as const;
export type PickerBoxSpacing = (typeof PICKER_BOX_SPACING)[number];

export const PICKER_BOX_STATE = [
  "enabled",
  "disabled",
  "active",
  "readOnly",
] as const;
export type PickerBoxState = (typeof PICKER_BOX_STATE)[number];

/**
 * Interface for the PickerBox props
 * @interface PickerBoxProps
 */
export interface PickerBoxBaseProps {
  /** Description of spacing prop */
  spacing?: PickerBoxSpacing;
  /** Description of state prop */
  state?: PickerBoxState;
  /** The value of the component */
  value?: string;
  /** The placeholder to be rendered inside the component */
  placeholder?: string;
  /** Whether to hide the placeholder when the value is empty */
  hidePlaceholder?: boolean;
  /** The badge to be rendered after the value */
  badge?: React.ReactElement<typeof Badge>;
  /** Additional CSS class names to be applied */
  className?: string;
  /** Optional class name for the text */
  textClassName?: string;
  /** Optional inline style for the text */
  textStyle?: React.CSSProperties;
  /** Text variant for the value / placeholder (defaults to label). */
  textVariant?: TextVariant;
  /** Text size for the value / placeholder (defaults to md). */
  textSize?: TextSize;
  /** Whether the component is disabled */
  isDisabled?: boolean;
  /** Minimum width of the component */
  minWidth?: string;
  /** Maximum width of the component */
  maxWidth?: string;
  /** Whether to hide the arrow icon */
  hideArrow?: boolean;
  /** Whether to rotate the arrow icon on active state */
  shouldRotateArrowOnActive?: boolean;
  /** Click handler for the component */
  onClick?: (event: React.MouseEvent<HTMLDivElement>) => void;
}

/** Props when using an icon */
interface PickerBoxWithIconProps extends PickerBoxBaseProps {
  icon?: React.ReactNode;
  thumbnail?: never;
}

/** Props when using a thumbnail */
interface PickerBoxWithThumbnailProps extends PickerBoxBaseProps {
  thumbnail?: ThumbnailProps;
  icon?: never;
}

/** Props for the PickerBox component */
export type PickerBoxProps =
  | PickerBoxWithIconProps
  | PickerBoxWithThumbnailProps;

/**
 * This component serves as a customizable trigger for dropdown/picker components.
 * It can be used standalone or as a trigger for DropdownPicker.
 *
 * @component
 * @example
 * ```tsx
 * // Basic usage
 * <PickerBox value="Selected Value" />
 *
 * // As a DropdownPicker trigger
 * <DropdownPicker>
 *   <PickerBox value={selectedValue} placeholder="Select an option" />
 * </DropdownPicker>
 * ```
 */
export const PickerBox = forwardRef<
  HTMLDivElement,
  PickerBoxProps & React.HTMLAttributes<HTMLDivElement>
>((props, ref): JSX.Element => {
  const {
    spacing = "compact",
    state = "enabled",
    value,
    placeholder = "Select an option",
    hidePlaceholder = false,
    icon,
    thumbnail,
    badge,
    className,
    textClassName,
    textStyle,
    textVariant = "label",
    textSize = "md",
    isDisabled,
    minWidth,
    maxWidth,
    hideArrow,
    shouldRotateArrowOnActive,
    onClick,
    ...restProps // Spread remaining props for Radix UI
  } = props;

  const handleClick = (event: React.MouseEvent<HTMLDivElement>) => {
    if (isDisabled) return;
    onClick?.(event);
  };

  return (
    <div
      ref={ref}
      onClick={handleClick}
      className={clsx(
        styles.container,
        styles[spacing],
        styles[state],
        !value && styles.placeholder,
        isDisabled && styles.disabled,
        className,
      )}
      data-testid="picker-box"
      data-spacing={spacing}
      data-state={state}
      aria-label={value || placeholder}
      aria-disabled={isDisabled}
      style={{
        minWidth: minWidth,
        maxWidth: maxWidth,
      }}
      {...restProps}
    >
      {icon && <div className={styles.icon}>{icon}</div>}
      {thumbnail && <Thumbnail size="sm" {...thumbnail} />}
      {(value || (!value && !hidePlaceholder)) && (
        <Text
          className={clsx(styles.text, textClassName)}
          variant={textVariant}
          size={textSize}
          shouldTruncate
          shouldBreakAllWords
          style={textStyle}
        >
          {value ? value : placeholder}
        </Text>
      )}
      {badge && badge}
      {!hideArrow && (
        <div
          className={clsx(
            styles.arrow,
            shouldRotateArrowOnActive && state === "active" && styles.rotated,
          )}
        >
          <ArrowDownIcon />
        </div>
      )}
    </div>
  );
});

PickerBox.displayName = "PickerBox";
