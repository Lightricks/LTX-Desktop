import { clsx } from "clsx";
import { forwardRef } from "react";

import { ActivityCircular } from "@ds/ActivityCircular/ActivityCircular";
import { Text } from "@ds/Text/Text";

import styles from "./Button.module.scss";
import { type ButtonProps, SPINNER_SIZE_MAP } from "./types";

function getSpinnerAppearance(
  appearance: string,
  hierarchy: string,
): "default" | "over-background" {
  const variantKey = `${appearance}_${hierarchy}`;

  // These combinations have light backgrounds, use default (dark) spinner
  const defaultSpinnerVariants = [
    "brand_secondary",
    "brand_elevated",
    "neutral_secondary",
    "neutral_plain",
    "neutral_elevated",
    "overlay_secondary",
    "white_primary",
    "white_secondary",
  ];

  return defaultSpinnerVariants.includes(variantKey)
    ? "default"
    : "over-background";
}

/**
 * Button is a flexible button component that adapts its content and styling
 * based on composition, appearance, hierarchy, and state.
 */
export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  (props, ref) => {
    const {
      appearance = "neutral",
      hierarchy = "secondary",
      size = "md",
      isLoading,
      isActive,
      isIconOnly,
      leftIcon,
      rightIcon,
      label,
      className,
      onClick,
      type = "button",
      ...rest
    } = props;

    const buttonClasses = clsx(
      styles.button,
      isLoading && styles.loading,
      size && styles[size],
      appearance && styles[appearance],
      hierarchy && styles[hierarchy],
      isIconOnly && styles.iconOnly,
      isActive && styles.active,
      className,
    );
    const spinnerAppearance = getSpinnerAppearance(appearance, hierarchy);
    const spinnerSize = SPINNER_SIZE_MAP[size];

    const shouldShowLeftSpinner = isLoading && (!!leftIcon || !rightIcon);
    const shouldShowRightSpinner = isLoading && !!rightIcon && !leftIcon;

    const renderLeftIcon = () => {
      if (!leftIcon && !shouldShowLeftSpinner) return null;

      return (
        <div className={styles.iconContainer}>
          {shouldShowLeftSpinner ? (
            <ActivityCircular
              appearance={spinnerAppearance}
              size={spinnerSize}
            />
          ) : (
            leftIcon
          )}
        </div>
      );
    };

    const renderRightIcon = () => {
      if (!rightIcon && !shouldShowRightSpinner) return null;

      return (
        <div className={styles.iconContainer}>
          {shouldShowRightSpinner ? (
            <ActivityCircular
              appearance={spinnerAppearance}
              size={spinnerSize}
            />
          ) : (
            rightIcon
          )}
        </div>
      );
    };

    const renderLabel = () => {
      if (!label) return null;

      return (
        <Text
          as="span"
          variant="label"
          size={size === "xl" ? "lg" : size}
          align="center"
          shouldTruncate
          shouldBreakAllWords={true}
          className={styles.label}
        >
          {label}
        </Text>
      );
    };

    return (
      <button
        ref={ref}
        type={type}
        className={buttonClasses}
        onClick={isLoading ? undefined : onClick}
        {...rest}
        // Mobile chrome map: steps padding/height/icons by size.
        data-button-size={size}
        data-icon-only={isIconOnly ? "true" : undefined}
      >
        {renderLeftIcon()}
        {renderLabel()}
        {renderRightIcon()}
      </button>
    );
  },
);

Button.displayName = "Button";
