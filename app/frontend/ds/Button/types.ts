import { ButtonHTMLAttributes } from "react";

/**
 * Valid appearance + hierarchy combinations based on Figma design system.
 * Uses discriminated union to make invalid combinations impossible at compile-time.
 */
export type ButtonVariantProps =
  | { appearance?: "brand"; hierarchy?: "primary" | "secondary" | "plain" | "elevated" }
  | { appearance: "neutral"; hierarchy?: "primary" | "secondary" | "plain" | "elevated" }
  | { appearance: "monetization"; hierarchy?: "primary" }
  | { appearance: "danger"; hierarchy?: "primary" | "secondary" | "plain" }
  | { appearance: "white"; hierarchy?: "primary" | "secondary" | "plain" }
  | { appearance: "overlay"; hierarchy?: "primary" | "secondary" };

export interface ButtonBaseProps
  extends Omit<ButtonHTMLAttributes<HTMLButtonElement>, "children"> {
  /** Whether the button is in loading state */
  isLoading?: boolean;
  /** @deprecated use aria-pressed or aria-expanded to set a button in active state */
  isActive?: boolean;
  /** Button size */
  size?: ButtonSize;
  /** Additional CSS classes */
  className?: string;
}

/** Discriminated union to determine the composition of the button */
export type ButtonProps = ButtonBaseProps &
  ButtonVariantProps &
  (
    | {
        isIconOnly: true;
        leftIcon: React.ReactNode;
        "aria-label": string;
        label?: never;
        rightIcon?: never;
      }
    | {
        label: string;
        leftIcon?: React.ReactNode;
        rightIcon?: React.ReactNode;
        isIconOnly?: never;
        "aria-label"?: string;
      }
  );

export const BUTTON_APPEARANCES = [
  "brand",
  "neutral",
  "monetization",
  "danger",
  "white",
  "overlay",
] as const;
export type ButtonAppearance = (typeof BUTTON_APPEARANCES)[number];

export const BUTTON_HIERARCHIES = ["primary", "secondary", "plain", "elevated"] as const;
export type ButtonHierarchy = (typeof BUTTON_HIERARCHIES)[number];

export const BUTTON_SIZES = ["xs", "sm", "md", "lg", "xl"] as const;
export type ButtonSize = (typeof BUTTON_SIZES)[number];

/** Spinner sizes matching each button size */
export const SPINNER_SIZE_MAP = {
  xs: 9,
  sm: 10.5,
  md: 12,
  lg: 13.5,
  xl: 13.5,
} as const;

export type SpinnerSize = (typeof SPINNER_SIZE_MAP)[keyof typeof SPINNER_SIZE_MAP];
