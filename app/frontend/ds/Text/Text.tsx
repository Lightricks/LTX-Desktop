import { clsx } from "clsx";
import { createElement, forwardRef } from "react";

import styles from "./Text.module.scss";

export const TEXT_VARIANTS = [
  "display",
  "heading",
  "body",
  "label",
  "labelCaps",
  "labelCapsUnderline",
] as const;
export type TextVariant = (typeof TEXT_VARIANTS)[number];

export const TEXT_TAGS = [
  "h1",
  "h2",
  "h3",
  "h4",
  "h5",
  "h6",
  "p",
  "span",
  "div",
] as const;
export type TextTag = (typeof TEXT_TAGS)[number];

export const TEXT_SIZES = {
  display: ["sm", "md", "lg", "xl", "xxl"],
  heading: ["xxs", "xs", "sm", "md", "lg", "xl", "xxl"],
  body: ["xs", "sm", "md", "lg", "xl"],
  label: ["xs", "sm", "md", "lg", "xl", "xxl"],
  labelCaps: ["xs", "sm", "md", "lg", "xl", "xxl"],
  labelCapsUnderline: ["xs", "sm", "md", "lg", "xl", "xxl"],
} as const;
export type TextSize = (typeof TEXT_SIZES)[TextVariant][number];

export const TEXT_ALIGN = ["left", "center", "right"] as const;
export type TextAlign = (typeof TEXT_ALIGN)[number];

export interface TextProps {
  /** The content of the text. */
  children: React.ReactNode;
  /** Unique identifier for the element */
  id?: string;
  /** Native title attribute for hover labels */
  title?: string;
  /** HTML of the element to render */
  as?: TextTag;
  /** The variant of the text. */
  variant?: TextVariant;
  /** Controls the text size. Available sizes vary by variant */
  size?: TextSize;
  /** The alignment of the text. */
  align?: TextAlign;
  /** Whether the text should be truncated. */
  shouldTruncate?: boolean;
  /** Whether the text should be capitalized. */
  shouldCapitalize?: boolean;
  /** Whether the text should break all words. */
  shouldBreakAllWords?: boolean;
  /** Whether the text should be wrapped. */
  shouldWhiteSpacePreLine?: boolean;
  /** Additional CSS classes to apply to the text. */
  className?: string;
  /** The number of lines to truncate the text to. */
  truncateLines?: number;
  /** Inline styles */
  style?: React.CSSProperties;
}

/**
 * The Text component is a fundamental typography building block in our design system. It provides consistent text styling across the application while maintaining semantic HTML structure.
 *
 * ### Usage
 * Use the `as` prop to ensure proper HTML semantics while maintaining visual styles:

  ```tsx
  <Text variant="heading" size="lg" as="h1">Page Title</Text>
  <Text variant="body" size="md" as="p">Content</Text>
  ```

  ### Truncation
  For space-constrained areas, use the `truncate` prop:

  ```tsx
  <Text variant="label" size="md" truncate>Long text that will be truncated...</Text>
  ```

  ### Custom Styling
  Apply additional styles using the `className` prop:

  ```tsx
  <Text className="custom-text">Styled text</Text>
  ```

  ## Accessibility

  - Use appropriate HTML tags via the `as` prop
  - Maintain proper heading hierarchy (`h1-h6`). More on https://developer.mozilla.org/en-US/docs/Web/HTML/Element/Heading_Elements
  - Ensure sufficient color contrast for all text variants
  - Avoid using truncation for critical information
 */
export const Text = forwardRef<HTMLElement, TextProps>(function Text(props, ref) {
  const {
    as = "p",
    variant = "body",
    size = "md",
    align = "left",
    truncateLines = 1,
    children,
    shouldTruncate,
    shouldCapitalize,
    shouldBreakAllWords,
    shouldWhiteSpacePreLine,
    className,
    style,
    ...rest
  } = props;

  const capitalizedAlign = align[0].toUpperCase() + align.slice(1);

  return createElement(
    as,
    {
      ref,
      className: clsx(
        styles[`${variant}${size}`],
        shouldTruncate && styles.truncate,
        shouldBreakAllWords && styles.breakAllWords,
        shouldCapitalize && styles.capitalize,
        shouldWhiteSpacePreLine && styles.whiteSpacePreLine,
        styles[`align${capitalizedAlign}`],
        styles.text,
        className,
      ),
      "data-variant": variant,
      "data-truncate": shouldTruncate,
      "data-size": size,
      style: {
        WebkitLineClamp: shouldTruncate ? truncateLines : undefined,
        ...style,
      },
      ...rest,
    },
    children,
  );
});
