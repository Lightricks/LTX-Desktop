import { clsx } from "clsx";
import { CSSProperties, forwardRef } from "react";

import { FlexDisplay, LayoutProps, SpacingValue } from "../types";
import { convertSpacing, extractLayout } from "../utils";

import styles from "./Flex.module.scss";

export const FLEX_WRAP = {
  nowrap: "nowrap",
  wrap: "wrap",
  wrapReverse: "wrap-reverse",
} as const;

export const FLEX_JUSTIFY = ["start", "center", "end", "between"] as const;
export const FLEX_ALIGN = ["start", "center", "end", "baseline", "stretch"] as const;

export const FLEX_DIRECTION = {
  row: "row",
  column: "column",
  rowReverse: "row-reverse",
  columnReverse: "column-reverse",
} as const;

/** The direction of the flex container. */
export type FlexDirection = (typeof FLEX_DIRECTION)[keyof typeof FLEX_DIRECTION];

/** The wrapping behavior of the flex container. */
export type FlexWrap = (typeof FLEX_WRAP)[keyof typeof FLEX_WRAP];

/** The justify content alignment of the flex container. */
export type FlexJustify = (typeof FLEX_JUSTIFY)[number];

/** The align items positioning of the flex container. */
export type FlexAlign = (typeof FLEX_ALIGN)[number];

export interface FlexProps extends LayoutProps {
  /** The flex property of the flex container. */
  flex?: number | string;
  /** The display property of the flex container. */
  display?: FlexDisplay;
  /** The direction of the flex container. */
  direction?: FlexDirection;
  /** The wrapping behavior of the flex container. */
  wrap?: FlexWrap;
  /** The justify content alignment of the flex container. */
  justify?: FlexJustify;
  /** The align items positioning of the flex container. */
  align?: FlexAlign;
  /** The grow property of the flex container. */
  grow?: string;
  /** The shrink property of the flex container. */
  shrink?: string;
  /** The gap property of the flex container. */
  gap?: SpacingValue;
  /** The gapX property of the flex container. */
  gapX?: SpacingValue;
  /** The gapY property of the flex container. */
  gapY?: SpacingValue;
}

// Map the DS's Radix-style shorthand tokens onto real CSS values.
const JUSTIFY_CSS: Record<FlexJustify, CSSProperties["justifyContent"]> = {
  start: "flex-start",
  center: "center",
  end: "flex-end",
  between: "space-between",
};

const ALIGN_CSS: Record<FlexAlign, CSSProperties["alignItems"]> = {
  start: "flex-start",
  center: "center",
  end: "flex-end",
  baseline: "baseline",
  stretch: "stretch",
};

const FLEX_ONLY_KEYS = [
  "flex",
  "direction",
  "wrap",
  "justify",
  "align",
  "grow",
  "shrink",
  "gap",
  "gapX",
  "gapY",
] as const;

/**
 * A primitive layout component that provides flexible box functionality with comprehensive flex controls. Extends all Box component properties.
 * ## Features
 * - All Box component properties (padding, margin, dimensions, etc.)
 * - Flex direction control (row, column, and their reverse variants)
 * - Flex wrapping behavior
 * - Justify content and align items controls
 * - Configurable gap between items
 * - Grow and flex controls
 * - Shrink control for flex items
 * - Inline flex option
 * - Full width/height options
 *
 * ### Usage
 * ```tsx
 * <Flex
 *   direction="row"
 *   justify="between"
 *   align="center"
 *   gap="md"
 *   grow="1"
 *   shrink="0"
 *   p="lg"
 * >
 *   <div>Item 1</div>
 *   <div>Item 2</div>
 *   <div>Item 3</div>
 * </Flex>
 * ```
 *
 * Renders a plain `div` with inline flex styles (no `@radix-ui/themes`), keeping all
 * spacing scoped to the `.ltx-io` subtree.
 */
export const Flex = forwardRef<HTMLDivElement, FlexProps>((props, ref) => {
  const { style, rest } = extractLayout(props);

  const flexStyle: CSSProperties = { display: props.display ?? "flex", ...style };
  if (props.direction) flexStyle.flexDirection = props.direction;
  if (props.wrap) flexStyle.flexWrap = props.wrap;
  if (props.justify) flexStyle.justifyContent = JUSTIFY_CSS[props.justify];
  if (props.align) flexStyle.alignItems = ALIGN_CSS[props.align];
  if (props.flex !== undefined) flexStyle.flex = props.flex;
  if (props.grow !== undefined) flexStyle.flexGrow = props.grow;
  if (props.shrink !== undefined) flexStyle.flexShrink = props.shrink;

  const gap = convertSpacing(props.gap);
  const gapX = convertSpacing(props.gapX);
  const gapY = convertSpacing(props.gapY);
  if (gap !== undefined) flexStyle.gap = gap;
  if (gapX !== undefined) flexStyle.columnGap = gapX;
  if (gapY !== undefined) flexStyle.rowGap = gapY;

  // Flex-only props must not leak onto the DOM element.
  for (const key of FLEX_ONLY_KEYS) {
    delete rest[key];
  }

  return (
    <div
      ref={ref}
      className={clsx(styles.flex, props.className)}
      style={flexStyle}
      {...rest}
    >
      {props.children}
    </div>
  );
});

Flex.displayName = "Flex";
