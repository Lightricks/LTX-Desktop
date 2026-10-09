import { ELEVATION_VALUES, RADIUS_VALUES, SPACING_VALUES } from "../styles/constants";

export const AS_VALUES = ["div", "span"] as const;

export const BOX_DISPLAY = ["none", "inline", "inline-block", "block"] as const;
export type BoxDisplay = (typeof BOX_DISPLAY)[number];

export const FLEX_DISPLAY = ["none", "inline-flex", "flex"] as const;
export type FlexDisplay = (typeof FLEX_DISPLAY)[number];

export const LAYOUT_POSITION = ["relative", "absolute", "fixed", "sticky"] as const;
export type LayoutPosition = (typeof LAYOUT_POSITION)[number];

export type LayoutDisplay = BoxDisplay | FlexDisplay;
export type SpacingValue = (typeof SPACING_VALUES)[number];
export type RadiusValue = (typeof RADIUS_VALUES)[number];
export type ElevationValue = (typeof ELEVATION_VALUES)[number];
export type AsValue = (typeof AS_VALUES)[number];
export type MarginValue = SpacingValue | "auto";

export interface LayoutProps extends React.HTMLAttributes<HTMLElement> {
  /** The position property of the layout component. */
  position?: LayoutPosition;
  /** The display property of the layout component. */
  display?: LayoutDisplay;
  /** HTML element to render as. Can be 'div' or 'span' */
  as?: AsValue;
  /** When true, will not render its own element but instead pass styles to its child */
  asChild?: boolean;
  /** Additional CSS classes to apply */
  className?: string;
  /** Inline styles to apply */
  style?: React.CSSProperties;
  /** Child elements */
  children?: React.ReactNode;
  /** CSS width property */
  width?: string;
  /** CSS height property */
  height?: string;
  /** CSS min-width property */
  minWidth?: string;
  /** CSS max-width property */
  maxWidth?: string;
  /** CSS min-height property */
  minHeight?: string;
  /** CSS max-height property */
  maxHeight?: string;

  /** Padding on all sides */
  p?: SpacingValue;
  /** Horizontal padding (left and right) */
  px?: SpacingValue;
  /** Vertical padding (top and bottom) */
  py?: SpacingValue;
  /** Padding top */
  pt?: SpacingValue;
  /** Padding right */
  pr?: SpacingValue;
  /** Padding bottom */
  pb?: SpacingValue;
  /** Padding left */
  pl?: SpacingValue;

  /** Margin on all sides */
  m?: MarginValue;
  /** Horizontal margin (left and right) */
  mx?: MarginValue;
  /** Vertical margin (top and bottom) */
  my?: MarginValue;
  /** Margin top */
  mt?: MarginValue;
  /** Margin right */
  mr?: MarginValue;
  /** Margin bottom */
  mb?: MarginValue;
  /** Margin left */
  ml?: MarginValue;
}
