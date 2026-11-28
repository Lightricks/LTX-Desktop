export const SPACING_VALUES = [
  "no",
  "3xs",
  "xxs",
  "xs",
  "sm",
  "md",
  "lg",
  "xl",
  "xxl",
  "3xl",
  "4xl",
] as const;

export const SPACING_MAP = {
  auto: "auto",
  no: "0",
  "3xs": "1px",
  xxs: "2px",
  xs: "4px",
  sm: "6px",
  md: "8px",
  lg: "16px",
  xl: "24px",
  xxl: "32px",
  "3xl": "40px",
  "4xl": "48px",
} as const;

export const SPACING_PROPS = [
  "p",
  "m",
  "px",
  "py",
  "pt",
  "pr",
  "pb",
  "pl",
  "mx",
  "my",
  "mt",
  "mr",
  "mb",
  "ml",
] as const;

export const RADIUS_VALUES = [
  "3xs",
  "xxs",
  "xs",
  "sm",
  "md",
  "lg",
  "xl",
  "xxl",
  "round",
] as const;

export const RADIUS_MAP = {
  "3xs": "1px",
  xxs: "2px",
  xs: "4px",
  sm: "6px",
  md: "8px",
  lg: "12px",
  xl: "16px",
  xxl: "24px",
  round: "999px",
} as const;

export const ELEVATION_VALUES = ["low", "medium", "high"] as const;
