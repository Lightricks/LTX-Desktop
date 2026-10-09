import { CSSProperties } from "react";

import { SPACING_MAP } from "../styles/constants";

import { LayoutProps, MarginValue, SpacingValue } from "./types";

export const convertSpacing = (value: SpacingValue | MarginValue | undefined) => {
  if (value && value in SPACING_MAP) {
    return SPACING_MAP[value as keyof typeof SPACING_MAP];
  }
  return undefined;
};

// Layout props consumed into inline styles. Everything else on `LayoutProps`
// (it extends `React.HTMLAttributes`, e.g. onClick / id / data-* / aria-*) passes
// through untouched to the rendered element.
const LAYOUT_PROP_KEYS = new Set<string>([
  "position",
  "display",
  "as",
  "asChild",
  "className",
  "style",
  "children",
  "width",
  "height",
  "minWidth",
  "maxWidth",
  "minHeight",
  "maxHeight",
  "p",
  "px",
  "py",
  "pt",
  "pr",
  "pb",
  "pl",
  "m",
  "mx",
  "my",
  "mt",
  "mr",
  "mb",
  "ml",
]);

/**
 * Builds an inline `style` from the layout props and returns the remaining
 * HTML attributes (`rest`) to spread onto the element. Replaces the previous
 * `@radix-ui/themes` Box/Flex, whose spacing was applied via a global
 * `layout.css` utility sheet that leaked outside the `.ltx-io` scope.
 */
export function extractLayout(props: LayoutProps): {
  style: CSSProperties;
  rest: Record<string, unknown>;
} {
  const style: CSSProperties = {};

  if (props.position) style.position = props.position;
  if (props.display) style.display = props.display;
  if (props.width !== undefined) style.width = props.width;
  if (props.height !== undefined) style.height = props.height;
  if (props.minWidth !== undefined) style.minWidth = props.minWidth;
  if (props.maxWidth !== undefined) style.maxWidth = props.maxWidth;
  if (props.minHeight !== undefined) style.minHeight = props.minHeight;
  if (props.maxHeight !== undefined) style.maxHeight = props.maxHeight;

  // Specific sides win over the axis (px/py) and all-sides (p) shorthands.
  const p = convertSpacing(props.p);
  const px = convertSpacing(props.px);
  const py = convertSpacing(props.py);
  const paddingTop = convertSpacing(props.pt) ?? py ?? p;
  const paddingRight = convertSpacing(props.pr) ?? px ?? p;
  const paddingBottom = convertSpacing(props.pb) ?? py ?? p;
  const paddingLeft = convertSpacing(props.pl) ?? px ?? p;
  if (paddingTop !== undefined) style.paddingTop = paddingTop;
  if (paddingRight !== undefined) style.paddingRight = paddingRight;
  if (paddingBottom !== undefined) style.paddingBottom = paddingBottom;
  if (paddingLeft !== undefined) style.paddingLeft = paddingLeft;

  const m = convertSpacing(props.m);
  const mx = convertSpacing(props.mx);
  const my = convertSpacing(props.my);
  const marginTop = convertSpacing(props.mt) ?? my ?? m;
  const marginRight = convertSpacing(props.mr) ?? mx ?? m;
  const marginBottom = convertSpacing(props.mb) ?? my ?? m;
  const marginLeft = convertSpacing(props.ml) ?? mx ?? m;
  if (marginTop !== undefined) style.marginTop = marginTop;
  if (marginRight !== undefined) style.marginRight = marginRight;
  if (marginBottom !== undefined) style.marginBottom = marginBottom;
  if (marginLeft !== undefined) style.marginLeft = marginLeft;

  // Consumer-provided inline style wins over the derived spacing/layout.
  Object.assign(style, props.style);

  const rest: Record<string, unknown> = {};
  for (const key of Object.keys(props)) {
    if (!LAYOUT_PROP_KEYS.has(key)) {
      rest[key] = (props as Record<string, unknown>)[key];
    }
  }

  return { style, rest };
}
