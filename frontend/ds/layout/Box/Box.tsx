import { ElementType, forwardRef } from "react";

import { LayoutProps } from "../types";
import { extractLayout } from "../utils";

/**
 * A primitive layout component for creating layout containers. Provides basic styling and layout capabilities.
 *
 * ## Features
 * - Customizable HTML element type (e.g., `div`, `span`)
 * - Display property control (e.g., `block`, `inline-block`)
 * - Configurable spacing (e.g. padding and margin)
 * - Extends all LayoutProps for additional layout control
 *
 * ### Usage
 * ```tsx
 * <Box
 *   display="inline-block"
 *   width="48px"
 *   height="48px"
 *   p="lg"
 * >
 *   Content goes here
 * </Box>
 * ```
 *
 * Renders a plain element with inline styles (no `@radix-ui/themes`), keeping all
 * spacing scoped to the `.ltx-io` subtree.
 */
export const Box = forwardRef<HTMLDivElement, LayoutProps>((props, ref) => {
  const { as = "div", className, children } = props;
  const { style, rest } = extractLayout(props);
  const Element = as as ElementType;

  return (
    <Element ref={ref} className={className} style={style} {...rest}>
      {children}
    </Element>
  );
});

Box.displayName = "Box";
