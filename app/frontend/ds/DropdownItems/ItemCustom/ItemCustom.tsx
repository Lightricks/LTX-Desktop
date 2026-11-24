import { forwardRef } from "react";

export interface ItemCustomProps {
  children: React.ReactNode;
}

/**
 * ItemCustom is a visual-only sub-component for rendering arbitrary content
 * within a `DropdownMenu`. It behaves like `ItemHeadline` -- non-interactive
 * from the menu's perspective, so the content itself is responsible for its
 * own interaction and styling.
 */
export const ItemCustom = forwardRef<HTMLDivElement, ItemCustomProps>(function ItemCustom(
  { children },
  ref,
) {
  return <div ref={ref}>{children}</div>;
});

ItemCustom.displayName = "ItemCustom";
