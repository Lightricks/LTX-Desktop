import * as RadixDropdownMenu from "@radix-ui/react-dropdown-menu";
import { clsx } from "clsx";
import { forwardRef, useState } from "react";

import styles from "./DropdownMenu.module.scss";
import { DropdownMenuItem, MenuItem } from "./DropdownMenuItem";

export interface DropdownMenuProps
  extends RadixDropdownMenu.DropdownMenuContentProps {
  /** Array of menu items to be rendered */
  items: MenuItem[];
  /** Minimum width of the dropdown menu. Default is "160px" */
  minWidth?: string;
  /** Maximum width of the dropdown menu. Default is "auto" */
  maxWidth?: string;
  /** Maximum height of the dropdown menu. Default is "auto" */
  maxHeight?: string;
  /** Offset from the trigger element. Default is 10 */
  sideOffset?: number;
  /** Padding to maintain when handling collisions. Default is 20 */
  collisionPadding?: number;
  /** Control the open state of the dropdown. If not provided, the dropdown will be uncontrolled */
  isOpen?: boolean;
  /** Whether the dropdown menu should be modal. Default is true */
  shouldBeModal?: boolean;
  /** Callback fired when the menu open state changes. */
  onMenuStateChange?: (isOpen: boolean) => void;
  /** Whether the dropdown menu should be disabled. Default is false */
  disabled?: boolean;
  /** Optional DOM node to contain the portaled menu content. */
  portalContainer?: HTMLElement | null;
}

/**
 * DropdownMenu component provides a menu that appears when triggered by a button.
 * Built on top of Radix UI's DropdownMenu for accessibility and keyboard navigation.
 *
 * @see {@link DropdownMenuProps} for available props
 *
 * @example
 * ```tsx
 * <DropdownMenu
 *   items={[
 *     { type: 'action', label: 'Edit', onClick: () => {} },
 *     { type: 'action', label: 'Delete', onClick: () => {} }
 *   ]}
 * >
 *   <Button label="Open Menu" />
 * </DropdownMenu>
 * ```
 */
export const DropdownMenu = forwardRef<HTMLDivElement, DropdownMenuProps>(
  (props, ref) => {
    const {
      items,
      minWidth = "160px",
      maxWidth,
      maxHeight,
      children,
      sideOffset = 4,
      collisionPadding = 20,
      isOpen: controlledIsOpen,
      shouldBeModal = true,
      onMenuStateChange,
      className,
      disabled,
      portalContainer,
      ...rest
    } = props;
    const [uncontrolledIsOpen, setUncontrolledIsOpen] = useState(false);
    const isControlled = controlledIsOpen !== undefined;
    const isOpen = isControlled ? controlledIsOpen : uncontrolledIsOpen;

    const handleOpenChange = (valueIsOpen: boolean) => {
      if (!isControlled) {
        setUncontrolledIsOpen(valueIsOpen);
      }
      onMenuStateChange?.(valueIsOpen);
    };

    return (
      <RadixDropdownMenu.Root
        open={isOpen}
        onOpenChange={handleOpenChange}
        modal={shouldBeModal}
      >
        <RadixDropdownMenu.Trigger
          asChild
          data-state={isOpen ? "active" : "enabled"}
          disabled={disabled}
        >
          {children}
        </RadixDropdownMenu.Trigger>
        <RadixDropdownMenu.Portal container={portalContainer}>
          <RadixDropdownMenu.Content
            data-testid="dropdown-menu"
            ref={ref}
            className={clsx(styles.menuContent, className)}
            sideOffset={sideOffset}
            collisionPadding={collisionPadding}
            style={{ minWidth, maxWidth, maxHeight }}
            {...rest}
          >
            <div className={styles.menuContentInner}>
              {items.map((item, index) => (
                <DropdownMenuItem key={`item-${index}`} item={item} />
              ))}
            </div>
          </RadixDropdownMenu.Content>
        </RadixDropdownMenu.Portal>
      </RadixDropdownMenu.Root>
    );
  },
);

DropdownMenu.displayName = "DropdownMenu";
