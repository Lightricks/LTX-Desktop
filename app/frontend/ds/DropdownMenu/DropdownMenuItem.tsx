import * as RadixDropdownMenu from "@radix-ui/react-dropdown-menu";
import React from "react";

import { ItemAction } from "@ds/DropdownItems/ItemAction/ItemAction";
import { ItemActionInformative } from "@ds/DropdownItems/ItemActionInformative/ItemActionInformative";
import { ItemCustom } from "@ds/DropdownItems/ItemCustom/ItemCustom";
import { ItemHeadline } from "@ds/DropdownItems/ItemHeadline/ItemHeadline";
import { ItemUserDetails } from "@ds/DropdownItems/ItemUserDetails/ItemUserDetails";

import styles from "./DropdownMenuItem.module.scss";

type DropdownMenuItemType =
  | typeof ItemAction
  | typeof ItemActionInformative
  | typeof ItemCustom
  | typeof ItemUserDetails
  | typeof ItemHeadline;

/**
 * Interface defining the structure of a menu item in the dropdown
 * @interface MenuItem
 */
export interface MenuItem {
  /** React element to render as the menu item content */
  render: React.ReactElement<
    {
      isDisabled?: boolean;
      isVisualOnly?: boolean;
      isSelectable?: boolean;
      [key: string]: unknown;
    },
    DropdownMenuItemType
  >;
  /** Optional submenu configuration */
  subMenu?: {
    /** Controls whether submenu items have a selectable state.
     * When true, adds visual space on the left side of items for selection indicators.
     */
    isSelectableMenu?: boolean;
    /** Minimum width of the submenu */
    minWidth?: string;
    /** Maximum height of the submenu */
    maxHeight?: string;
    /** Maximum width of the submenu */
    maxWidth?: string;
    /** Array of submenu items */
    items: MenuItem[];
    /** Position of the submenu relative to its trigger */
    side?: RadixDropdownMenu.DropdownMenuContentProps["side"];
    /** Alignment of the submenu relative to its trigger */
    align?: RadixDropdownMenu.DropdownMenuContentProps["align"];
    /**
     * Fires when the submenu opens or closes. Useful for analytics — the
     * parent `ItemAction`'s `onClick` is NOT invoked for submenu triggers
     * (Radix handles open/close directly), so use this callback to emit a
     * "trigger clicked" and/or "view presented/dismissed" pair.
     */
    onOpenChange?: (open: boolean) => void;
  };
  /** Whether to show a separator above the item */
  hasUpperSeparator?: boolean;
  /** Whether the item is disabled */
  isDisabled?: boolean;
  /** Whether the item is for visual display only (non-interactive) */
  isVisualOnly?: boolean;
  /** When false, selecting this item does not close the dropdown. Defaults to true. */
  closeOnSelect?: boolean;
}

interface DropdownMenuItemProps {
  /** The menu item configuration */
  item: MenuItem;
  /** Whether the menu items are checkable */
  isSelectableMenu?: boolean;
}

/**
 * A dropdown menu item component that can render both simple items and nested submenus.
 * Built on top of Radix UI's DropdownMenu primitives.
 *
 * @component
 * @example
 * ```tsx
 * const menuItem: MenuItem = {
 *   render: <ItemAction>Click me</ItemAction>,
 *   isDisabled: false
 * };
 *
 * <DropdownMenuItem item={menuItem} />
 * ```
 */
const SUBMENU_GAP = 4;

export function DropdownMenuItem(props: DropdownMenuItemProps) {
  const { item, isSelectableMenu } = props;

  const isActionItem =
    item.render.type === ItemAction ||
    item.render.type === ItemActionInformative;
  const itemDisabled = item?.isDisabled || item.render.props?.isDisabled;
  const isVisualOnly =
    item?.isVisualOnly || item.render.props?.isVisualOnly || !isActionItem;
  const isItemSelectable = isSelectableMenu || item.render.props?.isSelectable;

  if (item.subMenu) {
    return (
      <div className={styles.subMenuContainer}>
        <Separator shouldShow={item.hasUpperSeparator} />
        <RadixDropdownMenu.Sub onOpenChange={item.subMenu.onOpenChange}>
          <RadixDropdownMenu.SubTrigger
            className={styles.itemWrapper}
            disabled={itemDisabled || isVisualOnly}
          >
            {React.cloneElement(item.render, {
              shouldShowSubmenuArrow: true,
              isSelectable: isItemSelectable,
            })}
          </RadixDropdownMenu.SubTrigger>
          <RadixDropdownMenu.Portal>
            <RadixDropdownMenu.SubContent
              data-testid="dropdown-menu-submenu"
              className={`${styles.menuContent} ${styles.subMenuContent}`}
              sideOffset={SUBMENU_GAP}
              collisionPadding={20}
              style={{
                ["--submenu-gap" as string]: `${SUBMENU_GAP}px`,
                minWidth: item.subMenu?.minWidth,
                maxWidth: item.subMenu?.maxWidth,
                maxHeight: item.subMenu?.maxHeight,
              }}
            >
              <div className={styles.menuContentInner}>
                {item.subMenu.items.map((subItem, index) => (
                  <DropdownMenuItem
                    data-testid="dropdown-menu-submenu-item"
                    key={`sub-item-${index}`}
                    item={subItem}
                    isSelectableMenu={item.subMenu?.isSelectableMenu}
                  />
                ))}
              </div>
            </RadixDropdownMenu.SubContent>
          </RadixDropdownMenu.Portal>
        </RadixDropdownMenu.Sub>
      </div>
    );
  }

  return (
    <>
      <Separator shouldShow={item.hasUpperSeparator} />
      <RadixDropdownMenu.Item
        data-testid="dropdown-menu-item"
        className={styles.itemWrapper}
        disabled={itemDisabled || isVisualOnly}
        onSelect={(event) => {
          if (item.closeOnSelect === false) {
            event.preventDefault();
          }
          if (item.render.props.onClick) {
            (item.render.props.onClick as (event: Event) => void)(event);
          }
        }}
      >
        <div className={styles.item}>
          {React.cloneElement(item.render, {
            isSelectable: isItemSelectable,
            onClick: undefined,
          })}
        </div>
      </RadixDropdownMenu.Item>
    </>
  );
}

function Separator({ shouldShow }: { shouldShow?: boolean }) {
  if (!shouldShow) return null;
  return (
    <RadixDropdownMenu.Separator
      className={styles.separator}
      data-testid="dropdown-menu-separator"
    />
  );
}

DropdownMenuItem.displayName = "DropdownMenuItem";
