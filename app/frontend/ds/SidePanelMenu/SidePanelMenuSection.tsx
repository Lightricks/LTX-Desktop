import clsx from "clsx";
import { forwardRef } from "react";

import { Text, type TextSize, type TextVariant } from "@ds/Text/Text";
import { Flex } from "@ds/layout/Flex/Flex";
import {
  AccordionArrow,
  AccordionMenu,
} from "@ds/AccordionMenu/AccordionMenu";

import { Button } from "../Button/Button";
import { type ButtonBaseProps } from "../Button/types";

import styles from "./SidePanelMenuSection.module.scss";

export type SidePanelMenuSectionButtonItem = Omit<ButtonBaseProps, "size"> & {
  type?: "button";
  label: string;
  leftIcon: React.ReactNode;
  rightIcon?: React.ReactNode;
  shouldShowSeparator?: boolean;
};

export type SidePanelMenuSectionCustomItem = {
  type: "custom";
  node: React.ReactNode;
};

export type SidePanelMenuSectionAccordionItem = {
  type: "accordion";
  label: string;
  items: SidePanelMenuSectionItem[];
  leftIcon?: React.ReactNode;
};

export type SidePanelMenuSectionItem =
  | SidePanelMenuSectionButtonItem
  | SidePanelMenuSectionCustomItem
  | SidePanelMenuSectionAccordionItem;

export interface SidePanelMenuSection {
  items: SidePanelMenuSectionItem[];
  label?: string;
  labelVariant?: TextVariant;
  labelSize?: TextSize;
  rightSideElement?: React.ReactNode;
  shouldShowSeparator?: boolean;
  separatorClassName?: string;
  labelClassName?: string;
}

export const SidePanelMenuSection = forwardRef<HTMLDivElement, SidePanelMenuSection>(
  (props, ref): JSX.Element => {
    const { items, label, labelVariant, labelSize, rightSideElement, shouldShowSeparator, separatorClassName, labelClassName } = props;

    return (
      <Flex ref={ref} direction="column" gap="lg">
        {shouldShowSeparator && <Separator className={separatorClassName} />}
        {label && (
          <Text variant={labelVariant ?? "labelCaps"} size={labelSize ?? "md"} className={clsx(styles.sectionLabel, labelClassName)}>
            {label}
          </Text>
        )}
        {rightSideElement && <div>{rightSideElement}</div>}

        <ul className={styles.menuItems}>
          {items.map((item, index) => {
            const key =
              item.type === "custom" ? `custom-${index}` : `${item.label}-${index}`;

            return (
              <li key={key}>
                <SideMenuSectionItem item={item} />
              </li>
            );
          })}
        </ul>
      </Flex>
    );
  },
);

SidePanelMenuSection.displayName = "SidePanelMenuSection";

function Separator({ className }: { className?: string }) {
  return <div className={clsx(styles.separator, className)} />;
}

function SideMenuSectionItem({ item }: { item: SidePanelMenuSectionItem }) {
  switch (item.type) {
    case "custom":
      return item.node;
    case "accordion":
      return <SidePanelMenuSectionAccordionItem item={item} />;
    case "button":
    case undefined: {
      /* This is the only way for now to override the Button padding without using !important in the CSS.
              Waiting for designers to create a special button variant with this of configuration. Then this can be removed. */
      const { shouldShowSeparator, ...buttonProps } = item;
      return (
        <>
          {shouldShowSeparator && <Separator />}
          <Button hierarchy="plain" className={styles.menuItemButton} {...buttonProps} />
        </>
      );
    }
  }
}

function SidePanelMenuSectionAccordionItem({
  item,
}: {
  item: SidePanelMenuSectionAccordionItem;
}) {
  return (
    <AccordionMenu
      trigger={
        <Button
          hierarchy="plain"
          leftIcon={item.leftIcon}
          label={item.label}
          rightIcon={<AccordionArrow />}
          className={clsx(styles.menuItemButton, styles.accordionTrigger)}
        />
      }
      headerClassName={styles.accordionHeader}
      contentClassName={styles.accordionContent}
    >
      {item.items.map((nestedItem, index) => {
        const key =
          nestedItem.type === "custom"
            ? `custom-${index}`
            : `${nestedItem.label}-${index}`;
        return (
          <div key={key} className={styles.accordionItem}>
            <SideMenuSectionItem item={nestedItem} />
          </div>
        );
      })}
    </AccordionMenu>
  );
}
